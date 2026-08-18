"""PPO trainer H-PPO + implementasi PolicyTrainer/OnlinePolicyTrainer (titik-colok
training.py). GAE + clipped surrogate. Selaras Spesifikasi §8-§9.
"""
import json
import logging
import math
import random
import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.rollout import collect_rollout, InferenceAgent, RLRolloutAgent, _gini


def _make_logger(verbose: bool):
    """Logger sederhana ber-timestamp (menggantikan print mentah, level toggle)."""
    logger = logging.getLogger("marl_spklu.rl")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger


def compute_gae(transitions, gamma=0.99, lam=0.95, avg_reward=False, r_bar=0.0,
                max_step_gap=None):
    """Return (returns, advantages). PENTING: `transitions` HARUS sudah terurut menurut
    waktu KEPUTUSAN (`t.step`), bukan urutan penyelesaian sesi (`resolved`) -- lihat
    PPOTrainer.update() yang men-sort sebelum memanggil fungsi ini (perbaikan C1).

    avg_reward=True -> differential return (Sutton & Barto Ch.10) untuk continuing task
    tanpa reset: delta_t = (r_t - r_bar) + V(s_{t+1}) - V(s_t), TANPA diskon gamma (gamma=1
    implisit -- sub-goal jauh di masa depan mis. Gini tak boleh di-discount lebih dari yang
    dekat). r_bar adalah estimasi laju reward rata-rata, dikelola PPOTrainer lintas chunk.

    max_step_gap (perbaikan C4, "time-distance gating"): reward di sistem ini TERTUNDA
    (baru diemit saat on_charge_complete, bisa berjam-jam setelah keputusan) -- transisi
    ke-t+1 dlm daftar `resolved` sering BUKAN keadaan sesaat setelah transisi ke-t secara
    waktu-sungguhan (bisa berjarak >10 langkah/>2 jam, diverifikasi empiris: rata2 ~2
    langkah, maks 12, 45% pasangan berjarak >1 langkah). Bootstrap V(s_{t+1}) lintas celah
    sebesar itu melanggar asumsi Markov TD -- mencemari advantage dgn noise struktural
    (explained_variance sempat -300rb, entropi aktor tak pernah menajam meski di-overfit
    batch tetap). Bila `max_step_gap` diberi & jarak `t.step` ke transisi berikutnya
    melampauinya, rantai bootstrap DIPUTUS di titik itu (nonterminal=0, spt batas episode
    lokal) alih-alih dipaksa menyambung ke keadaan yg tak relevan.

    KRITIK GANDA (adopsi MASTER): seluruh besaran di sini bersifat VEKTOR sepanjang K
    aliran reward (lihat STREAM_* di rollout.py). Untuk K=1 hasilnya identik secara
    numerik dgn versi skalar sebelumnya -- tiap aliran di-bootstrap sendiri memakai
    V^k(s) miliknya, tanpa interaksi antar-aliran. `r_bar` juga vektor (K,).
    Return: (returns, advantages) masing-masing berbentuk (T, K)."""
    # PRESISI: rekursi dijalankan di float64 (persis versi skalar sebelum refaktor, yg
    # memakai Python float), hasil akhir baru diturunkan ke float32 saat disimpan --
    # memakai float32 di dalam rekursi menggeser hasil ~1e-7/langkah yg teramplifikasi
    # kaotik lintas ratusan update (terverifikasi pada gerbang regresi).
    T = len(transitions)
    K = int(np.asarray(transitions[0].value).size)
    adv = np.zeros((T, K), dtype=np.float64)
    r_bar_v = np.broadcast_to(np.asarray(r_bar, dtype=np.float64).ravel(), (K,))
    last = np.zeros(K, dtype=np.float64)
    zero_v = np.zeros(K, dtype=np.float64)
    for t in reversed(range(T)):
        v = np.asarray(transitions[t].value, dtype=np.float64).ravel()
        nonterminal = 0.0 if transitions[t].done else 1.0
        if nonterminal and max_step_gap is not None and (t + 1) < T:
            gap = transitions[t + 1].step - transitions[t].step
            if gap > max_step_gap:
                nonterminal = 0.0
        v_next = (np.asarray(transitions[t + 1].value, dtype=np.float64).ravel()
                  if (t + 1 < T) else zero_v)
        r = transitions[t].reward_vec(K).astype(np.float64).ravel()
        if avg_reward:
            delta = (r - r_bar_v) + v_next * nonterminal - v
            last = delta + lam * nonterminal * last
        else:
            delta = r + gamma * v_next * nonterminal - v
            last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    # Penurunan ke float32 dilakukan di SINI saja (persis titik yang sama dgn versi lama:
    # `adv` lama adalah array float32 yang diisi dari rekursi float64).
    adv = adv.astype(np.float32)
    values = np.stack([np.asarray(tr.value, dtype=np.float32).ravel()
                       for tr in transitions])          # (T,K)
    returns = adv + values
    return returns, adv


class PPOTrainer:
    def __init__(self, policy, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=10, minibatch=64, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 target_kl=0.03, avg_reward=False, r_bar_lr=0.01, max_step_gap=4,
                 beta_mode="fixed", beta_sigma=0.1):
        self.policy = policy
        # --- Kritik ganda (adopsi MASTER) ---
        # K diambil dari policy; K=1 => seluruh jalur di bawah runtuh ke perilaku lama.
        self.n_critics = int(getattr(policy, "n_critics", 1))
        # beta_mode: cara menentukan bobot penggabungan advantage antar-aliran.
        #   "fixed"    -> bobot seragam 1/K (tahap M1: mengisolasi efek pemisahan kritik
        #                 + normalisasi per-aliran SAJA, tanpa bobot dinamis)
        #   "gap_ratio"-> ala MASTER (tahap M2): objektif yang paling JAUH dari nilai
        #                 terbaiknya yang pernah teramati mendapat bobot lebih besar.
        #                 g^k = (R*^k - R^k)/|R*^k|, beta = softmax(g^k / sigma).
        self.beta_mode = str(beta_mode)
        self.beta_sigma = float(beta_sigma)
        # Rekam return terbaik per aliran (proksi R*_optimal utk gap-ratio), persisten
        # lintas update -- MASTER memakai Q*_optimal yg juga tak tersedia analitis.
        self._ret_best = np.full(self.n_critics, -np.inf, dtype=np.float64)
        self._last_beta = np.full(self.n_critics, 1.0 / self.n_critics, dtype=np.float64)
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        # Early-stop epoch bila policy menyimpang terlalu jauh (cegah update meledak;
        # KL besar = tanda lr/epochs terlalu agresif). None untuk menonaktifkan.
        self.target_kl = target_kl
        # Continuing task tanpa reset (IV.1.7): average-reward differential return
        # menggantikan discounted return. r_bar (estimasi laju reward) dikelola sbg EMA
        # PERSISTEN lintas chunk/update -- bukan dihitung ulang per-batch.
        self.avg_reward = bool(avg_reward)
        self.r_bar_lr = float(r_bar_lr)
        # r_bar kini VEKTOR (K,) -- satu laju reward rata-rata per aliran (K=1 identik
        # dgn skalar sebelumnya).
        self.r_bar = np.zeros(self.n_critics, dtype=np.float64)
        # Perbaikan C4 (time-distance gating, lihat compute_gae docstring): reward
        # tertunda (on_charge_complete) bikin urutan `resolved` TIDAK selaras waktu
        # keputusan sungguhan -- default 4 langkah (~1 jam @ dt=15menit) memotong rantai
        # bootstrap GAE saat celah waktu ke transisi berikutnya terlalu jauh utk
        # dianggap "penerus Markov" yg valid. None utk menonaktifkan (perilaku lama).
        self.max_step_gap = max_step_gap

    def _compute_beta(self, returns):
        """Bobot penggabungan advantage antar-aliran, bentuk (K,), berjumlah 1.

        `fixed`     -> seragam 1/K (tahap M1).
        `gap_ratio` -> adaptasi *dynamic gradient re-weighting* MASTER (tahap M2):
                       g^k = (R*^k - R^k) / (|R*^k| + eps), beta = softmax(g^k / sigma).
                       Objektif yang paling TERTINGGAL dari nilai terbaik yang pernah
                       teramati mendapat bobot lebih besar. MASTER memakai Q*_optimal
                       yang juga tak tersedia analitis; di sini proksinya adalah return
                       rata-rata TERBAIK yang pernah teramati per aliran (running max,
                       persisten lintas update)."""
        K = self.n_critics
        if K == 1:
            return np.ones(1, dtype=np.float64)
        ret_mean = np.asarray(returns, dtype=np.float64).mean(axis=0)   # (K,)
        self._ret_best = np.maximum(self._ret_best, ret_mean)
        if self.beta_mode != "gap_ratio":
            return np.full(K, 1.0 / K, dtype=np.float64)
        gap = (self._ret_best - ret_mean) / (np.abs(self._ret_best) + 1e-8)
        gap = np.clip(gap, 0.0, 10.0)          # jaga agar softmax tak meledak
        z = gap / max(self.beta_sigma, 1e-8)
        z -= z.max()                            # stabilitas numerik
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def update(self, transitions):
        if len(transitions) < 2:
            return {"loss": 0.0, "n": len(transitions)}
        # Perbaikan C1 (re-sort by decision time): `transitions` datang dlm urutan
        # PENYELESAIAN sesi (`resolved`), BUKAN urutan keputusan diambil -- compute_gae
        # butuh urutan waktu keputusan (`t.step`) supaya t+1 benar2 "penerus" t. Re-assign
        # `transitions` lokal spy SEMUA tensor batch di bawah (obs_b dst.) ikut selaras.
        transitions = sorted(transitions, key=lambda t: t.step)
        if self.avg_reward:
            # Per-aliran (K,) -- utk K=1 identik dgn rata-rata skalar sebelumnya.
            batch_mean_r = np.mean(
                np.stack([t.reward_vec(self.n_critics).ravel() for t in transitions]),
                axis=0)
            self.r_bar += self.r_bar_lr * (batch_mean_r - self.r_bar)
            returns, adv = compute_gae(transitions, self.gamma, self.lam,
                                       avg_reward=True, r_bar=self.r_bar,
                                       max_step_gap=self.max_step_gap)
        else:
            returns, adv = compute_gae(transitions, self.gamma, self.lam,
                                       max_step_gap=self.max_step_gap)
        # --- Normalisasi advantage PER ALIRAN (kunci adopsi kritik-ganda) ---
        # Tiap aliran dinormalisasi memakai mean/std MILIKNYA SENDIRI (axis=0), sehingga
        # ketimpangan skala antar-objektif (13,7x, Rumusan_Masalah_Teknis_RL.md §3.2)
        # HILANG scr struktural -- tak perlu lagi dikalibrasi lewat bobot reward.
        # Utk K=1 ini identik dgn normalisasi skalar sebelumnya.
        adv = (adv - adv.mean(axis=0, keepdims=True)) / (adv.std(axis=0, keepdims=True) + 1e-8)

        # --- Bobot beta penggabungan antar-aliran ---
        beta = self._compute_beta(returns)
        self._last_beta = beta
        adv_combined = adv @ beta.astype(np.float32)      # (T,K)@(K,) -> (T,)

        obs_b = torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32)
        critic_obs_b = torch.as_tensor(np.stack([t.critic_obs for t in transitions]), dtype=torch.float32)
        hist_b = torch.as_tensor(np.stack([t.hist for t in transitions]), dtype=torch.float32)
        mask_b = torch.as_tensor(np.stack([t.mask for t in transitions]), dtype=torch.bool)
        chosen_b = torch.as_tensor(np.stack([t.chosen_indices for t in transitions]), dtype=torch.long)
        n_rec_b = torch.as_tensor(np.array([t.n_rec for t in transitions]), dtype=torch.long)
        old_logp = torch.as_tensor(np.array([t.logp for t in transitions]), dtype=torch.float32)
        ret_b = torch.as_tensor(returns, dtype=torch.float32)        # (B,K) utk value loss
        adv_b = torch.as_tensor(adv_combined, dtype=torch.float32)   # (B,)  utk policy loss

        # PDQN kontinu (opsional): policy dgn modul preferensi (lihat
        # pdqn_continuous_policy.py) butuh tensor riwayat (a_hat,a) tambahan -- H-PPO biasa
        # (HPPOPolicy) tak punya `pref_lstm`, jalur ini murni tak dieksekusi untuknya.
        use_pref = hasattr(self.policy, "pref_lstm")
        pref_hist_b = (torch.as_tensor(np.stack([t.pref_hist for t in transitions]), dtype=torch.float32)
                      if use_pref else None)

        # Lengan stasiun-sebagai-agen (MasterBiddingPolicy): AKSI sesungguhnya adalah
        # vektor bid, bukan himpunan terpilih -- seleksi k-terendah deterministik begitu
        # bid diketahui. Rasio PPO karenanya dihitung atas bid. Deteksi lewat `bid_log_std`
        # (parameter yang hanya dimiliki kebijakan bidding), pola yg sama dgn `use_pref`.
        use_bids = hasattr(self.policy, "bid_log_std")
        if use_bids and any(t.bids is None for t in transitions):
            raise ValueError(
                "kebijakan bidding tetapi ada transisi tanpa `bids` -- cek "
                "rollout.py::get_recommendation (tr.bids = act.get('bids')).")
        bids_b = (torch.as_tensor(np.stack([t.bids for t in transitions]), dtype=torch.float32)
                  if use_bids else None)

        B = len(transitions)
        idx = np.arange(B)
        last = {}
        grad_norm = 0.0
        n_skipped = 0
        epochs_ran = 0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, B, self.minibatch):
                mb = idx[start:start + self.minibatch]
                eval_kwargs = {"critic_obs_b": critic_obs_b[mb]}
                if use_pref:
                    eval_kwargs["pref_hist_b"] = pref_hist_b[mb]
                if use_bids:
                    eval_kwargs["bids_b"] = bids_b[mb]
                logp, ent, value = self.policy.evaluate(
                    obs_b[mb], mask_b[mb], chosen_b[mb], n_rec_b[mb], hist_b[mb],
                    **eval_kwargs)
                ratio = torch.exp(logp - old_logp[mb])
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = nn.functional.mse_loss(value, ret_b[mb])
                ent_loss = -ent.mean()
                loss = pi_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss
                # Guard: lewati langkah bila loss non-finite (cegah bobot rusak senyap).
                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norm = float(gn)
                self.opt.step()
                last = {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(),
                        "entropy": ent.mean().item(), "loss": loss.item()}
            epochs_ran += 1
            # Early-stop epoch bila KL sudah melewati ambang (update terlalu jauh).
            if self.target_kl is not None:
                with torch.no_grad():
                    full_kwargs = {"critic_obs_b": critic_obs_b}
                    if use_pref:
                        full_kwargs["pref_hist_b"] = pref_hist_b
                    if use_bids:
                        full_kwargs["bids_b"] = bids_b
                    nlp, _, _ = self.policy.evaluate(obs_b, mask_b, chosen_b, n_rec_b, hist_b, **full_kwargs)
                    kl = float((old_logp - nlp).mean())
                if kl > 1.5 * self.target_kl:
                    break

        # --- Diagnostik kesehatan PPO (satu pass akhir pada seluruh batch) ---
        full_kwargs = {"critic_obs_b": critic_obs_b}
        if use_pref:
            full_kwargs["pref_hist_b"] = pref_hist_b
        if use_bids:
            full_kwargs["bids_b"] = bids_b
        with torch.no_grad():
            new_logp, new_ent, new_val = self.policy.evaluate(obs_b, mask_b, chosen_b, n_rec_b, hist_b, **full_kwargs)
            ratio = torch.exp(new_logp - old_logp)
            approx_kl = float((old_logp - new_logp).mean())      # drift policy (~0.01-0.05 sehat)
            clip_frac = float((torch.abs(ratio - 1.0) > self.clip).float().mean())
            # explained variance PER ALIRAN (1=sempurna, <=0=buruk). Tiap kritik dinilai
            # sendiri -- kritik yg satu bisa sehat sementara yg lain gagal, dan itu justru
            # informasi yang dibutuhkan utk mendiagnosis kritik ganda.
            ev_per = []
            for k in range(ret_b.shape[1]):
                var_k = float(ret_b[:, k].var())
                ev_per.append(float(1.0 - (ret_b[:, k] - new_val[:, k]).var() / var_k)
                              if var_k > 1e-8 else 0.0)
            ev = float(np.mean(ev_per))
        last.update({
            "approx_kl": approx_kl, "clip_frac": clip_frac, "explained_var": ev,
            "grad_norm": grad_norm, "entropy_final": float(new_ent.mean()),
            "adv_mean": float(adv.mean()), "adv_std": float(adv.std()),
            "n_skipped": n_skipped, "epochs_ran": epochs_ran,
            "r_bar": float(np.mean(self.r_bar)),
        })
        if self.n_critics > 1:   # diagnostik khusus kritik ganda
            last["explained_var_per_stream"] = ev_per
            last["beta"] = [float(b) for b in self._last_beta]
            last["r_bar_per_stream"] = [float(x) for x in np.atleast_1d(self.r_bar)]
        return last


# ----------------------------- PolicyTrainer (mode A/B/C) -----------------------------
class TorchPolicyTrainer:
    """Implementasi PolicyTrainer: tiap `train(forecaster, n_updates)` menjalankan
    n_updates iterasi (rollout episode -> PPO update) dgn forecaster memasok ŵ."""

    # Titik-colok kelas policy -- default HPPOPolicy (perilaku lama tak berubah).
    # Subclass cukup override atribut ini utk memakai policy lain dgn signature
    # konstruktor sama (obs_dim, critic_obs_dim, n_spklu) TANPA menduplikasi __init__ ini.
    POLICY_CLS = HPPOPolicy

    def __init__(self, dataset_path, k=3, t_max=120.0, rollout_steps=384,
                 reward_calc=None, seed=0, verbose=True, log_path=None,
                 epsilon=0.0, threshold=0.20, personalize=False, embed_dim=8, equity_calc=None,
                 pref_feature_mode=False, willingness_ratio=None,
                 willingness_radius_km=None, n_critics: int = 1, **ppo_kw):
        # personalize/embed_dim dipertahankan sbg kwarg agar caller lama tak pecah, tapi
        # TAK dipakai: policy tak lagi punya embedding per-pengguna (§Arsitektur Agen tesis
        # eksplisit menolak agent indication; diferensiasi murni dari riwayat observasi).
        self.dataset_path = dataset_path
        self.k = k; self.t_max = t_max; self.rollout_steps = rollout_steps
        # Perbaikan T1-serupa (Validasi_Generik/LAPORAN_VALIDASI.md, Addendum 2 §"Fase 4"):
        # default None menyamakan dgn jalur kalibrasi Tier 6/9 (wait_mean/tol_low/tol_high/
        # gamma di marl_spklu/env/user.py diturunkan dari willingness_ratio=None). Sebelumnya
        # _fresh_sim TAK PUNYA jalur override sama sekali -- selalu diam-diam memakai
        # default lama training.py::_fresh_sim (5.0, salah skala), tak bisa dikoreksi lewat
        # API publik kelas ini.
        self.willingness_ratio = willingness_ratio
        self.willingness_radius_km = willingness_radius_km
        self.rc = reward_calc or RewardCalculator()
        # v2: EstWait selalu jujur (tak ada lagi honest_estwait toggle) -- epsilon/threshold
        # dipakai policy.act() (top-K/threshold + eksplorasi ε-greedy, lihat policy.py).
        self.epsilon = float(epsilon)
        self.threshold = float(threshold)
        # equity_calc (opsional): EquityRewardCalculator ala PDQN diskrit, ditambahkan
        # SEBAGAI SUKU TAMBAHAN reward di RLRolloutAgent.on_decision (lihat rollout.py).
        self.equity_calc = equity_calc
        # pref_feature_mode (opsional, HANYA berlaku bila POLICY_CLS punya modul preferensi
        # PDQN spt PDQNContinuousPolicy): pasangan (a_hat,a) sbg VEKTOR FITUR stasiun,
        # bukan one-hot identitas -- lihat pdqn_policy.py::hist_feat_dim_feature.
        self.pref_feature_mode = bool(pref_feature_mode)
        self.verbose = verbose
        self.log_path = log_path            # JSONL per-iterasi (untuk plot/bandingkan run)
        self.seed = seed
        # PERBAIKAN (2026-08-16): `random.seed` SEBELUMNYA TIDAK dipanggil di sini (hanya
        # torch & np.random), padahal simulator memakai modul `random` utk menyampling
        # waktu/energi pengisian (`env/spklu.py`: random.gauss / random.betavariate).
        # Akibatnya SELURUH training jalur PPO TIDAK DETERMINISTIK -- seed sama, kode sama,
        # hasil berbeda tiap proses (terverifikasi: gini 0,65593 vs 0,65632 pada 2 proses
        # identik). `dqn_trainer.py` sudah menyemai `random` sejak awal, jadi jalur PDQN
        # deterministik -- ketimpangan reproduktibilitas antar-lengan yang tak disengaja.
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim = self._fresh_sim()
        self.N = len(sim.spklus)
        from marl_spklu.rl.policy import STATION_FEAT_DIM, CRITIC_STATION_FEAT_DIM
        self.obs_dim = 6 + STATION_FEAT_DIM * self.N   # 6 scalar (soc,x,y,sin_h,cos_h,battery) +
                                        # STATION_FEAT_DIM x N: onehot,dist,wait_hat,queue,conn,rec_activity
        self.critic_obs_dim = CRITIC_STATION_FEAT_DIM * self.N + 10   # utilisasi,antrean,rec_activity,
                                                # conn,oracle_wait (5N) + gini,trust_mean,trust_std,
                                                # trust_min,delta_gini,user_trust,user_freq_i,user_w4,
                                                # sin_h,cos_h (10)
        # pref_feature_mode hanya diteruskan bila POLICY_CLS benar2 punya param itu
        # (mis. PDQNContinuousPolicy) -- HPPOPolicy biasa tak punya modul preferensi PDQN.
        import inspect
        policy_kw = {}
        if "pref_feature_mode" in inspect.signature(self.POLICY_CLS.__init__).parameters:
            policy_kw["pref_feature_mode"] = self.pref_feature_mode
        # n_critics (kritik ganda MASTER) diteruskan ke policy; default 1 = perilaku lama.
        if "n_critics" in inspect.signature(self.POLICY_CLS.__init__).parameters:
            policy_kw["n_critics"] = int(n_critics)
        self.policy = self.POLICY_CLS(self.obs_dim, self.critic_obs_dim, self.N,
                                      **policy_kw)
        self.ppo = PPOTrainer(self.policy, **ppo_kw)
        self.history = []
        self._logger = _make_logger(verbose)
        self._it_global = 0
        # Echo konfigurasi lengkap di awal (reproducibility/identifikasi run).
        self.config = {"dataset": dataset_path, "N": self.N, "obs_dim": self.obs_dim,
                       "willingness_ratio": self.willingness_ratio,
                       "k": k, "t_max": t_max, "rollout_steps": rollout_steps, "seed": seed,
                       "alpha_wait": self.rc.alpha_wait, "beta_prox": self.rc.beta_prox,
                       "alpha_gini": self.rc.alpha_gini,
                       "alpha_flock": self.rc.alpha_flock, "lr": self.ppo.opt.param_groups[0]["lr"],
                       "gamma": self.ppo.gamma, "avg_reward": self.ppo.avg_reward, "clip": self.ppo.clip,
                       "ent_coef": self.ppo.ent_coef, "epochs": self.ppo.epochs,
                       "minibatch": self.ppo.minibatch}
        self._logger.info("CONFIG %s", self.config)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"event": "config", **self.config}) + "\n")

    def _fresh_sim(self):
        from marl_spklu.rl.training import _fresh_sim
        return _fresh_sim(self.dataset_path, willingness_ratio=self.willingness_ratio,
                          willingness_radius_km=self.willingness_radius_km)

    def _reset_rc(self):
        """Bersihkan state lintas-keputusan RewardCalculator saat simulasi diganti.
        WAJIB dipanggil berpasangan dgn setiap `_fresh_sim()` yang diikuti rollout --
        lihat `RewardCalculator.reset_episode_state` utk bug yang ditutupnya."""
        for rc in (self.rc, getattr(self, "equity_calc", None)):
            if rc is not None and hasattr(rc, "reset_episode_state"):
                rc.reset_episode_state()

    def train(self, forecaster, n_updates: int):
        for _ in range(n_updates):
            it = self._it_global
            sim = self._fresh_sim()
            self._reset_rc()          # simulasi baru -> delta-Gini harus mulai bersih
            trs, info = collect_rollout(sim, self.policy, self.rc, forecaster,
                                        k=self.k, max_steps=self.rollout_steps,
                                        epsilon=self.epsilon, threshold=self.threshold)
            stats = self.ppo.update(trs)
            rec = {"iter": it, **info, **stats}
            self.history.append(rec)
            self._it_global += 1
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"event": "iter", **rec}) + "\n")
            if self.verbose:
                self._logger.info(
                    "[upd %3d] R=%+.4f (acc=%+.4f shape=%+.4f) | accept=%.2f relieve=%.2f "
                    "estwait=%.1f | KL=%.4f clip=%.2f EV=%.2f grad=%.2f ent=%.2f | "
                    "gini=%.3f served=%d%s",
                    it, info["mean_reward"], info.get("mean_reward_accept", 0),
                    info.get("mean_reward_shape", 0), info.get("acceptance_rate", 0),
                    info.get("relieves_rate", 0), info.get("mean_disp_estwait", 0),
                    stats.get("approx_kl", 0), stats.get("clip_frac", 0),
                    stats.get("explained_var", 0), stats.get("grad_norm", 0),
                    stats.get("entropy_final", 0), info["gini_served"],
                    info.get("total_served", 0),
                    (" SKIP=%d" % stats["n_skipped"]) if stats.get("n_skipped") else "")
        return self.policy

    def make_agent(self, forecaster):
        """Factory: sim -> InferenceAgent (policy terkini) untuk pengumpulan mode C."""
        pol, k, eps, thr = self.policy, self.k, self.epsilon, self.threshold
        return lambda sim: InferenceAgent(pol, sim, forecaster, k, epsilon=eps, threshold=thr)


# ----------------------------- Continuing-task trainer -----------------------------
class TorchContinuingTrainer(TorchPolicyTrainer):
    """Melatih sebagai TUGAS CONTINUING (bukan episode reset per-update): satu simulasi
    PERSISTEN di-step per potongan (`rollout_steps`, mis. 96 = 1 hari), lalu PPO update tiap
    potongan dgn GAE bootstrapping. Trust/user state TIDAK di-reset antar potongan -> dinamika
    trust berkembang lintas potongan. Jauh lebih efisien: satu pass horizon = (horizon/chunk)
    update (mis. 180 update per satu simulasi 180-hari), vs 1 update/simulasi pada trainer
    episodik. Saat horizon habis, sim di-reset (pass baru) — batas ini ditandai done=True.

    avg_reward=True secara default (bisa dioverride via ppo_kw): continuing task tanpa
    reset memakai differential return (§IV.1.7), bukan discounted return -- sub-goal jauh
    di masa depan (mis. Gini yg membaik berhari-hari kemudian) tak boleh di-discount lebih
    dari yang dekat."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("avg_reward", True)
        super().__init__(*args, **kwargs)

    def train(self, forecaster, n_updates: int, alpha_gini_schedule=None):
        """alpha_gini_schedule: opsional, callable(frac:0..1)->float. Dipanggil TIAP
        potongan (frac = it/n_updates) untuk MENGUBAH `self.rc.alpha_gini` di tempat --
        krn `self.rc` adalah objek yang SAMA dipakai `agent` sepanjang training
        (termasuk lintas reset horizon), mutasi ini otomatis berlaku ke potongan
        berikutnya tanpa perlu reset state lain (trust dsb tetap kontinu)."""
        chunk = self.rollout_steps
        sim = self._fresh_sim()
        self._reset_rc()              # simulasi baru -> delta-Gini harus mulai bersih
        agent = RLRolloutAgent(self.policy, sim, self.rc, forecaster, self.k,
                               epsilon=self.epsilon, threshold=self.threshold)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            if alpha_gini_schedule is not None:
                self.rc.alpha_gini = float(alpha_gini_schedule(it / max(1, n_updates - 1)))
            boundary = False
            for _ in range(chunk):
                sim.step_once(step, agent=agent)
                step += 1
                if step >= sim.max_steps:               # horizon habis -> tutup pass
                    boundary = True
                    break
            # Di batas horizon, transisi yang BELUM resolved tak bisa dibawa ke pass
            # berikutnya (sim di-reset). SEBELUMNYA ia di-resolve PAKSA lalu ikut update --
            # padahal suku `wait` (delayed) tak pernah di-emit untuknya, jadi reward-nya
            # TIDAK LENGKAP. Terukur akibatnya (18 run x 300 iterasi):
            #     batch batas  : n_tr 1337 vs 440 normal (maks 3949)
            #     reward       : 0,046 vs 0,15 normal   <- gejala reward tak lengkap
            #     approx_kl    : 0,156 vs 0,03-0,05     <- 4x, jauh di atas ambang 0,045
            # Batch 3x lebih besar = 3x lebih banyak langkah minibatch per epoch, sementara
            # rem KL hanya diperiksa SETELAH epoch selesai -> kebijakan melompat jauh sekali
            # tiap 10 iterasi. 20 dari 33 onset kolaps entropi terjadi tepat setelahnya
            # (harapan acak 3,3).
            # SEKARANG: transisi tak-resolved DIBUANG di batas. Transisi yang hasilnya tak
            # pernah terwujud tidak boleh melatih kebijakan.
            # Hanya transisi RESOLVED (wait sudah di-emit) yang dipakai update; sisanya
            # (user masih di jalan/antre/charging) DITAHAN ke potongan berikutnya.
            resolved = [t for t in agent.transitions if t.resolved]
            pending = [t for t in agent.transitions if not t.resolved]
            self._it_global += 1
            if resolved:
                resolved[-1].done = boundary
                stats = self.ppo.update(resolved)
                rewards = np.array([t.reward for t in resolved])
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                info = {"mean_reward": float(rewards.mean()),
                        "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                        "mean_flock_penalty": float(np.mean([t.flock_penalty for t in resolved])),
                        "gini_served": _gini(served), "n_tr": len(resolved),
                        "n_pending": len(pending), "alpha_gini": float(self.rc.alpha_gini)}
                rec = {"iter": it, **info, **stats}
                self.history.append(rec)
                # Tulis per-iterasi ke JSONL (SEBELUMNYA TIDAK ADA di trainer continuing --
                # hanya trainer episodik yang menulis, shg run panjang tak bisa dipantau
                # progresnya dan hasilnya baru terlihat saat proses selesai).
                if self.log_path:
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps({"event": "iter", **rec}, default=float) + "\n")
                if self.verbose:
                    self._logger.info(
                        "[chunk %3d] R=%+.4f accept=%.2f flock=%.2f | KL=%.4f EV=%.2f | gini=%.3f "
                        "n=%d pend=%d%s", it, info["mean_reward"], info["acceptance_rate"],
                        info["mean_flock_penalty"],
                        stats.get("approx_kl", 0), stats.get("explained_var", 0),
                        info["gini_served"], info["n_tr"], info["n_pending"],
                        " |PASS-BARU" if boundary else "")
            if boundary:                                  # mulai pass baru (reset trust)
                sim = self._fresh_sim()
                # BUG UTAMA yang ditutup di sini: tanpa reset, keputusan pertama pass
                # BARU menghitung delta-Gini terhadap Gini pass LAMA -- sinyal palsu ~10x
                # sinyal sebenarnya, disiarkan ke semua transisi. 16 dari 23 onset kolaps
                # entropi terukur terjadi tepat pada iterasi setelah batas ini.
                self._reset_rc()
                agent = RLRolloutAgent(self.policy, sim, self.rc, forecaster, self.k,
                                       epsilon=self.epsilon, threshold=self.threshold)
                step = 0
            else:
                agent.transitions = pending               # bawa transisi belum-resolved
        return self.policy


# ----------------------------- OnlinePolicyTrainer (mode D) -----------------------------
class TorchOnlineTrainer(TorchPolicyTrainer):
    """Mode D: tiap train_step = satu rollout + PPO update, sekaligus mengembalikan
    (X, y) supervised (fitur SPKLU vs wait t+1) dari episode untuk partial_fit forecaster."""

    def train_step(self, forecaster):
        from marl_spklu.rl.forecaster import extract_features
        sim = self._fresh_sim()
        # Rollout sambil merekam pasangan (fitur_t, wait_virtual) untuk forecaster.
        X, y = [], []
        sids = list(sim.spklus.keys())
        from marl_spklu.rl.rollout import RLRolloutAgent
        agent = RLRolloutAgent(self.policy, sim, self.rc, forecaster, self.k)
        for step in range(self.rollout_steps):
            time_now = step * sim.dt_minutes
            spawns = sim.spawn_schedule.get(step, [])
            if spawns:
                for spawn_tuple in spawns:
                    # Unpack safely (in case old dataset without soc)
                    if len(spawn_tuple) == 3:
                        user, spawn_loc, soc = spawn_tuple
                    else:
                        user, spawn_loc = spawn_tuple
                        soc = 50.0
                    
                    for s in sids:
                        recent = sim.recent_recs.get(s, 0) if hasattr(sim, "recent_recs") else 0
                        dist = float(math.dist(spawn_loc, sim.spklus[s].location))
                        feat = extract_features(
                            sim.spklus[s], time_now, user=user, soc=soc,
                            recent_recs_count=recent, dist_override=dist
                        )
                        w = sim.compute_virtual_wait(user, sim.spklus[s], time_now)
                        if np.isfinite(w):
                            X.append(feat)
                            y.append(w)
            else:
                for s in sids:
                    recent = sim.recent_recs.get(s, 0) if hasattr(sim, "recent_recs") else 0
                    feat = extract_features(
                        sim.spklus[s], time_now, user=None, soc=50.0,
                        recent_recs_count=recent
                    )
                    w = sim.compute_virtual_wait(None, sim.spklus[s], time_now)
                    if np.isfinite(w):
                        X.append(feat)
                        y.append(w)
                
            sim.step_once(step, agent=agent)

        if agent.transitions:
            agent.transitions[-1].done = True
        stats = self.ppo.update(agent.transitions)
        self.history.append({"mean_reward": float(np.mean([t.reward for t in agent.transitions]))
                             if agent.transitions else 0.0, **stats})
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
