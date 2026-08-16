"""Modul bersama untuk notebook eksekusi (Eksekusi_RL/0X_*.ipynb).

Mengumpulkan fungsi yang dipakai lintas tahap agar tiap notebook singkat dan konsisten:
- konfigurasi substrat & pembekuannya (Tahap 0)
- uji alignment reward (Tahap 0)
- sensitivitas performatif & sapuan beban (Tahap 1)
- runner baseline S0-S3 (Tahap 1)
- helper carry-forward trust & filter resolved (Tahap 0/3)

Rujukan: Dokumen_Penting/Rencana_Eksekusi_Penelitian.md (tahapan, tujuan, ukuran ketercapaian),
Dokumen_Penting/Rumusan_Masalah_Teknis_RL.md (rumusan masalah),
Dokumen_Penting/Metodologi_Perbandingan_PDQN_RRM.md (aturan atribusi, tangga ablasi).
"""
import contextlib
import json
import math
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_KANONIK = os.path.join(ROOT, "scenario_dataset_klaster12.json")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Substrat (Tahap 0.5) -- SATU sumber kebenaran konfigurasi bersama.
# Setelah dibekukan (README Tahap 0), field ini TIDAK BOLEH berubah tanpa
# mengulang seluruh tahap yang sudah dijalankan.
# ---------------------------------------------------------------------------
SUBSTRAT = {
    "dataset_kanonik": "scenario_dataset_klaster12.json",
    "willingness_ratio": None,          # Perbaikan T1 -- konsisten kalibrasi Tier 6/9
    "willingness_radius_km": None,
    "horizon_days": 30,
    "dt_minutes": 15.0,
    "n_spklu": 6,
    "tau": 0.68,
    "beku_pada": None,                  # diisi tanggal pembekuan setelah Tahap 0 lulus gerbang
    # Rezim operasi (Tahap 1, 01_Tetapkan_Rezim.ipynb): 4x dibekukan sbg beban dipakai
    # SELURUH tahap sesudahnya (loop performatif MATI di 1x/dataset_kanonik, hidup
    # jelas mulai ~4x -- rentang trust 34,0x). BUG DITEMUKAN & DIPERBAIKI: sel notebook
    # Tahap 1 sebelumnya cuma menyetel `common.SUBSTRAT["rezim_operasi_load_multiplier"]`
    # DI MEMORI KERNEL notebook itu (tak pernah tertulis balik ke file ini) -- SETIAP
    # proses Python terpisah (skrip _tahapN_*.py dijalankan via `python script.py`)
    # memuat ulang SUBSTRAT dari sini & diam2 jatuh ke 1x. Akibatnya SELURUH Tahap 2 & 3
    # (termasuk temuan lock-in trust-awal) ternyata dijalankan di rezim 1x, BUKAN 4x yg
    # seharusnya dibekukan -- lihat catatan di Rencana_Eksekusi_Penelitian.md Tahap 2/3.
    "rezim_operasi_load_multiplier": 4.0,
    "rezim_referensi_load_multiplier": 1.0,
}


def gini(a) -> float:
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a)
    n = a.shape[0]
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def fresh_sim(dataset_path=None, willingness_ratio=None, willingness_radius_km=None):
    """Sim baru dari substrat kanonik (default) atau dataset lain (mis. sapuan beban)."""
    from marl_spklu.rl.training import _fresh_sim
    dataset_path = dataset_path or DATASET_KANONIK
    return _fresh_sim(dataset_path,
                      willingness_ratio=willingness_ratio if willingness_ratio is not None
                      else SUBSTRAT["willingness_ratio"],
                      willingness_radius_km=willingness_radius_km)


# ---------------------------------------------------------------------------
# Tahap 0.4 -- Uji alignment reward
# ---------------------------------------------------------------------------
def reward_alignment_test(dataset_path=None, seed=42, reward_calc=None):
    """Jalankan kebijakan aturan-tetap (rentang Gini rendah..tinggi) lewat RLRolloutAgent
    memakai RewardCalculator BIASA (bukan EquityRewardCalculator), korelasikan reward
    rata-rata per skenario vs gini_mean yang dihasilkan. Reward BAIK -> korelasi NEGATIF.

    Ini menguji `RewardCalculator` (default jalur H-PPO) -- BUKAN `EquityRewardCalculator`
    (yang sudah pernah diverifikasi selaras di archive/docs/LAPORAN_IMPLEMENTASI_PDQN.md
    §3.7, dipakai jalur PDQN diskrit). RewardCalculator biasa belum pernah diverifikasi
    ulang alignment-nya pada desain reward saat ini (lihat Rumusan_Masalah_Teknis_RL.md §4.3).
    """
    from marl_spklu.agents.greedy_agent import GreedyAgent
    from marl_spklu.agents.diversified_agent import DiversifiedAgent
    from marl_spklu.agents.wait_predictor import DeterministicWaitPredictor
    from marl_spklu.rl.rollout import RLRolloutAgent
    from marl_spklu.rl.rewards import RewardCalculator
    from marl_spklu.rl.forecaster import FormulaForecaster
    from marl_spklu.env.history_buffer import HistoryBuffer
    from marl_spklu.experiments import metrics as _metrics
    import torch

    class RandomAgent:
        def get_recommendation(self, spklus, **kw):
            return [random.choice(sorted(spklus.keys()))]

        def predict_waits(self, spklus, **kw):
            return DeterministicWaitPredictor().predict(spklus)

    class AntiGreedyAgent:
        """Kebalikan Greedy-util: rekomendasikan SPKLU PALING ramai -> Gini terburuk."""

        def _score(self, spklu):
            total_evs = sum(len(q) for q in spklu.queues.values())
            total_evs += sum(len(c) for c in spklu.charging.values())
            total_cap = sum(spklu.capacities.values())
            return total_evs / total_cap if total_cap > 0 else 0.0

        def get_recommendation(self, spklus, **kw):
            scores = {sid: self._score(s) for sid, s in spklus.items()}
            worst = max(scores.values())
            return [sorted(sid for sid, s in scores.items() if s == worst)[0]]

        def predict_waits(self, spklus, **kw):
            return DeterministicWaitPredictor().predict(spklus)

    policies = {
        "greedy_util": lambda: GreedyAgent(mode="utilization"),
        "greedy_queue": lambda: GreedyAgent(mode="queue"),
        "diversified": lambda: DiversifiedAgent(mode="d_choices", d=2),
        "random": lambda: RandomAgent(),
        "anti_greedy": lambda: AntiGreedyAgent(),
    }

    class _WaitPredAdapter:
        """Bungkus agen non-RL agar predict_waits-nya dipakai sbg forecaster RLRolloutAgent
        (RLRolloutAgent butuh forecaster.predict(spklus, time_now, user=, soc=, sim=))."""
        def __init__(self, inner):
            self.inner = inner

        def predict(self, spklus, time_now=0.0, user=None, soc=50.0, sim=None):
            return self.inner.predict_waits(spklus)

    def _station_feat(sim_, sid, wait_hat):
        feat = sim_.spklu_features.get(sid, {})
        loc = feat.get('loc', (0.0, 0.0))
        conn = feat.get('conn', 1.0)
        w = wait_hat.get(sid, 0.0)
        return np.array([loc[0], loc[1], conn, w])

    rc = reward_calc or RewardCalculator()
    rows = []
    for name, factory in policies.items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # `rc` dipakai ulang lintas kebijakan dlm loop ini -- reset state internal
        # delta-Gini (`_prev_gini`) supaya episode kebijakan berikutnya tak mewarisi
        # sisa Gini dari episode kebijakan sebelumnya (leak lintas-episode).
        if hasattr(rc, "_prev_gini"):
            rc._prev_gini = None
        sim = fresh_sim(dataset_path)
        max_steps = getattr(sim, "max_steps", max(sim.spawn_schedule) + 1)
        sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=max_steps)
        rule_agent = factory()

        # Replikasi PENUH rantai reward RewardCalculator (prox segera + global per-langkah
        # + wait delayed di on_charge_complete) TANPA bergantung pada kebijakan RL.
        # PENTING: satu ENTRI PER KEPUTUSAN (bukan per user_id) -- user yang sama bisa
        # membuat banyak keputusan sepanjang 30 hari (trip berurutan, tak pernah tumpang
        # tindih -- lihat rollout.py), jadi kunci per-user_id akan MENGGABUNGKAN banyak
        # keputusan jadi satu skalar dan merusak rata-rata per-keputusan.
        decision_rewards = []          # list of dict(local=..) diisi bertahap via indeks
        trip_open = {}                 # user_id -> indeks ke decision_rewards (trip berjalan)
        # Jendela per-langkah HANYA dipakai utk tahu indeks mana yg dibuat step ini (utk
        # broadcast gini_reward) -- BUKAN lagi utk flocking (itu kini per-keputusan via
        # sim.recent_recs, persis rollout.py::get_recommendation/on_decision).
        recs_this_step = {"step": None, "indices": []}

        class Hooked:
            def get_recommendation(self_, spklus, **kw):
                recs = rule_agent.get_recommendation(spklus)
                step = sim.current_step
                if recs_this_step["step"] != step:
                    recs_this_step["step"] = step
                    recs_this_step["indices"] = []
                # recent_rec_count DIBACA SEBELUM Simulator menambah hitungan (recs[0])
                # -- persis rollout.py::get_recommendation.
                self_._recent_rec_count = (float(sim.recent_recs.get(recs[0], 0))
                                           if recs else 0.0)
                self_._last_wait_hat = rule_agent.predict_waits(spklus)
                return recs

            def predict_waits(self_, spklus, **kw):
                return rule_agent.predict_waits(spklus)

            def on_decision(self_, user, chosen_spklu_id, recs, feasible_spklus):
                if not recs:
                    return
                primary = recs[0]
                wait_hat = self_._last_wait_hat
                feat_rec = _station_feat(sim, primary, wait_hat)
                feat_chosen = _station_feat(sim, chosen_spklu_id, wait_hat)
                prox_value = rc.prox(feat_rec, feat_chosen)
                r_local = rc.decision_reward(prox_value)
                # Anti-herding jendela bergulir (Tahap 0.1) -- persis rollout.py::on_decision.
                r_local += rc.flock_reward_rolling(self_._recent_rec_count)
                decision_rewards.append(dict(local=r_local, gini=0.0, wait=0.0, resolved=False))
                recs_this_step["indices"].append(len(decision_rewards) - 1)

                # wait_default = wait ke stasiun PREFERENSI ASLI pengguna (argmax LCMNL,
                # BUKAN rekomendasi agen) -- persis semantik rollout.py::get_recommendation
                # (default_idx dari U(s) pengguna, wait_default via compute_virtual_wait).
                time_now = sim.current_step * sim.dt_minutes
                best_sid, best_u = None, float("-inf")
                for sid, spklu in feasible_spklus.items():
                    feat = sim.spklu_features.get(sid, {})
                    dk = math.dist(user.location, feat.get('loc', (0.0, 0.0)))
                    pop = feat.get('pop', 1.0); conn_j = feat.get('conn', 1.0)
                    is_prev = 1.0 if sid == user.prev_spklu else 0.0
                    soc = getattr(user, "current_soc", getattr(user, "soc", 50.0))
                    soc_urgency = (1.0 - soc / 100.0) * dk
                    u_s = (user.w1 * dk + user.w2 * math.log(1 + pop) +
                          user.w3 * math.log(1 + conn_j) + user.w4 * is_prev +
                          user.w5 * soc_urgency)
                    if u_s > best_u:
                        best_u, best_sid = u_s, sid
                default_sid = best_sid or primary
                disp_estwait = float(wait_hat.get(primary, 0.0))
                if default_sid != primary and default_sid in feasible_spklus:
                    wait_default = float(sim.compute_virtual_wait(
                        user, feasible_spklus[default_sid], time_now))
                else:
                    wait_default = disp_estwait
                trip_open[user.user_id] = dict(
                    idx=len(decision_rewards) - 1,
                    disp_estwait=disp_estwait, wait_default=wait_default)

            def on_step_end(self_, sim_, step):
                pass  # suku gini diakumulasi via step_once_with_gini (lihat di bawah)

            def on_charge_complete(self_, user):
                trip = trip_open.pop(user.user_id, None)
                if trip is None:
                    return
                r_wait = rc.wait_reward(trip["wait_default"], user.wait_time, trip["disp_estwait"])
                decision_rewards[trip["idx"]]["wait"] = r_wait
                decision_rewards[trip["idx"]]["resolved"] = True

        agent = Hooked()
        # Suku Gini (CTDE) -- broadcast identik ke semua keputusan yg dibuat LANGKAH ini
        # (persis rollout.py::on_step_end). Disimpan TERPISAH dari `local` (kolom "gini"
        # per-keputusan) agar reward_mean_local (prox+flock, kini individual) &
        # reward_mean_global (gini, tetap broadcast) tetap bisa dilaporkan terpisah.
        orig_step_once = sim.step_once
        def step_once_with_gini(step, agent=None):
            orig_step_once(step, agent=agent)
            indices = recs_this_step["indices"] if recs_this_step["step"] == step else []
            if indices:
                utils = np.array([s.get_utilization() for s in sim.spklus.values()])
                g_term = rc.gini_reward(utils)
                for idx in indices:
                    decision_rewards[idx]["gini"] = g_term
        sim.step_once = step_once_with_gini

        sim.run(max_steps=max_steps, agent=agent)

        gs = _metrics.gini_series_from_history(sim.history)
        gini_mean = float(gs.mean()) if gs.size else 0.0
        served = np.array([s.total_served for s in sim.spklus.values()], float)

        n_dec = len(decision_rewards)
        n_resolved = sum(1 for d in decision_rewards if d["resolved"])
        local_vals = [d["local"] for d in decision_rewards]   # prox + flock (individual, Tahap 0.1)
        gini_vals = [d["gini"] for d in decision_rewards]     # broadcast per-langkah
        wait_vals = [d["wait"] for d in decision_rewards]  # 0.0 utk trip unresolved (di ujung horizon)
        mean_local = float(np.mean(local_vals)) if local_vals else 0.0
        mean_wait = float(np.mean(wait_vals)) if wait_vals else 0.0
        mean_global = float(np.mean(gini_vals)) if gini_vals else 0.0
        reward_total_mean = mean_local + mean_wait + mean_global
        rows.append(dict(policy=name, gini_mean=gini_mean, gini_served=gini(served),
                         reward_mean_local=mean_local, reward_mean_wait=mean_wait,
                         reward_mean_global=mean_global, reward_mean_total=reward_total_mean,
                         n_decisions=n_dec, n_resolved=n_resolved))
    return rows


def reward_alignment_correlation(rows, key="reward_mean_total"):
    ginis = np.array([r["gini_mean"] for r in rows])
    rewards = np.array([r[key] for r in rows])
    if np.std(rewards) < 1e-12 or np.std(ginis) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rewards, ginis)[0, 1])


# ---------------------------------------------------------------------------
# Tahap 0.1 -- Dekomposisi reward (RewardCalculator penuh, kebijakan RL untrained)
# ---------------------------------------------------------------------------
def reward_decomposition(dataset_path=None, seed=0, k=2, reward_calc=None):
    """Jalankan kebijakan HPPOPolicy BELUM TERLATIH sepanjang episode penuh, instrumentasi
    tiap suku reward terpisah. Mengukur sifat STRUKTURAL reward, bukan hasil optimasi.
    `reward_calc`: instance RewardCalculator kustom (mis. `.seimbang()`, `.individual()`,
    `.kumulatif()`) -- default None -> RewardCalculator() biasa (lama)."""
    import torch
    from marl_spklu.rl.rollout import RLRolloutAgent
    from marl_spklu.rl.policy import HPPOPolicy, STATION_FEAT_DIM, CRITIC_STATION_FEAT_DIM
    from marl_spklu.rl.rewards import RewardCalculator
    from marl_spklu.rl.forecaster import FormulaForecaster

    torch.manual_seed(seed)
    np.random.seed(seed)
    sim = fresh_sim(dataset_path)
    N = len(sim.spklus)
    rc = reward_calc if reward_calc is not None else RewardCalculator()
    pol = HPPOPolicy(6 + STATION_FEAT_DIM * N, 10 + CRITIC_STATION_FEAT_DIM * N, N)
    ag = RLRolloutAgent(pol, sim, rc, FormulaForecaster(), k=k)

    # Instrumentasi jalur PRODUKSI saat ini (Tahap 0.1): gini_reward (on_step_end) &
    # flock_reward_rolling (on_decision) TERPISAH, bukan lagi global_reward gabungan
    # (deprecated, tak dipanggil rollout.py lagi). "global" = jumlah keduanya, utk
    # kompatibilitas mundur tampilan ringkasan.
    comp = {"prox": [], "gini": [], "flock": [], "wait": []}
    o_dec, o_gini, o_flock, o_wait = (rc.decision_reward, rc.gini_reward,
                                      rc.flock_reward_rolling, rc.wait_reward)
    rc.decision_reward = lambda p: (lambda v: (comp["prox"].append(v), v)[1])(o_dec(p))
    rc.gini_reward = lambda u: (lambda v: (comp["gini"].append(v), v)[1])(o_gini(u))
    rc.flock_reward_rolling = lambda cnt, scale=10.0: (
        lambda v: (comp["flock"].append(v), v)[1])(o_flock(cnt, scale))
    rc.wait_reward = lambda a, b, c: (lambda v: (comp["wait"].append(v), v)[1])(o_wait(a, b, c))

    for s in range(sim.max_steps):
        sim.step_once(s, agent=ag)

    summary = {}
    for name, vals in comp.items():
        v = np.array(vals)
        summary[name] = dict(n=len(v), mean=float(v.mean()) if len(v) else 0.0,
                             std=float(v.std()) if len(v) else 0.0,
                             frac_nonzero=float(np.mean(v != 0)) if len(v) else 0.0)
    # "global" = kompatibilitas mundur (gini+flock). CATATAN: `gini` dipanggil sekali per
    # LANGKAH-berdecision (dibagi ke semua transisi langkah itu), `flock` sekali per
    # TRANSISI -- n keduanya BERBEDA scr desain (bukan bug), jadi "global" di sini adalah
    # JUMLAH MEAN (aproksimasi pelaporan), bukan array elemen-per-elemen yg valid utk std.
    summary["global"] = dict(mean=summary["gini"]["mean"] + summary["flock"]["mean"],
                             note="mean gini+flock (n berbeda scr desain, lihat komentar)")
    R = np.array([t.reward for t in ag.transitions])
    summary["total"] = dict(n=len(R), mean=float(R.mean()) if len(R) else 0.0,
                            std=float(R.std()) if len(R) else 0.0)

    fp = np.array([t.flock_penalty for t in ag.transitions])
    summary["flocking"] = dict(frac_active=float(np.mean(fp > 0)) if len(fp) else 0.0,
                               mean=float(fp.mean()) if len(fp) else 0.0)
    from collections import Counter
    step_counts = Counter(t.step for t in ag.transitions)
    n_conc = np.array(list(step_counts.values()))
    summary["konkurensi"] = dict(
        frac_1_keputusan=float(np.mean(n_conc == 1)) if len(n_conc) else 0.0,
        mean_per_step_aktif=float(n_conc.mean()) if len(n_conc) else 0.0)

    resolved = sum(1 for t in ag.transitions if t.resolved)
    summary["resolusi"] = dict(total=len(ag.transitions), resolved=resolved,
                               unresolved=len(ag.transitions) - resolved)
    return summary


def reward_five_way(reward_calc, dataset_path=None, seed=0, k=2):
    """Pecah reward jadi 4 suku MENTAH-nya v2 (prox, gini(-delta), flock, improvement) --
    lebih presisi dari `reward_decomposition`. Suku `honesty` DIHAPUS (Spesifikasi_Teknis_RL.md
    v2 -- EstWait selalu jujur, tak lagi atribusi ke aksi agen). Dipakai memverifikasi
    preset `RewardCalculator.seimbang/.individual/.kumulatif` (Tahap 0.1)."""
    import torch
    from marl_spklu.rl.rollout import RLRolloutAgent
    from marl_spklu.rl.policy import HPPOPolicy, STATION_FEAT_DIM, CRITIC_STATION_FEAT_DIM
    from marl_spklu.rl.rewards import _gini as _gini_calc

    rc = reward_calc
    torch.manual_seed(seed)
    np.random.seed(seed)
    sim = fresh_sim(dataset_path)
    N = len(sim.spklus)
    pol = HPPOPolicy(6 + STATION_FEAT_DIM * N, 10 + CRITIC_STATION_FEAT_DIM * N, N)
    ag = RLRolloutAgent(pol, sim, rc, __import__(
        "marl_spklu.rl.forecaster", fromlist=["FormulaForecaster"]).FormulaForecaster(), k=k)

    gini_terms, flock_terms, prox_terms, imp_terms = [], [], [], []

    # Instrumentasi jalur PRODUKSI (Tahap 0.1): gini_reward (on_step_end, sekali per
    # LANGKAH-berdecision) & flock_reward_rolling (on_decision, sekali per TRANSISI) --
    # bukan lagi global_reward gabungan (deprecated, tak dipanggil rollout.py lagi).
    orig_gini = rc.gini_reward
    def patched_gini(utilizations):
        v = orig_gini(utilizations)
        gini_terms.append(v)
        return v
    rc.gini_reward = patched_gini

    orig_flock = rc.flock_reward_rolling
    def patched_flock(recent_rec_count, scale=10.0):
        v = orig_flock(recent_rec_count, scale)
        flock_terms.append(v)
        return v
    rc.flock_reward_rolling = patched_flock

    orig_dec = rc.decision_reward
    def patched_dec(prox_value):
        prox_terms.append(rc.beta_prox * prox_value)
        return orig_dec(prox_value)
    rc.decision_reward = patched_dec

    orig_wait = rc.wait_reward
    def patched_wait(wd, wa, de=0.0):
        imp = max(0.0, wd - wa) / rc.wait_scale
        imp_terms.append(rc.alpha_wait * imp)
        return orig_wait(wd, wa, de)
    rc.wait_reward = patched_wait

    for s in range(sim.max_steps):
        sim.step_once(s, agent=ag)

    out = {}
    for name, arr in [("prox", prox_terms), ("gini", gini_terms), ("flock", flock_terms),
                      ("improvement", imp_terms)]:
        a = np.array(arr)
        out[name] = dict(mean=float(a.mean()) if len(a) else 0.0,
                         std=float(a.std()) if len(a) else 0.0)

    ind_var = out["prox"]["std"]**2 + out["improvement"]["std"]**2
    glob_var = out["gini"]["std"]**2 + out["flock"]["std"]**2
    out["_frac_varians_individual"] = ind_var / (ind_var + glob_var) if (ind_var + glob_var) else 0.0
    return out


# ---------------------------------------------------------------------------
# Tahap 0.2 -- Horizon pelatihan (DIREVISI: bukan lagi carry-forward, lihat
# Rumusan_Masalah_Teknis_RL.md §4.1). Trainer dasar TIDAK carry-forward trust lintas
# horizon -- ini SUDAH DIKONFIRMASI TIDAK MASALAH: kurva performativitas terbukti muncul
# dalam satu pass tunggal, dan reset per-horizon konsisten dgn protokol evaluasi S0-S3.
# Yang tersisa hanya soal VOLUME DATA (kunjungan/pengguna) -- diukur di sini, solusinya
# perpanjang horizon dataset (60/90 hari), bukan carry-forward.
# ---------------------------------------------------------------------------
def check_carry_forward_available():
    """Audit statis (dipertahankan sbg CATATAN, bukan lagi kriteria gerbang): konfirmasi
    trainer dasar (H-PPO) memang tidak carry-forward trust lintas horizon, hanya
    PDQNContinuousTrainer yang punya. Ini BUKAN cacat -- lihat header modul ini."""
    import inspect
    from marl_spklu.rl.ppo import TorchContinuingTrainer
    from marl_spklu.rl.pdqn_continuous_trainer import PDQNContinuousTrainer
    has_base = hasattr(TorchContinuingTrainer, "_carry_forward")
    has_pdqn = hasattr(PDQNContinuousTrainer, "_carry_forward")
    src = inspect.getsource(TorchContinuingTrainer.train)
    resets_fresh_sim = "self._fresh_sim()" in src and "_carry_forward" not in src
    return dict(base_trainer_has_carry_forward=has_base,
               pdqn_trainer_has_carry_forward=has_pdqn,
               base_trainer_resets_without_carry_forward=resets_fresh_sim)


def generate_horizon_datasets(horizon_days_list=(60, 90), seed=42, n_users=2636,
                              load_multiplier=1.0):
    """Generate dataset horizon-diperpanjang sbg ARTEFAK PERMANEN (bukan scratch) di root
    repo, mengikuti konvensi penamaan yang sudah ada (`scenario_dataset_klaster12.json` =
    30-hari kanonik; preseden `scenario_dataset_klaster31_*_90d.json` utk horizon panjang).

    Mengembalikan (config_input, output_summary) TERPISAH secara eksplisit:
    - config_input: parameter yang MENENTUKAN isi dataset (harus sama utk semua horizon
      kecuali horizon_days itu sendiri, agar perbandingan antar-horizon adil)
    - output_summary: apa yang DIHASILKAN generator (path, ukuran berkas, jumlah event,
      densitas kunjungan) -- diukur, bukan ditentukan.

    30-hari TIDAK digenerate ulang di sini -- `scenario_dataset_klaster12.json` yang sudah
    ada & tervalidasi tetap dipakai sbg referensi (lihat Rumusan_Masalah_Teknis_RL.md §4.1)."""
    import sys
    from collections import Counter
    sys.path.insert(0, os.path.join(ROOT, "Synthetic Model"))
    from generate_klaster12 import generate

    cfg_path = os.path.join(ROOT, "Synthetic Model", "model_config.json")
    config_input = dict(config_path=os.path.relpath(cfg_path, ROOT), n_users=n_users,
                        load_multiplier=load_multiplier, seed=seed,
                        horizon_days_list=list(horizon_days_list))

    output_rows = []
    for h in horizon_days_list:
        out_path = os.path.join(ROOT, f"scenario_dataset_klaster12_{h}d.json")
        if not os.path.exists(out_path):
            generate(cfg_path, out_path, n_users=n_users, horizon_days=h,
                     load_multiplier=load_multiplier, seed=seed)
        ds = json.load(open(out_path, encoding="utf-8"))
        sched = ds["schedule"]
        counts = Counter(e["user_id"] for e in sched)
        uids = [u["user_id"] for u in ds["users"]]
        visits = np.array([counts.get(uid, 0) for uid in uids])
        output_rows.append(dict(
            horizon_days=h, out_path=os.path.relpath(out_path, ROOT),
            file_size_kb=round(os.path.getsize(out_path) / 1024, 1),
            n_events=len(sched), n_users=len(uids),
            median_kunjungan=float(np.median(visits)), mean_kunjungan=float(visits.mean()),
            frac_0_kunjungan=float(np.mean(visits == 0)), frac_ge5=float(np.mean(visits >= 5)),
            max_kunjungan=int(visits.max())))

    # Baris referensi 30-hari (dataset kanonik yg SUDAH ADA, tervalidasi -- tidak digenerate
    # ulang, hanya diringkas dgn metodologi PENGUKURAN yang sama utk perbandingan apel-apel).
    ds30 = json.load(open(DATASET_KANONIK, encoding="utf-8"))
    sched30 = ds30["schedule"]
    counts30 = Counter(e["user_id"] for e in sched30)
    uids30 = [u["user_id"] for u in ds30["users"]]
    visits30 = np.array([counts30.get(uid, 0) for uid in uids30])
    output_rows.insert(0, dict(
        horizon_days=30, out_path=os.path.relpath(DATASET_KANONIK, ROOT) + "  (REFERENSI, sudah ada)",
        file_size_kb=round(os.path.getsize(DATASET_KANONIK) / 1024, 1),
        n_events=len(sched30), n_users=len(uids30),
        median_kunjungan=float(np.median(visits30)), mean_kunjungan=float(visits30.mean()),
        frac_0_kunjungan=float(np.mean(visits30 == 0)), frac_ge5=float(np.mean(visits30 >= 5)),
        max_kunjungan=int(visits30.max())))

    return config_input, output_rows


def measure_visit_density(horizon_days_list=(30, 60, 90), seed=42, n_users=2636):
    """Ukur kunjungan/pengguna langsung dari jadwal dataset (independen dari agen/hook RL)
    pada beberapa panjang horizon, load kanonik (1x). Dipakai memutuskan apakah horizon
    30-hari cukup, atau perlu diperpanjang ke 60/90-hari untuk volume data pelatihan."""
    from collections import Counter
    rows = []
    for h in horizon_days_list:
        path = generate_load_dataset(1.0, seed=seed, n_users=n_users, horizon_days=h,
                                     out_path=os.path.join(OUTDIR, f"_ds_h{h}.json"))
        ds = json.load(open(path, encoding="utf-8"))
        sched = ds["schedule"]
        counts = Counter(e["user_id"] for e in sched)
        uids = [u["user_id"] for u in ds["users"]] if "users" in ds else list(range(n_users))
        visits = np.array([counts.get(uid, 0) for uid in uids])
        rows.append(dict(horizon_days=h, n_events=len(sched),
                         median_kunjungan=float(np.median(visits)),
                         mean_kunjungan=float(visits.mean()),
                         frac_0_kunjungan=float(np.mean(visits == 0)),
                         frac_ge2=float(np.mean(visits >= 2)),
                         frac_ge5=float(np.mean(visits >= 5)),
                         max_kunjungan=int(visits.max())))
    return rows


# ---------------------------------------------------------------------------
# Tahap 1 -- Sensitivitas performatif & sapuan beban
# ---------------------------------------------------------------------------
def generate_load_dataset(load_multiplier, seed=42, n_users=2636, horizon_days=30,
                          out_path=None):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "Synthetic Model"))
    from generate_klaster12 import generate
    cfg = os.path.join(ROOT, "Synthetic Model", "model_config.json")
    out_path = out_path or os.path.join(OUTDIR, f"_ds_lm{load_multiplier}.json")
    if not os.path.exists(out_path):
        generate(cfg, out_path, n_users=n_users, horizon_days=horizon_days,
                 load_multiplier=load_multiplier, seed=seed)
    return out_path


class BiasedEstWaitAgent:
    """Bungkus GreedyAgent: EstWait yang DITAMPILKAN dikalikan bias `b`, rekomendasi
    (pilihan SPKLU) tetap identik -- mengisolasi jalur kualitas-janji->trust dari jalur
    alokasi. b=1 jujur; b<1 under-promise; b>1 over-promise."""

    def __init__(self, bias=1.0):
        from marl_spklu.agents.greedy_agent import GreedyAgent
        self.g = GreedyAgent()
        self.bias = bias

    def get_recommendation(self, spklus, **kw):
        return self.g.get_recommendation(spklus, **kw)

    def predict_waits(self, spklus, **kw):
        return {k: v * self.bias for k, v in self.g.predict_waits(spklus, **kw).items()}


def performativity_sensitivity(dataset_path, bias, seed=42):
    """Satu run: kembalikan trust_mean, fraksi trust bergerak, gini, wait mean."""
    sim = fresh_sim(dataset_path)
    random.seed(seed)
    np.random.seed(seed)
    sim.run(max_steps=sim.max_steps, agent=BiasedEstWaitAgent(bias))
    tr = np.array([u.trust for u in sim.users])
    moved = np.abs(tr - 0.5) > 1e-9
    served = np.array([s.total_served for s in sim.spklus.values()], float)
    waits = [L["wait_time"] for L in sim.logs]
    return dict(bias=bias, trust_mean=float(tr.mean()),
               trust_moved_frac=float(moved.mean()),
               trust_mean_moved=float(tr[moved].mean()) if moved.any() else 0.5,
               gini=gini(served), wait_mean=float(np.mean(waits)) if waits else 0.0,
               n_sesi=len(sim.logs))


def performativity_sweep(load_multipliers, biases=(0.25, 1.0, 2.0), seeds=(42,)):
    """Sapuan lengkap: untuk tiap beban x bias x seed, ukur sensitivitas. Kembalikan
    DataFrame-like list of dict, siap dipakai plotting/agregasi CI."""
    rows = []
    for lm in load_multipliers:
        ds = DATASET_KANONIK if lm == 1.0 else generate_load_dataset(lm)
        for seed in seeds:
            for b in biases:
                r = performativity_sensitivity(ds, b, seed=seed)
                r["load_multiplier"] = lm
                r["seed"] = seed
                rows.append(r)
    return rows


def performativity_trajectory(dataset_path, seed=42, snapshot_every_days=5):
    """Trajektori temporal (drift) trust/acceptance sepanjang horizon -- GreedyAgent jujur."""
    from marl_spklu.agents.greedy_agent import GreedyAgent
    sim = fresh_sim(dataset_path)
    random.seed(seed)
    np.random.seed(seed)
    ag = GreedyAgent()
    day_steps = int(24 * 60 / sim.dt_minutes)
    snaps = []
    for s in range(sim.max_steps):
        sim.step_once(s, agent=ag)
        if (s + 1) % (day_steps * snapshot_every_days) == 0:
            tr = np.array([u.trust for u in sim.users])
            moved = np.abs(tr - 0.5) > 1e-9
            comp = [c for u in sim.users for c in u.compliance_history]
            snaps.append(dict(hari=(s + 1) // day_steps, trust_mean=float(tr.mean()),
                              trust_mean_aktif=float(tr[moved].mean()) if moved.any() else 0.5,
                              frac_aktif=float(moved.mean()),
                              acceptance_kumulatif=float(np.mean(comp)) if comp else 0.0))
    return snaps


# ---------------------------------------------------------------------------
# Tahap 1.3 -- Baseline S0-S3 via harness
# ---------------------------------------------------------------------------
def run_all_baselines(dataset_path=None, seeds=range(10), scenario_names=None):
    from marl_spklu.experiments import harness
    dataset_path = dataset_path or DATASET_KANONIK
    return harness.compare_scenarios(dataset_path, scenario_names, seeds=seeds,
                                     willingness_ratio=SUBSTRAT["willingness_ratio"])


# ---------------------------------------------------------------------------
# Tahap 3 -- Ablasi trust (constant_trust / initial_trust)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def frozen_trust(value=0.5):
    """Wrapper tipis di atas marl_spklu.experiments.ablations.constant_trust."""
    from marl_spklu.experiments.ablations import constant_trust
    with constant_trust(value=value):
        yield


def save_json(obj, name):
    path = os.path.join(OUTDIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path
