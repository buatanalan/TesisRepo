"""Master-Hybrid PPO trainer (2026-08-29) -- rollout+update KATEGORIKAL (softmax,
`MasterHybridPPOActor`), BEDA mekanisme dari `MasterPurePPOTrainer` (bid kontinu
`Normal`+`bid_log_std`) shg TAK bisa cukup diganti lewat `actor_cls` spt varian DDPG
(`MasterPureTrainer` sudah mendukung itu langsung, TANPA kelas baru -- lih.
`_run_master_pure_hybrid_pipeline.py`). Reuse `compute_gae`/`_SlotRawLog`/
`snapshot_slots_raw` via impor (BUKAN duplikasi), hanya interaksi aktor+rollout+PPO
loss yg ditulis ulang.

PENYEDERHANAAN yg WAJIB dilaporkan: seleksi top-K di sini HANYA menyampel 1 stasiun
utama secara stokastik (Categorical atas SEMUA stasiun feasible, log-prob dipakai
rasio PPO), sisa (k-1) slot diisi DETERMINISTIK by urutan logit tertinggi berikutnya
-- BUKAN sequential-Categorical Plackett-Luce penuh spt `HPPOPolicy.act()`
(`policy.py`). Trade-off disengaja demi anggaran waktu implementasi; log-prob/rasio
PPO tetap valid krn HANYA dihitung atas keputusan utama yg sungguh stokastik."""
import random

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.rollout import (RLRolloutAgent, RewardCalculatorStub, STREAM_INDIVIDUAL,
                                   STREAM_GLOBAL, N_REWARD_STREAMS,
                                   N_REWARD_STREAMS_PURE, _gini)
from marl_spklu.rl.master_paper_obs import (build_joint_obs_master, build_joint_obs_master_ev,
                                            STATION_FEAT_DIM_MASTER, STATION_FEAT_DIM_MASTER_EV)
from marl_spklu.rl.master_pure_hybrid_policy import MasterHybridPPOActor, MasterHybridPPOCritic
from marl_spklu.rl.master_pure_ppo_policy import MasterPurePPOCritic
from marl_spklu.rl.master_pure_trainer import snapshot_slots_raw, _SlotRawLog
from marl_spklu.rl.ppo import compute_gae
from marl_spklu.rl.rewards import RewardCalculator


class MasterHybridPPOTransition:
    def __init__(self, obs, mask, primary_idx, logp, value, step, pref_hist=None,
                 n_streams: int = N_REWARD_STREAMS):
        self.obs = obs; self.mask = mask
        # `pref_hist` WAJIB disimpan (2026-08-29): langkah update PPO menghitung ULANG
        # logit, dan bila di sana `pref_hist=None` maka SELURUH parameter preferensi
        # (pref_lstm/pref_attn/pref_gate) TAK PERNAH masuk graf gradien -> `pref_gate`
        # tetap persis 0.0 -> `attended_pref = 0 * (...)` -> forward tak bergantung isi
        # riwayat sama sekali. Itulah sebabnya lengan +P menghasilkan perilaku IDENTIK
        # dgn lengan tanpa P (metrik bit-identik lintas 3 seed).
        self.pref_hist = pref_hist
        self.logp = float(logp); self.value = value
        self.step = step
        self.chosen_indices = [int(primary_idx)]
        # `n_streams`=3 pd mode aliran-murni (wait/gini/acceptance, satu suku per aliran)
        self.reward_streams = np.zeros(int(n_streams), dtype=np.float64)
        self.done = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.resolved = False; self.pushed = False; self.flock_penalty = 0.0
        self.I_raw = None
        self.stream_select = None

    @property
    def reward(self) -> float:
        return float(self.reward_streams.sum())

    def reward_vec(self, n_critics: int) -> np.ndarray:
        if self.stream_select is not None:
            return np.array([self.reward_streams[self.stream_select]], dtype=np.float64)
        if n_critics == 1:
            return np.array([self.reward_streams.sum()], dtype=np.float64)
        if n_critics != len(self.reward_streams):
            raise ValueError(f"n_critics={n_critics} != jumlah aliran={len(self.reward_streams)}")
        return self.reward_streams

    def add_reward(self, value: float, stream: int = STREAM_INDIVIDUAL) -> None:
        self.reward_streams[stream] += value


class MasterHybridPPORolloutAgent(RLRolloutAgent):
    """Observasi §3.1 MURNI. Keputusan UTAMA disampel `Categorical(logits)` atas
    stasiun feasible (log-prob dipakai rasio PPO) -- sisa (k-1) slot rekomendasi
    diisi deterministik by logit tertinggi berikutnya (lih. catatan penyederhanaan
    docstring modul)."""

    def __init__(self, actor, critic, sim, reward_calc, forecaster=None, k: int = 3,
                equity_calc=None, stream_select=None, pref_feature_mode: bool = False,
                pref_pair_outcome: bool = False, deterministic: bool = False,
                pref_hist_k: int = None, accept_stream: int = STREAM_INDIVIDUAL,
                pure_streams: bool = False):
        # `pref_pad_right=True` WAJIB & tak bersyarat di sini: `_PrefStationBackbone.
        # _encode_pref` memakai `pack_padded_sequence` yg mensyaratkan padding di BELAKANG,
        # sedangkan basis `RLRolloutAgent` mem-padding di DEPAN. Ketidakcocokan inilah bug
        # yg ditemukan 2026-08-29 -- `pref_lstm` seluruh lengan Hybrid selama ini hanya
        # membaca baris PADDING NOL, tak pernah riwayat sungguhan.
        super().__init__(actor, sim, reward_calc, forecaster, k=k, equity_calc=equity_calc,
                         pref_feature_mode=pref_feature_mode,
                         pref_pair_outcome=pref_pair_outcome, pref_pad_right=True,
                         pref_hist_k=pref_hist_k, accept_stream=accept_stream,
                         pure_streams=pure_streams)
        self.actor = actor
        self.critic = critic
        # DUA VERSI OBSERVASI -- lih. catatan identik di MasterPureRolloutAgent.
        self._ev_obs = (getattr(actor, "station_feat_dim", STATION_FEAT_DIM_MASTER)
                        == STATION_FEAT_DIM_MASTER_EV)
        self.stream_select = stream_select
        # `deterministic=True` -> argmax logit, TANPA sampling (evaluasi bersih). Dipakai
        # `MasterHybridPPOInferenceAgent` yg kini MENDELEGASIKAN ke kelas ini supaya
        # `pref_hist` sungguhan ikut terbangun saat uji -- lihat catatan di kelas itu.
        self.deterministic = bool(deterministic)
        # Hanya dipakai sbg bentuk placeholder `value` saat critic=None (jalur inferensi).
        self.n_critics_hint = int(getattr(critic, "n_critics", N_REWARD_STREAMS))

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        _rich_obs_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)
        joint_obs = (build_joint_obs_master_ev(self.sim, self.sids, time_now, user, soc)
                     if self._ev_obs else
                     build_joint_obs_master(self.sim, self.sids, time_now))
        mask = self._feasible_mask(feasible_ids)

        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            pref_hist = self._build_pref_hist(user) if self._use_pref else None
            pref_hist_t = (torch.as_tensor(pref_hist, dtype=torch.float32).unsqueeze(0)
                          if pref_hist is not None else None)
            logits = self.actor(obs_t, mask_t, pref_hist_t)          # (1,N), -inf di tak-feasible
            dist = torch.distributions.Categorical(logits=logits)
            primary_t = logits.argmax(dim=-1) if self.deterministic else dist.sample()
            logp = float(dist.log_prob(primary_t).item())
            # `critic=None` pd jalur INFERENSI (`MasterHybridPPOInferenceAgent`) -- V(s)
            # tak dipakai sama sekali di sana krn transisi dibuang tiap langkah. Dilewati
            # supaya kelas ini bisa dipakai ulang utk evaluasi tanpa memuat kritik.
            if self.critic is not None:
                zero_I = torch.zeros_like(mask_t, dtype=torch.float32)
                if hasattr(self.critic, "pref_lstm"):
                    value, _ = self.critic(obs_t, mask_t, zero_I, pref_hist_t)
                else:
                    value, _ = self.critic(obs_t, mask_t, zero_I)
            else:
                value = torch.zeros(1, self.n_critics_hint)
        logits_np = logits.squeeze(0).numpy()
        primary_idx = int(primary_t.item())

        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            chosen_order = []
        else:
            k_eff = max(1, min(self.k, int(feasible_idx.size)))
            rest = [i for i in feasible_idx if i != primary_idx]
            rest_sorted = sorted(rest, key=lambda i: -logits_np[i])
            chosen_order = [primary_idx] + rest_sorted[:k_eff - 1]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_order}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf")
                           for sid in feasible_ids}
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)
        recs = [self.sids[i] for i in chosen_order]

        tr = MasterHybridPPOTransition(joint_obs, mask, primary_idx, logp,
                                       value.squeeze(0).numpy().astype(np.float64),
                                       self.sim.current_step, pref_hist=pref_hist,
                                       n_streams=self.n_streams)
        tr.stream_select = self.stream_select
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterHybridPPOInferenceAgent:
    """Evaluasi bersih: argmax logit (deterministik, TANPA sampling).

    DIUBAH 2026-08-29 -- sebelumnya kelas ini BERDIRI SENDIRI dan memanggil
    `self.actor(obs, mask, None)`, yakni `pref_hist=None`. Akibatnya modul P **netral
    (nol) sepanjang evaluasi**, sehingga seluruh metrik pembanding tak pernah mengukur
    kontribusi P sama sekali. Komentar lama membenarkannya dgn menyebut
    `MasterEVPPOInferenceAgent` sbg preseden -- KLAIM ITU KELIRU: kelas tsb justru
    MENDELEGASIKAN ke `MasterEVPPORolloutAgent` penuh, yang membangun `pref_hist`
    sungguhan DAN meneruskan `on_decision`/`on_charge_complete` supaya riwayat
    preferensi terakumulasi selama evaluasi.

    Kini kelas ini memakai pola delegasi yang sama: rollout agent dipakai apa adanya
    (dgn `deterministic=True`, `RewardCalculatorStub`, transisi dibuang tiap langkah),
    sehingga jalur `pref_hist` saat UJI identik dgn saat LATIH -- kelas bug 'latih dan
    uji beda mode' yang sudah berulang di repo ini."""

    def __init__(self, actor, forecaster=None, k: int = 3):
        self.actor = actor
        self.actor.eval()
        from marl_spklu.rl.forecaster import FormulaForecaster
        self.forecaster = forecaster or FormulaForecaster()
        self.k = int(k)
        self._roll = None

    def bind_to_sim(self, sim):
        _bb = getattr(self.actor, "backbone", None)
        self._roll = MasterHybridPPORolloutAgent(
            self.actor, None, sim, RewardCalculatorStub(), self.forecaster, k=self.k,
            pref_feature_mode=getattr(_bb, "pref_feature_mode", False),
            pref_pair_outcome=getattr(_bb, "pref_pair_outcome", False),
            pref_hist_k=getattr(_bb, "pref_hist_k", None),
            deterministic=True)
        self.sim = sim

    def get_recommendation(self, feasible_spklus: dict):
        assert self._roll is not None, "panggil bind_to_sim(sim) sebelum sim.run(agent=...)"
        recs = self._roll.get_recommendation(feasible_spklus)
        self._roll.transitions.clear()
        return recs

    def predict_waits(self, feasible_spklus: dict):
        return self._roll.predict_waits(feasible_spklus)

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        # WAJIB diteruskan -- di sinilah `_record_pref` menambah pasangan (a_hat,a) ke
        # riwayat. Tanpa ini `pref_hist` selalu kosong walau sudah dibangun.
        self._roll.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        self._roll.transitions.clear()

    def on_charge_complete(self, user):
        # WAJIB diteruskan -- di sinilah blok HASIL (`realized_gap_norm`) di-backfill.
        self._roll.on_charge_complete(user)
        self._roll.transitions.clear()


class MasterHybridPPOTrainer:
    """Sama struktur `MasterPurePPOTrainer` (3 tahap, R* tetap dari spesialis
    pra-latih) -- HANYA aktor+rollout diganti varian Hybrid (P+attention+vektor)."""

    def __init__(self, dataset_path, mode: str = "dgr", stream_select: int = None,
                specialist_r_star: list = None, rollout_steps: int = 96, gamma: float = 0.99,
                lam: float = 0.95, lr: float = 5e-4, clip: float = 0.2, epochs: int = 10,
                minibatch: int = 32, ent_coef: float = 0.01, vf_coef: float = 0.5,
                max_grad_norm: float = 0.5, target_kl: float = 0.03,
                delay_minutes: float = 30.0, hidden: int = 64,
                actor_kwargs: dict = None,
                beta_mode: str = "gap_ratio", beta_sigma: float = 0.2,
                reward_calc=None, seed: int = 0, verbose: bool = True,
                equity_calc=None, max_step_gap: int = 4, critic_pref: bool = False,
                critic_pref_gate_init: float = 0.1,
                accept_stream: int = STREAM_GLOBAL, pure_streams: bool = False):
        # `pure_streams` (2026-08-30): SATU suku per aliran (wait / gini / acceptance),
        # `prox` & `flock` dibuang -- menghapus kebutuhan kalibrasi `alpha` sepenuhnya
        # krn penskalaan seragam satu-suku lenyap baik di normalisasi advantage maupun
        # di gap-ratio. Lihat catatan lengkap STREAM_PURE_* di rollout.py.
        self.pure_streams = bool(pure_streams)
        # `accept_stream` BAKU STREAM_GLOBAL di TRAINER (beda dari baku
        # STREAM_INDIVIDUAL di `RLRolloutAgent`, yg sengaja mempertahankan perilaku
        # lengan lama). Alasan: lih. catatan `accept_stream` di rollout.py -- acceptance
        # bermagnitudo +-1 SEGERA, menumpuknya dgn wait (kecil & tertunda) di aliran
        # INDIVIDUAL adalah konfigurasi yg SUDAH terdiagnosis bermasalah pd Kandidat A.
        # Suku ini BARU dipakai di keluarga Hybrid (belum pernah aktif), jadi tak ada
        # perilaku lama yg dilanggar dgn memilih baku yg lebih aman sejak awal.
        self.accept_stream = int(accept_stream)
        # `critic_pref` (2026-08-30): pakai `MasterHybridPPOCritic` (BER-P, param
        # TERPISAH dari aktor) menggantikan `MasterPurePPOCritic` (SELALU buta P) --
        # uji hipotesis kritik jadi sumber variansi advantage tambahan khusus utk
        # keputusan berbasis P (lih. docstring `MasterHybridPPOCritic`). Baku False ->
        # perilaku lama TAK berubah.
        self.critic_pref = bool(critic_pref)
        # `critic_pref_gate_init` SENGAJA TERPISAH dari `pref_gate_init` milik aktor
        # (bukan ikut `actor_kwargs`) -- supaya varian "Attn-saja + kritik-ber-P" bisa
        # menguji kritik dgn P AKTIF (gerbang kritik >0) SEMENTARA gerbang P AKTOR
        # tetap 0 (P aktor sengaja inert, sesuai definisi "Attn-saja"). Kalau gerbang
        # kritik ikut default 0 aktor, kritik akan terjebak deadlock gradien SAMA PERSIS
        # spt bug lama di aktor (gate=0 -> grad pref_lstm=0) -- baku 0.1 di sini utk
        # menghindarinya scr otomatis, bukan menunggu user mengingat set manual.
        self.critic_pref_gate_init = float(critic_pref_gate_init)
        assert mode in ("pretrain_specialist", "dgr"), f"mode={mode!r} tak dikenal"
        self.mode = mode
        self.dataset_path = dataset_path
        self.equity_calc = equity_calc
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma); self.lam = float(lam); self.clip = float(clip)
        self.epochs = int(epochs); self.minibatch = int(minibatch)
        self.ent_coef = float(ent_coef); self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm); self.target_kl = target_kl
        self.max_step_gap = max_step_gap

        # Jumlah aliran: 3 pd mode murni (wait/gini/acceptance, satu suku per aliran),
        # 2 pd mode lama (wait+prox / gini+flock). Lihat catatan STREAM_PURE_* di rollout.py.
        self.n_streams = N_REWARD_STREAMS_PURE if self.pure_streams else N_REWARD_STREAMS
        _nama_aliran = (["wait", "gini", "acceptance"] if self.pure_streams
                        else ["wait(+prox)", "gini(+flock)"])
        if mode == "pretrain_specialist":
            assert stream_select in range(self.n_streams), (
                f"mode='pretrain_specialist' WAJIB --stream-select 0..{self.n_streams - 1} "
                f"({', '.join(f'{i}={n}' for i, n in enumerate(_nama_aliran))})")
            self.stream_select = int(stream_select)
            self.n_critics = 1
            self._ret_best = np.full(1, -np.inf, dtype=np.float64)
            self._fixed_ret_best = False
        else:
            assert (specialist_r_star is not None
                    and len(specialist_r_star) == self.n_streams), (
                f"mode='dgr' WAJIB {self.n_streams} r_star: [{', '.join(_nama_aliran)}]")
            self.stream_select = None
            self.n_critics = self.n_streams
            self._ret_best = np.array(specialist_r_star, dtype=np.float64)
            self._fixed_ret_best = True

        self.beta_mode = str(beta_mode); self.beta_sigma = float(beta_sigma)
        self._last_beta = np.full(self.n_critics, 1.0 / self.n_critics, dtype=np.float64)
        self._last_ret_mean = np.zeros(self.n_critics, dtype=np.float64)
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim0 = self._fresh_sim()
        self.dt_minutes = sim0.dt_minutes
        self.delay_steps = max(1, round(float(delay_minutes) / self.dt_minutes))
        self._slot_log = _SlotRawLog(maxlen=self.rollout_steps + self.delay_steps + 4)
        self.N = len(sim0.spklus)

        self.actor = MasterHybridPPOActor(self.N, **(actor_kwargs or {}))
        _sfd = getattr(self.actor, "station_feat_dim", STATION_FEAT_DIM_MASTER)
        _bb = getattr(self.actor, "backbone", None)
        if self.critic_pref:
            # Kritik BER-P dgn param TERPISAH dari aktor (bobot sendiri), tapi mode
            # fitur/outcome/gerbang-init HARUS sama dgn aktor supaya dimensi cocok &
            # eksperimennya adil (P yg SAMA dilihat kedua sisi, bukan mode berbeda).
            self.critic = MasterHybridPPOCritic(
                _sfd, hidden=hidden, n_critics=self.n_critics, n_spklu=self.N,
                pref_feature_mode=getattr(_bb, "pref_feature_mode", False),
                pref_pair_outcome=getattr(_bb, "pref_pair_outcome", False),
                pref_gate_init=self.critic_pref_gate_init,
                pref_hist_k=getattr(_bb, "pref_hist_k", None))
        else:
            self.critic = MasterPurePPOCritic(_sfd, hidden=hidden,
                                              n_critics=self.n_critics)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.history = []
        self._it_global = 0

    def _fresh_sim(self):
        from marl_spklu.rl.training import _fresh_sim
        return _fresh_sim(self.dataset_path)

    def _carry_forward(self, sim, agent):
        old_users_by_id = {u.user_id: u for u in sim.users}
        new_sim = self._fresh_sim()
        for u in new_sim.users:
            old = old_users_by_id.get(u.user_id)
            if old is not None:
                u.trust_alpha = old.trust_alpha
                u.trust_beta = old.trust_beta
                u.compliance_history = list(old.compliance_history)
                u.interaction_history = list(old.interaction_history)
        agent.sim = new_sim
        agent.sids = list(new_sim.spklus.keys())
        agent.sid_to_idx = {s: i for i, s in enumerate(agent.sids)}
        agent.transitions = []
        agent._pending = None
        agent._user_trip_tr = {}
        agent._prev_gini = None
        self._slot_log.reset()
        return new_sim

    def _push_ready_pairs(self, agent, current_step: int, boundary: bool = False):
        trs = agent.transitions
        ready = []
        for i in range(len(trs) - 1):
            t = trs[i]
            if t.pushed or not t.resolved:
                continue
            target_step = t.step + self.delay_steps
            I_snap = self._slot_log.get(target_step)
            if I_snap is None and not boundary:
                continue
            if I_snap is None:
                I_snap = self._slot_log.get(current_step) or {}
            I_snap = dict(I_snap)
            if t.complied and t.chosen_indices:
                winner_sid = agent.sids[t.chosen_indices[0]]
                if winner_sid in I_snap:
                    I_snap[winner_sid] = I_snap[winner_sid] + 1.0
            t.I_raw = np.array([I_snap.get(sid, 0.0) for sid in agent.sids], dtype=np.float32)
            t.pushed = True
            ready.append(t)
        if boundary and trs:
            last = trs[-1]
            if last.resolved and not last.pushed:
                I_snap = dict(self._slot_log.get(current_step) or {})
                if last.complied and last.chosen_indices:
                    winner_sid = agent.sids[last.chosen_indices[0]]
                    if winner_sid in I_snap:
                        I_snap[winner_sid] = I_snap[winner_sid] + 1.0
                last.I_raw = np.array([I_snap.get(sid, 0.0) for sid in agent.sids], dtype=np.float32)
                last.pushed = True
                ready.append(last)
        k = 0
        while k < len(trs) and trs[k].pushed:
            k += 1
        agent.transitions = trs[k:]
        return ready

    def _compute_beta(self, returns):
        K = self.n_critics
        ret_mean = np.asarray(returns, dtype=np.float64).mean(axis=0)
        self._last_ret_mean = ret_mean
        if not self._fixed_ret_best:
            self._ret_best = np.maximum(self._ret_best, ret_mean)
        if self.beta_mode != "gap_ratio":
            return np.full(K, 1.0 / K, dtype=np.float64)
        gap = (self._ret_best - ret_mean) / (np.abs(self._ret_best) + 1e-8)
        gap = np.clip(gap, 0.0, 10.0)
        z = gap / max(self.beta_sigma, 1e-8)
        z -= z.max()
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def _ppo_update(self, transitions):
        if len(transitions) < 2:
            return None
        transitions = sorted(transitions, key=lambda t: t.step)
        returns, adv = compute_gae(transitions, self.gamma, self.lam,
                                   max_step_gap=self.max_step_gap)
        adv = (adv - adv.mean(axis=0, keepdims=True)) / (adv.std(axis=0, keepdims=True) + 1e-8)
        beta = self._compute_beta(returns)
        self._last_beta = beta
        adv_combined = adv @ beta.astype(np.float32)

        obs_b = torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32)
        mask_b = torch.as_tensor(np.stack([t.mask for t in transitions]), dtype=torch.bool)
        primary_b = torch.as_tensor([t.chosen_indices[0] for t in transitions], dtype=torch.long)
        I_b = torch.as_tensor(np.stack([t.I_raw for t in transitions]), dtype=torch.float32)
        old_logp = torch.as_tensor(np.array([t.logp for t in transitions]), dtype=torch.float32)
        _ph = [t.pref_hist for t in transitions]
        pref_b = (torch.as_tensor(np.stack(_ph), dtype=torch.float32)
                  if all(x is not None for x in _ph) else None)
        ret_b = torch.as_tensor(returns, dtype=torch.float32)
        adv_b = torch.as_tensor(adv_combined, dtype=torch.float32)

        B = len(transitions)
        idx = np.arange(B)
        last = {}
        grad_norm = 0.0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, B, self.minibatch):
                mb = idx[start:start + self.minibatch]
                logits = self.actor(obs_b[mb], mask_b[mb],
                                    None if pref_b is None else pref_b[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(primary_b[mb])
                ent = dist.entropy()
                if hasattr(self.critic, "pref_lstm"):
                    value, _ = self.critic(obs_b[mb], mask_b[mb], I_b[mb],
                                           None if pref_b is None else pref_b[mb])
                else:
                    value, _ = self.critic(obs_b[mb], mask_b[mb], I_b[mb])

                ratio = torch.exp(logp - old_logp[mb])
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = nn.functional.mse_loss(value, ret_b[mb])
                ent_loss = -ent.mean()
                loss = pi_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss
                if not torch.isfinite(loss):
                    continue
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                grad_norm = float(gn)
                self.opt.step()
                last = {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(),
                        "entropy": ent.mean().item(), "loss": loss.item(),
                        "grad_norm": grad_norm, "beta": beta.tolist(),
                        "ret_mean": self._last_ret_mean.tolist()}
            if self.target_kl is not None:
                with torch.no_grad():
                    logits = self.actor(obs_b, mask_b, pref_b)
                    dist = torch.distributions.Categorical(logits=logits)
                    nlp = dist.log_prob(primary_b)
                    kl = float((old_logp - nlp).mean())
                if kl > 1.5 * self.target_kl:
                    break
        return last

    def _run_one_chunk(self, sim, agent, step: int, chunk: int):
        boundary = False
        for _ in range(chunk):
            sim.step_once(step, agent=agent)
            self._slot_log.record(step, sim)
            step += 1
            if step >= sim.max_steps:
                boundary = True
                break
        if boundary:
            for t in agent.transitions:
                t.resolved = True
            if agent.transitions:
                agent.transitions[-1].done = True

        ready = self._push_ready_pairs(agent, current_step=step, boundary=boundary)
        stats = self._ppo_update(ready) if ready else None

        info = None
        if ready:
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            utils = np.array([s.get_utilization() for s in sim.spklus.values()])
            info = {"n_ready": len(ready), "n_backlog": len(agent.transitions),
                    "gini_served": _gini(served), "gini_util": _gini(utils), **(stats or {})}

        if boundary:
            sim = self._carry_forward(sim, agent)
            step = 0
        return sim, agent, step, boundary, info

    def train(self, n_updates: int):
        chunk = self.rollout_steps
        sim = self._fresh_sim()
        # Mode pref DITURUNKAN dari aktor (bukan diteruskan terpisah) -- menjamin dimensi
        # `pref_hist` yg dibangun rollout SELALU cocok dgn yg diharapkan `pref_lstm`.
        _bb = getattr(self.actor, "backbone", None)
        agent = MasterHybridPPORolloutAgent(self.actor, self.critic, sim, self.rc,
                                            equity_calc=self.equity_calc,
                                            stream_select=self.stream_select,
                                            pref_feature_mode=getattr(_bb, "pref_feature_mode", False),
                                            pref_pair_outcome=getattr(_bb, "pref_pair_outcome", False),
                                            pref_hist_k=getattr(_bb, "pref_hist_k", None),
                                            accept_stream=self.accept_stream,
                                            pure_streams=self.pure_streams)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            sim, agent, step, boundary, info = self._run_one_chunk(sim, agent, step, chunk)
            self._it_global += 1
            if info is not None:
                self.history.append({"iter": it, **info})
                if self.verbose:
                    print(f"[hybrid-ppo {self.mode} chunk {it:3d}] ready={info['n_ready']} "
                         f"backlog={info['n_backlog']} pi_loss={info.get('pi_loss', 0):+.4f} "
                         f"v_loss={info.get('v_loss', 0):.4f} beta={info.get('beta')} | "
                         f"gini_util={info['gini_util']:.3f}" + (" |PASS-BARU" if boundary else ""))
        if self.mode == "pretrain_specialist":
            rets = [h["ret_mean"][0] for h in self.history if "ret_mean" in h]
            tail_n = max(1, len(rets) // 5)
            r_star = float(np.mean(rets[-tail_n:])) if rets else 0.0
            return self.actor, self.critic, r_star
        return self.actor, self.critic
