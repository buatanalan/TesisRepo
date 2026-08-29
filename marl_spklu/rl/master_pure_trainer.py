"""Trainer MASTER **benar-benar murni** (2026-08-28) -- kelas BARU, TIDAK mewarisi
`MasterDDPGRolloutAgent`/`MasterDDPGReplayBuffer`/`MasterDDPGTrainer` (`master_ddpg_
trainer.py`) sama sekali, hanya `RLRolloutAgent` (kelas dasar generik dipakai SEMUA
lengan RL repo ini, bukan kelas khusus MASTER). Dipasangkan dgn `master_pure_policy.py`.

Diverifikasi thd `2102.07359v1.pdf` §3.1-3.3, Algoritma 1 -- rujukan lengkap di
`Eksekusi_RL/ARSITEKTUR_MASTER_REFERENSI.md`. Menahan KEEMPAT mekanisme MASTER dgn
perbaikan atas deviasi `master_ddpg_trainer.py` lama:

  1. Bid TERTINGGI menang (argmax, §3.1) -- bukan terendah.
  6. `I^i_t` MENTAH bisa NEGATIF (antrean, §3.2.2) -- dihitung dari `queues` SPKLU
     sungguhan (queue_length), BUKAN cuma slots_avail_norm ∈[0,1] spt lama.
     Pengaruh q_t SENDIRI DIHAPUS dari I^i_t (kutipan paper: "we erase the influence
     of q_t for I^i_t") -- lih. `_push_ready_pairs` utk implementasi persis.
  8. Dynamic Gradient Re-weighting SUNGGUHAN (Pers. 13): gap-ratio dihitung thd
     kritik+aktor SPESIALIS per-objektif yg SUDAH DILATIH TERPISAH lebih dulu
     (`mode="pretrain_specialist"`), dievaluasi PADA STATE x_t YANG SAMA dgn AKSI
     SPESIALIS sendiri -- BUKAN proksi running-max spt `master_ddpg_trainer.py` lama.
  9. Update target SELALU soft (tau, Algoritma 1 Baris 26-31) -- tak ada opsi
     hard-copy periodik (itu tambahan repo lama, bukan bagian Algoritma 1 asli).

Hyperparameter baku MENGIKUTI §4.1.2 (bukan default lama):
  d=30 menit, sigma=0.2, gamma=0.99, buffer=1000, batch=32, tau=0.001, lr=5e-4,
  hidden=64 (aktor & kritik).

Objektif ke-2 (pengganti CP, keputusan bersama 2026-08-28): Gini/pemerataan
(STREAM_GLOBAL) -- BUKAN CWT/CP literal paper (CP tak dimodelkan simulator ini).
STREAM_INDIVIDUAL (wait) dipetakan sbg analog CWT.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.rollout import RLRolloutAgent, STREAM_INDIVIDUAL, STREAM_GLOBAL, N_REWARD_STREAMS, _gini
from marl_spklu.rl.master_paper_obs import build_joint_obs_master, STATION_FEAT_DIM_MASTER
from marl_spklu.rl.master_pure_policy import MasterPureActor, MasterPureCritic
from marl_spklu.rl.rewards import RewardCalculator


def snapshot_slots_raw(sim) -> dict:
    """I^i_t MENTAH (Pers. 10, §3.2.2) -- `(cap_total - charging_total) - panjang_
    antrean`. Charging SELALU <= cap_total (kapasitas ditegakkan simulator), jadi
    `cap-charging` >= 0 selalu; begitu stasiun PENUH (cap-charging=0) DAN ada EV
    di `spklu.queues`, hasilnya jadi NEGATIF -- persis semantik paper: "the number
    of available charging spots can be negative, means the number of EVs queuing
    at the station." TIDAK dinormalisasi ke [0,1] (beda dari `snapshot_slots_avail`
    lama yg dipakai `future_avail`)."""
    out = {}
    for sid, s in sim.spklus.items():
        cap_total = sum(s.capacities.values())
        charging_total = sum(len(c) for c in s.charging.values())
        queue_total = sum(len(q) for q in s.queues.values())
        out[sid] = float((cap_total - charging_total) - queue_total)
    return out


class _SlotRawLog:
    """Snapshot {sid: I^i_t MENTAH} per langkah -- fondasi Delayed Access Strategy,
    versi MENTAH (bukan ternormalisasi) utk `MasterPureCritic`."""

    def __init__(self, maxlen: int = 64):
        self.by_step = {}
        self.order = deque(maxlen=maxlen)

    def record(self, step: int, sim):
        self.by_step[step] = snapshot_slots_raw(sim)
        self.order.append(step)
        while len(self.by_step) > self.order.maxlen:
            self.by_step.pop(self.order[0], None)

    def get(self, step: int):
        return self.by_step.get(step)

    def reset(self):
        self.by_step.clear(); self.order.clear()


class MasterPureTransition:
    """Duck-type Transition (rollout.py), field-demi-field spy hook RLRolloutAgent
    bekerja tanpa modifikasi. `obs`=joint (N,7) §3.1 MURNI (TANPA +EV)."""

    def __init__(self, obs, mask, action, step, primary_idx, pref_hist=None):
        self.obs = obs; self.mask = mask; self.action = action
        # WAJIB disimpan (2026-08-29) -- lih. catatan identik di MasterHybridPPOTransition:
        # tanpa ini langkah `_update` memanggil aktor TANPA pref_hist, sehingga parameter
        # preferensi tak pernah masuk graf gradien & modul P efektif mati.
        self.pref_hist = pref_hist
        self.step = step
        self.chosen_indices = [int(primary_idx)]
        self.reward_streams = np.zeros(N_REWARD_STREAMS, dtype=np.float64)
        self.done = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.resolved = False; self.pushed = False; self.flock_penalty = 0.0
        self.next_obs = None; self.next_mask = None; self.I_raw = None

    @property
    def reward(self) -> float:
        return float(self.reward_streams.sum())

    def reward_vec(self, n_critics: int, stream_select=None) -> np.ndarray:
        if stream_select is not None:
            return np.array([self.reward_streams[stream_select]], dtype=np.float64)
        if n_critics == 1:
            return np.array([self.reward_streams.sum()], dtype=np.float64)
        if n_critics != N_REWARD_STREAMS:
            raise ValueError(f"n_critics={n_critics} != N_REWARD_STREAMS={N_REWARD_STREAMS}")
        return self.reward_streams

    def add_reward(self, value: float, stream: int = STREAM_INDIVIDUAL) -> None:
        self.reward_streams[stream] += value


class MasterPureRolloutAgent(RLRolloutAgent):
    """Eksekusi DESENTRALISASI, observasi §3.1 MURNI (7 fitur, TANPA fitur pemohon --
    `build_joint_obs_master`, BUKAN varian +EV). Pemenang = bid TERTINGGI (argmax,
    §3.1) -- KEBALIKAN `MasterDDPGRolloutAgent` lama."""

    def __init__(self, actor, sim, reward_calc, forecaster=None, noise_std: float = 8.0,
                k: int = 3, equity_calc=None, pref_feature_mode: bool = False,
                pref_pair_outcome: bool = False):
        # `pref_pad_right=True` -- lihat catatan identik di `MasterHybridPPORolloutAgent`
        # (bug padding vs pack_padded_sequence, ditemukan 2026-08-29). Tak berpengaruh
        # bila aktor tanpa modul P (jalur pref tak dieksekusi sama sekali).
        super().__init__(actor, sim, reward_calc, forecaster, k=k, equity_calc=equity_calc,
                         pref_feature_mode=pref_feature_mode,
                         pref_pair_outcome=pref_pair_outcome, pref_pad_right=True)
        self.actor = actor
        self.noise_std = float(noise_std)
        self.ou_theta = 0.15
        self._ou_state = np.zeros(self.N, dtype=np.float32)

    def _sample_noise(self):
        self._ou_state += (self.ou_theta * (0.0 - self._ou_state)
                           + self.noise_std * np.random.normal(size=self.N))
        return self._ou_state.copy()

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        _rich_obs_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)
        joint_obs = build_joint_obs_master(self.sim, self.sids, time_now)   # (N,7) §3.1 MURNI
        mask = self._feasible_mask(feasible_ids)

        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        # Modul P (opsional, 2026-08-29 Master-Hybrid): _use_pref/_build_pref_hist
        # DIWARISI RLRolloutAgent.__init__ (auto-deteksi hasattr(actor,'pref_lstm')) --
        # aktor LAMA (`MasterPureActor`, tanpa pref_lstm) tak terpengaruh, tetap
        # dipanggil obs-saja spt semula (kompatibel mundur penuh).
        pref_hist = None
        with torch.no_grad():
            if self._use_pref:
                pref_hist = self._build_pref_hist(user)
                pref_hist_t = torch.as_tensor(pref_hist, dtype=torch.float32).unsqueeze(0)
                bids = self.actor(obs_t, mask_t, pref_hist_t).squeeze(0).numpy()
            else:
                bids = self.actor(obs_t).squeeze(0).numpy()
        raw_bids = bids.copy()
        if self.noise_std > 0:
            bids = bids + self._sample_noise()   # tanpa clip -- bid abstrak, bukan waktu

        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            chosen_order = []
        else:
            k_eff = max(1, min(self.k, int(feasible_idx.size)))
            # PAPER §3.1: r_c_t = argmax(u_t) -- bid TERTINGGI menang (argsort DESCENDING).
            order = feasible_idx[np.argsort(-bids[feasible_idx], kind="stable")]
            chosen_order = [int(i) for i in order[:k_eff]]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_order}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf")
                           for sid in feasible_ids}

        primary_idx = chosen_order[0] if chosen_order else int(feasible_idx[0]) if feasible_idx.size else 0
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)
        recs = [self.sids[i] for i in chosen_order]

        tr = MasterPureTransition(joint_obs, mask, raw_bids.astype(np.float32),
                                  self.sim.current_step, primary_idx, pref_hist=pref_hist)
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterPureInferenceAgent:
    """Evaluasi BERSIH (tanpa noise eksplorasi, tanpa akumulasi transisi/reward) --
    dipasang langsung ke `Simulator`, pola sama `MasterDDPGInferenceAgent` lama.
    Bid TERTINGGI menang (argmax, §3.1) -- KEBALIKAN versi lama. Observasi §3.1
    MURNI (7 fitur, `build_joint_obs_master`, TANPA +EV)."""

    def __init__(self, actor, forecaster=None, k: int = 3):
        self.actor = actor
        self.actor.eval()
        from marl_spklu.rl.forecaster import FormulaForecaster
        self.forecaster = forecaster or FormulaForecaster()
        self.k = int(k)
        self.sids = None
        self.sid_to_idx = None
        self.N = None

    def bind_to_sim(self, sim):
        self.sim = sim
        self.sids = list(sim.spklus.keys())
        self.sid_to_idx = {s: i for i, s in enumerate(self.sids)}
        self.N = len(self.sids)

    def get_recommendation(self, feasible_spklus: dict):
        assert self.sids is not None, "panggil bind_to_sim(sim) sebelum sim.run(agent=...)"
        time_now = self.sim.current_step * self.sim.dt_minutes
        joint_obs = build_joint_obs_master(self.sim, self.sids, time_now)
        mask = np.zeros(self.N, dtype=bool)
        for sid in feasible_spklus:
            if sid in self.sid_to_idx:
                mask[self.sid_to_idx[sid]] = True

        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            bids = self.actor(obs_t).squeeze(0).numpy()

        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            return []
        k_eff = max(1, min(self.k, int(feasible_idx.size)))
        order = feasible_idx[np.argsort(-bids[feasible_idx], kind="stable")]   # TERTINGGI menang
        chosen = [int(i) for i in order[:k_eff]]
        return [self.sids[i] for i in chosen]

    def predict_waits(self, feasible_spklus: dict):
        time_now = self.sim.current_step * self.sim.dt_minutes
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        return self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)


class MasterPureReplayBuffer:
    """Berdiri sendiri (BUKAN reuse `MasterDDPGReplayBuffer`) -- field `I_raw`
    (menggantikan `future_avail`), TANPA `gini_now` (global_scalar dihapus,
    lih. `master_pure_policy.py` poin 7)."""

    def __init__(self, capacity: int = 1000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, mask, action, reward_vec, next_obs, next_mask, I_raw, done,
             pref_hist=None, next_pref_hist=None):
        self.buf.append((obs, mask, action, reward_vec, next_obs, next_mask, I_raw, done,
                         pref_hist, next_pref_hist))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        cols = list(zip(*batch))
        (obs, mask, action, reward_vec, next_obs, next_mask, I_raw, done,
         pref_hist, next_pref_hist) = cols
        # `pref_hist` di-stack HANYA bila SELURUH sampel batch punya (aktor bermodul-P);
        # None utk aktor polos -> jalur lama tak berubah sama sekali.
        ph = np.stack(pref_hist) if all(x is not None for x in pref_hist) else None
        nph = np.stack(next_pref_hist) if all(x is not None for x in next_pref_hist) else None
        return (np.stack(obs), np.stack(mask), np.stack(action), np.stack(reward_vec),
                np.stack(next_obs), np.stack(next_mask), np.stack(I_raw),
                np.array(done, dtype=np.float32), ph, nph)

    def __len__(self):
        return len(self.buf)


class MasterPureTrainer:
    """Trainer BARU berdiri sendiri. Dua mode:

    `mode="pretrain_specialist"` (`n_critics=1`, `stream_select` WAJIB diisi 0/1):
        Melatih SATU pasang aktor+kritik objektif-tunggal SAMPAI CONVERGENT --
        inilah `Q*_b*`/`b*` yg dibutuhkan Pers. (13). Dipanggil 2x (sekali per
        objektif: wait, gini) SEBELUM `mode="dgr"` dimulai.

    `mode="dgr"` (`n_critics=2`, `specialists` WAJIB diisi -- list 2 tuple
        `(aktor_beku, kritik_beku)` hasil `mode="pretrain_specialist"`):
        Gap-ratio (Pers. 13) SUNGGUHAN: `Q*_k(x*_t)|a=b*_k(o)` (kritik+aktor
        SPESIALIS k, action MILIK SPESIALIS itu, state x_t SAMA) dibandingkan
        `Q_k(x_t)|a=b(o)` (kritik+aktor multi-objektif SEKARANG) -- BUKAN
        running-max proksi.
    """

    def __init__(self, dataset_path, mode: str, rollout_steps: int = 96,
                gamma: float = 0.99, lr: float = 5e-4, batch_size: int = 32,
                buffer_capacity: int = 1000, tau: float = 0.001,
                noise_start: float = 12.0, noise_end: float = 2.0, noise_decay_frac: float = 0.6,
                reward_calc=None, seed: int = 0, verbose: bool = True,
                updates_per_chunk: int = 20, equity_calc=None,
                delay_minutes: float = 30.0, hidden: int = 64,
                beta_mode: str = "gap_ratio", beta_sigma: float = 0.2,
                n_critics: int = None, stream_select: int = None, specialists: list = None,
                actor_cls=None, actor_kwargs: dict = None):
        """`actor_cls`/`actor_kwargs` (2026-08-29, Master-Hybrid): opsional, ganti
        `MasterPureActor` baku dgn kelas lain (mis. `MasterHybridDDPGActor`, modul
        P+attention) -- TAMBAHAN murni, BAKU `None` = perilaku lama TAK BERUBAH."""
        assert mode in ("pretrain_specialist", "dgr"), f"mode={mode!r} tak dikenal"
        self.mode = mode
        self.dataset_path = dataset_path
        self.equity_calc = equity_calc
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        assert tau is not None and tau > 0, (
            "Algoritma 1 (Baris 26-31) SELALU soft-update -- tau wajib diisi, "
            "tak ada opsi hard-copy periodik di paper.")
        self.tau = float(tau)
        self.updates_per_chunk = int(updates_per_chunk)
        self.noise_start = float(noise_start)
        self.noise_end = float(noise_end)
        self.noise_decay_frac = float(noise_decay_frac)
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        if mode == "pretrain_specialist":
            assert stream_select in (0, 1), (
                "mode='pretrain_specialist' WAJIB --stream-select 0 (wait/CWT-analog) "
                "atau 1 (gini/pemerataan, pengganti CP)")
            self.n_critics = 1
            self.stream_select = int(stream_select)
            self.specialists = None
        else:
            assert specialists is not None and len(specialists) == 2, (
                "mode='dgr' WAJIB --specialists: list 2 (aktor_beku,kritik_beku) "
                "hasil mode='pretrain_specialist' (stream 0 & 1)")
            self.n_critics = 2
            self.stream_select = None
            self.specialists = specialists
            for act_s, crit_s in specialists:
                act_s.eval(); crit_s.eval()
                for p in act_s.parameters(): p.requires_grad_(False)
                for p in crit_s.parameters(): p.requires_grad_(False)

        self.beta_mode = str(beta_mode)
        self.beta_sigma = float(beta_sigma)

        sim0 = self._fresh_sim()
        self.dt_minutes = sim0.dt_minutes
        self.delay_steps = max(1, round(float(delay_minutes) / self.dt_minutes))
        self._slot_log = _SlotRawLog(maxlen=self.rollout_steps + self.delay_steps + 4)

        self.N = len(sim0.spklus)
        if actor_cls is None:
            self.actor = MasterPureActor(STATION_FEAT_DIM_MASTER, hidden=hidden)
            self.actor_target = MasterPureActor(STATION_FEAT_DIM_MASTER, hidden=hidden)
        else:
            # Kelas baru (mis. MasterHybridDDPGActor): argumen pertama n_spklu, BUKAN
            # station_feat_dim spt `MasterPureActor` -- signature BERBEDA sengaja.
            # BUG (2026-08-29, ditemukan smoke-test): actor_target SEBELUMNYA tetap
            # hardcode MasterPureActor -- state_dict actor_cls baru tak cocok
            # dimuat ke situ (RuntimeError key mismatch). actor_target WAJIB kelas
            # SAMA persis dgn actor.
            self.actor = actor_cls(self.N, **(actor_kwargs or {}))
            self.actor_target = actor_cls(self.N, **(actor_kwargs or {}))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = MasterPureCritic(STATION_FEAT_DIM_MASTER, hidden=hidden, n_critics=self.n_critics)
        self.critic_target = MasterPureCritic(STATION_FEAT_DIM_MASTER, hidden=hidden, n_critics=self.n_critics)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = MasterPureReplayBuffer(buffer_capacity)

        self.history = []
        self._n_updates = 0
        self._it_global = 0
        self._total_chunks = None

    def _fresh_sim(self):
        from marl_spklu.rl.training import _fresh_sim
        return _fresh_sim(self.dataset_path)

    def _noise_std(self):
        horizon = max(1.0, self.noise_decay_frac * (self._total_chunks or 1))
        frac = min(1.0, self._it_global / horizon)
        return self.noise_start + frac * (self.noise_end - self.noise_start)

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

    def _push_ready_pairs(self, agent, current_step: int, boundary: bool = False) -> int:
        """Delayed Access Strategy (§3.2.2) + PENGHAPUSAN PENGARUH q_t SENDIRI dari
        I^i_t -- kutipan paper: "we erase the influence of q_t for I^i_t". Kalau q_t
        PATUH (`t.complied`) ke stasiun pemenang (`t.chosen_indices[0]`), EV itu
        sendiri menempati 1 slot di sana selama sesi charging -- ini "mengotori"
        I^i_t stasiun tsb dgn kontribusi q_t sendiri. Dihapus dgn menambah kembali
        +1 pada I^i_t stasiun pemenang SAAT q_t patuh (pendekatan: hapus 1 slot
        occupancy MILIK q_t, TANPA pencocokan durasi presisi -- keterbatasan
        didokumentasikan, bukan disembunyikan)."""
        trs = agent.transitions
        n_new = 0
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
                    I_snap[winner_sid] = I_snap[winner_sid] + 1.0   # hapus pengaruh q_t sendiri
            nx = trs[i + 1]
            I_vec = np.array([I_snap.get(sid, 0.0) for sid in agent.sids], dtype=np.float32)
            self.buffer.push(t.obs, t.mask, t.action,
                             t.reward_vec(self.n_critics, self.stream_select),
                             nx.obs, nx.mask, I_vec, 1.0 if t.done else 0.0,
                             pref_hist=t.pref_hist, next_pref_hist=nx.pref_hist)
            t.pushed = True
            n_new += 1
        if boundary and trs:
            last = trs[-1]
            if last.resolved and not last.pushed:
                I_snap = dict(self._slot_log.get(current_step) or {})
                if last.complied and last.chosen_indices:
                    winner_sid = agent.sids[last.chosen_indices[0]]
                    if winner_sid in I_snap:
                        I_snap[winner_sid] = I_snap[winner_sid] + 1.0
                I_vec = np.array([I_snap.get(sid, 0.0) for sid in agent.sids], dtype=np.float32)
                self.buffer.push(last.obs, last.mask, last.action,
                                 last.reward_vec(self.n_critics, self.stream_select),
                                 last.obs, last.mask, I_vec, 1.0,
                                 pref_hist=last.pref_hist, next_pref_hist=last.pref_hist)
                last.pushed = True
                n_new += 1
        k = 0
        while k < len(trs) and trs[k].pushed:
            k += 1
        agent.transitions = trs[k:]
        return n_new

    def _compute_beta_dgr(self, obs_t, mask_t, I_t, pref_t=None):
        """Pers. (13)-(14) SUNGGUHAN -- gap-ratio thd SPESIALIS beku, dievaluasi pada
        state (obs_t, I_t) YANG SAMA dgn data batch saat ini, TAPI dgn AKSI MILIK
        SPESIALIS `b*_k(o)` sendiri (bukan aksi kebijakan multi-objektif sekarang)."""
        if self.mode != "dgr":
            return np.ones(1, dtype=np.float64)
        gaps = []
        with torch.no_grad():
            for k, (act_s, crit_s) in enumerate(self.specialists):
                a_star = act_s(obs_t, mask_t, pref_t)                              # b*_k(o^i_t)
                q_star, _ = crit_s(obs_t, a_star, mask_t, I_t)       # Q*_k(x*_t), (B,1)
                a_now = self.actor(obs_t, mask_t, pref_t)
                q_now, _ = self.critic(obs_t, a_now, mask_t, I_t)    # Q_k(x_t) dgn kritik MULTI-obj
                q_star_m = q_star.mean().item()
                q_now_k = q_now[:, k].mean().item()
                gap = (q_star_m - q_now_k) / (abs(q_star_m) + 1e-8)   # Pers.13
                gaps.append(gap)
        gap = np.clip(np.array(gaps, dtype=np.float64), 0.0, 10.0)
        z = gap / max(self.beta_sigma, 1e-8)
        z -= z.max()
        e = np.exp(z)
        return e / (e.sum() + 1e-12)                                   # Pers.14

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return None
        (obs, mask, action, reward_vec, next_obs, next_mask, I_raw, done,
         pref_hist, next_pref_hist) = self.buffer.sample(self.batch_size)
        pref_t = None if pref_hist is None else torch.as_tensor(pref_hist, dtype=torch.float32)
        next_pref_t = (None if next_pref_hist is None
                       else torch.as_tensor(next_pref_hist, dtype=torch.float32))
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        action_t = torch.as_tensor(action, dtype=torch.float32)
        reward_t = torch.as_tensor(reward_vec, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        next_mask_t = torch.as_tensor(next_mask, dtype=torch.bool)
        I_t = torch.as_tensor(I_raw, dtype=torch.float32)
        done_t = torch.as_tensor(done, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            next_action = self.actor_target(next_obs_t, next_mask_t, next_pref_t)
            q_next, _ = self.critic_target(next_obs_t, next_action, next_mask_t, I_t)
            target = reward_t + self.gamma * (1.0 - done_t) * q_next
        q_sa, _ = self.critic(obs_t, action_t, mask_t, I_t)
        critic_loss = nn.functional.mse_loss(q_sa, target)
        self.opt_critic.zero_grad(); critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_critic.step()

        beta = self._compute_beta_dgr(obs_t, mask_t, I_t, pref_t)
        beta_t = torch.as_tensor(beta, dtype=torch.float32)
        actor_action = self.actor(obs_t, mask_t, pref_t)
        q_pi, _ = self.critic(obs_t, actor_action, mask_t, I_t)
        actor_loss = -(q_pi @ beta_t).mean()
        self.opt_actor.zero_grad(); actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_actor.step()

        self._n_updates += 1
        with torch.no_grad():   # Algoritma 1 Baris 26-31: SELALU soft-update
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()),
               "critic_grad": float(critic_grad), "actor_grad": float(actor_grad), "beta": beta.tolist()}

    def _run_one_chunk(self, sim, agent, step: int, chunk: int, do_update: bool):
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

        n_new = self._push_ready_pairs(agent, current_step=step, boundary=boundary)
        stats = None
        if do_update:
            for _ in range(min(n_new, self.updates_per_chunk)):
                s = self._update()
                if s is not None:
                    stats = s

        info = None
        if n_new:
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            utils = np.array([s.get_utilization() for s in sim.spklus.values()])
            info = {"n_new": n_new, "n_backlog": len(agent.transitions),
                    "noise_std": agent.noise_std, "buffer_size": len(self.buffer),
                    "gini_served": _gini(served), "gini_util": _gini(utils), **(stats or {})}

        if boundary:
            sim = self._carry_forward(sim, agent)
            step = 0
        return sim, agent, step, boundary, info

    def train(self, n_updates: int):
        chunk = self.rollout_steps
        self._total_chunks = int(n_updates)
        sim = self._fresh_sim()
        # Mode pref DITURUNKAN dari aktor (lih. catatan identik di trainer Hybrid-PPO) --
        # `MasterPureActor` polos tak punya `backbone`, jadi getattr -> False (perilaku lama).
        _bb = getattr(self.actor, "backbone", None)
        agent = MasterPureRolloutAgent(self.actor, sim, self.rc, noise_std=self._noise_std(),
                                       equity_calc=self.equity_calc,
                                       pref_feature_mode=getattr(_bb, "pref_feature_mode", False),
                                       pref_pair_outcome=getattr(_bb, "pref_pair_outcome", False))
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            agent.noise_std = self._noise_std()
            sim, agent, step, boundary, info = self._run_one_chunk(sim, agent, step, chunk, do_update=True)
            self._it_global += 1
            if info is not None:
                self.history.append({"iter": it, **info})
                if self.verbose:
                    print(f"[{self.mode} chunk {it:3d}] new={info['n_new']} backlog={info['n_backlog']} "
                         f"critic_loss={info.get('critic_loss', 0):.4f} "
                         f"actor_loss={info.get('actor_loss', 0):+.4f} beta={info.get('beta')} | "
                         f"gini_util={info['gini_util']:.3f}" + (" |PASS-BARU" if boundary else ""))
        return self.actor, self.critic
