"""Lengan MASTER "asli" -- agen rollout desentralisasi + trainer DDPG menahan KEEMPAT
mekanisme inti MASTER (Zhang dkk., WWW 2021) sekaligus:

    1. Stasiun sbg agen desentralisasi         -> master_paper_obs.py (obs §3.1 murni)
    2. Bidding game + Centralized Attentive
       Critic                                  -> master_ddpg_policy.py
    3. Delayed Access Strategy                 -> _SlotAvailLog + _push_ready_pairs di sini
    4. Multi-Critics + Dynamic Gradient
       Re-weighting                            -> _compute_beta_ddpg di sini

BATAS KLAIM (baca sebelum mengutip sbg "replikasi penuh"): tulang punggung tetap DDPG
diskrit-per-langkah (bukan async multi-agent penuh), reward memakai dekomposisi
2-aliran REPO INI (`rollout.STREAM_INDIVIDUAL/STREAM_GLOBAL`, bukan CWT/CP MASTER
literal -- CP tak dimodelkan simulator ini, lihat master_paper_obs.py), lingkungan &
populasi pengguna tetap Klaster 12 (bukan kota metropolitan berjuta EV). Yang ditahan
SETIA: bentuk observasi (§3.1), bentuk aksi (bid kontinu per-stasiun), arsitektur
kritik (atensi terpusat multi-kepala), dan KEDUA mekanisme koordinasi (delayed access +
gradient reweighting) -- empat klaim user yg diminta diimplementasikan, seluruhnya ada
di sini, bukan sebagian spt `master_bidding_policy.py`.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.rollout import RLRolloutAgent, STREAM_INDIVIDUAL, STREAM_GLOBAL, N_REWARD_STREAMS, _gini
from marl_spklu.rl.master_paper_obs import build_joint_obs_master, snapshot_slots_avail, STATION_FEAT_DIM_MASTER
from marl_spklu.rl.master_ddpg_policy import MasterStationActor, MasterAttentiveCritic
from marl_spklu.rl.rewards import RewardCalculator


class MasterDDPGTransition:
    """Duck-type Transition (rollout.py) -- REPLIKA field-demi-field spy hook warisan
    RLRolloutAgent (on_decision/on_step_end/on_charge_complete, TAK diubah di sini)
    bekerja tanpa modifikasi. `obs`=joint (N,7) §3.1, `action`=bid mentah (N,) SEBELUM
    noise (dipakai training; bid+noise sesungguhnya yg dipakai seleksi disimpan di
    `bids_noisy`, hanya utk logging)."""

    def __init__(self, obs, mask, action, step, primary_idx, gini_now: float):
        self.obs = obs; self.mask = mask; self.action = action
        self.step = step
        self.chosen_indices = [int(primary_idx)]   # dibaca on_step_end/on_decision (compat)
        self.reward_streams = np.zeros(N_REWARD_STREAMS, dtype=np.float64)
        self.done = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.resolved = False; self.pushed = False; self.flock_penalty = 0.0
        self.gini_now = float(gini_now)   # global_scalar kritik (privileged, snapshot saat keputusan)
        # Diisi trainer saat dipasangkan (s,a,r,s',future_avail):
        self.next_obs = None; self.next_mask = None; self.future_avail = None

    @property
    def reward(self) -> float:
        return float(self.reward_streams.sum())

    def reward_vec(self, n_critics: int) -> np.ndarray:
        if n_critics == 1:
            return np.array([self.reward_streams.sum()], dtype=np.float64)
        if n_critics != N_REWARD_STREAMS:
            raise ValueError(f"n_critics={n_critics} != N_REWARD_STREAMS={N_REWARD_STREAMS}")
        return self.reward_streams

    def add_reward(self, value: float, stream: int = STREAM_INDIVIDUAL) -> None:
        self.reward_streams[stream] += value


class _SlotAvailLog:
    """Snapshot {sid: slots_avail_norm} PER LANGKAH -- fondasi Delayed Access Strategy
    (MASTER §4.3): kritik hanya boleh melihat ketersediaan slot BENAR di t+d SETELAH
    simulasi sungguhan mencapai t+d, bukan diprediksi/dikira-kira. Jendela dibatasi
    (`maxlen`) krn hanya perlu menyimpan ~delay_steps snapshot terakhir."""

    def __init__(self, maxlen: int = 64):
        self.by_step = {}
        self.order = deque(maxlen=maxlen)

    def record(self, step: int, sim):
        self.by_step[step] = snapshot_slots_avail(sim)
        self.order.append(step)
        while len(self.by_step) > self.order.maxlen:
            oldest = self.order[0]
            self.by_step.pop(oldest, None)

    def get(self, step: int):
        return self.by_step.get(step)

    def reset(self):
        self.by_step.clear(); self.order.clear()


class MasterDDPGRolloutAgent(RLRolloutAgent):
    """Eksekusi DESENTRALISASI: tiap stasiun feasible menghitung bid-nya SENDIRI dari
    o^i_t (§3.1) via `MasterStationActor` (bobot dibagi, dipanggil batched tapi scr
    matematis identik per-baris -- lihat docstring kelas itu). `k` bid TERENDAH
    menang. EstWait yg DITAMPILKAN ke pengguna tetap forecaster jujur (Spesifikasi_
    Teknis_RL.md v2) -- bid HANYA dipakai peringkat internal, sama batas yg dipegang
    `master_bidding_policy.py`."""

    def __init__(self, actor, sim, reward_calc, forecaster=None, noise_std: float = 8.0,
                k: int = 3, equity_calc=None):
        super().__init__(actor, sim, reward_calc, forecaster, k=k, equity_calc=equity_calc)
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

        # default_idx/wait_hat: bookkeeping REWARD/CTDE saja (dipakai on_decision via
        # self._pending, lihat rollout.py) -- TIDAK bocor ke observasi aktor (dibangun
        # terpisah di bawah, murni §3.1, tanpa user).
        _rich_obs_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)

        joint_obs = build_joint_obs_master(self.sim, self.sids, time_now)   # (N,7)
        mask = self._feasible_mask(feasible_ids)

        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            bids = self.actor(obs_t).squeeze(0).numpy()
        raw_bids = bids.copy()
        if self.noise_std > 0:
            bids = np.clip(bids + self._sample_noise(), 0.0, self.actor.bid_max)

        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            chosen_order = []
        else:
            k_eff = max(1, min(self.k, int(feasible_idx.size)))
            order = feasible_idx[np.argsort(bids[feasible_idx], kind="stable")]
            chosen_order = [int(i) for i in order[:k_eff]]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_order}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf")
                           for sid in feasible_ids}

        primary_idx = chosen_order[0] if chosen_order else int(feasible_idx[0]) if feasible_idx.size else 0
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)
        recs = [self.sids[i] for i in chosen_order]

        utils = np.array([s.get_utilization() for s in self.sim.spklus.values()])
        tr = MasterDDPGTransition(joint_obs, mask, raw_bids.astype(np.float32),
                                  self.sim.current_step, primary_idx, gini_now=_gini(utils))
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterDDPGReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, mask, action, reward_vec, next_obs, next_mask, future_avail,
             gini_now, done):
        self.buf.append((obs, mask, action, reward_vec, next_obs, next_mask,
                         future_avail, gini_now, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        cols = list(zip(*batch))
        obs, mask, action, reward_vec, next_obs, next_mask, future_avail, gini_now, done = cols
        return (np.stack(obs), np.stack(mask), np.stack(action), np.stack(reward_vec),
                np.stack(next_obs), np.stack(next_mask), np.stack(future_avail),
                np.array(gini_now, dtype=np.float32), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class MasterDDPGInferenceAgent:
    """Evaluasi BERSIH (tanpa noise eksplorasi, tanpa akumulasi transisi/reward) --
    dipasang langsung ke `Simulator`, sama pola `PDQNInferenceAgent`. `k` bid terendah
    menang; EstWait yang ditampilkan tetap forecaster jujur (sama batas
    `MasterDDPGRolloutAgent`)."""

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
        order = feasible_idx[np.argsort(bids[feasible_idx], kind="stable")]
        chosen = [int(i) for i in order[:k_eff]]
        return [self.sids[i] for i in chosen]

    def predict_waits(self, feasible_spklus: dict):
        time_now = self.sim.current_step * self.sim.dt_minutes
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        return self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)


class MasterDDPGTrainer:
    """Melatih `MasterStationActor` + `MasterAttentiveCritic` (K kritik) via DDPG,
    MENAHAN Delayed Access Strategy (transisi baru boleh dipasangkan setelah simulasi
    sungguhan maju >= `delay_minutes`, kritik menerima ketersediaan slot NYATA di t+d)
    dan Dynamic Gradient Re-weighting (bobot softmax-Boltzmann atas gap-ratio,
    IDENTIK rumus `ppo.py::PPOTrainer._compute_beta`, diadaptasi ke sasaran TD DDPG)."""

    def __init__(self, dataset_path, rollout_steps: int = 96, gamma: float = 0.95,
                lr: float = 1e-3, batch_size: int = 64, buffer_capacity: int = 100_000,
                target_update_every: int = 100, tau: float = None,
                noise_start: float = 12.0, noise_end: float = 2.0, noise_decay_frac: float = 0.6,
                reward_calc=None, seed: int = 0, verbose: bool = True,
                updates_per_chunk: int = 20, equity_calc=None,
                delay_minutes: float = 15.0, n_critics: int = N_REWARD_STREAMS,
                beta_mode: str = "gap_ratio", beta_sigma: float = 0.1):
        self.dataset_path = dataset_path
        self.equity_calc = equity_calc
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.tau = tau
        self.updates_per_chunk = int(updates_per_chunk)
        self.noise_start = float(noise_start)
        self.noise_end = float(noise_end)
        self.noise_decay_frac = float(noise_decay_frac)
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # --- Delayed Access Strategy (mekanisme 3) ---
        sim0 = self._fresh_sim()
        self.dt_minutes = sim0.dt_minutes
        self.delay_steps = max(1, round(float(delay_minutes) / self.dt_minutes))
        self._slot_log = _SlotAvailLog(maxlen=self.delay_steps + 4)

        self.N = len(sim0.spklus)
        self.actor = MasterStationActor(STATION_FEAT_DIM_MASTER)
        self.actor_target = MasterStationActor(STATION_FEAT_DIM_MASTER)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        # --- Multi-Critics + Dynamic Gradient Re-weighting (mekanisme 4) ---
        self.n_critics = int(n_critics)
        self.beta_mode = str(beta_mode)
        self.beta_sigma = float(beta_sigma)
        self._ret_best = np.full(self.n_critics, -np.inf, dtype=np.float64)
        self._last_beta = np.full(self.n_critics, 1.0 / self.n_critics, dtype=np.float64)

        self.critic = MasterAttentiveCritic(STATION_FEAT_DIM_MASTER, n_critics=self.n_critics)
        self.critic_target = MasterAttentiveCritic(STATION_FEAT_DIM_MASTER, n_critics=self.n_critics)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = MasterDDPGReplayBuffer(buffer_capacity)

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
                u.trust = old.trust
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
        """Pasangkan (o,a,r,o',future_avail) HANYA bila (a) reward wait sudah diketahui
        (`resolved`, sama semantik lengan lain) DAN (b) `future_avail` di step
        keputusan+delay_steps SUDAH TERCATAT `_slot_log` -- inilah Delayed Access
        Strategy: kritik tak bisa memakai info yg secara kausal belum "terjadi" saat
        keputusan diambil, dan trainer tak boleh memakainya SEBELUM waktunya juga
        (bukan cuma "tersedia di masa depan simulasi", tapi ditunda SUNGGUHAN)."""
        trs = agent.transitions
        n_new = 0
        for i in range(len(trs) - 1):
            t = trs[i]
            if t.pushed or not t.resolved:
                continue
            target_step = t.step + self.delay_steps
            future = self._slot_log.get(target_step)
            if future is None and not boundary:
                continue   # belum waktunya -- INI gerbang delayed-access, bukan sekadar antrean
            if future is None:
                future = self._slot_log.get(current_step) or {}
            nx = trs[i + 1]
            future_vec = np.array([future.get(sid, 0.0) for sid in agent.sids], dtype=np.float32)
            self.buffer.push(t.obs, t.mask, t.action, t.reward_vec(self.n_critics),
                             nx.obs, nx.mask, future_vec, t.gini_now,
                             1.0 if t.done else 0.0)
            t.pushed = True
            n_new += 1
        if boundary and trs:
            last = trs[-1]
            if last.resolved and not last.pushed:
                future = self._slot_log.get(current_step) or {}
                future_vec = np.array([future.get(sid, 0.0) for sid in agent.sids], dtype=np.float32)
                self.buffer.push(last.obs, last.mask, last.action, last.reward_vec(self.n_critics),
                                 last.obs, last.mask, future_vec, last.gini_now, 1.0)
                last.pushed = True
                n_new += 1
        k = 0
        while k < len(trs) and trs[k].pushed:
            k += 1
        agent.transitions = trs[k:]
        return n_new

    def _compute_beta_ddpg(self, target_q):
        """IDENTIK rumus `ppo.py::PPOTrainer._compute_beta` (gap-ratio Boltzmann),
        diadaptasi: `target_q` (B,K) menggantikan `returns` GAE -- proksi Q*_optimal
        (tak tersedia analitis, spt MASTER) = running-max rerata Q per aliran yg
        pernah teramati."""
        K = self.n_critics
        if K == 1:
            return np.ones(1, dtype=np.float64)
        q_mean = target_q.detach().numpy().astype(np.float64).mean(axis=0)   # (K,)
        self._ret_best = np.maximum(self._ret_best, q_mean)
        if self.beta_mode != "gap_ratio":
            return np.full(K, 1.0 / K, dtype=np.float64)
        gap = (self._ret_best - q_mean) / (np.abs(self._ret_best) + 1e-8)
        gap = np.clip(gap, 0.0, 10.0)
        z = gap / max(self.beta_sigma, 1e-8)
        z -= z.max()
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def _ddpg_update(self):
        if len(self.buffer) < self.batch_size:
            return None
        (obs, mask, action, reward_vec, next_obs, next_mask, future_avail,
         gini_now, done) = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        action_t = torch.as_tensor(action, dtype=torch.float32)
        reward_t = torch.as_tensor(reward_vec, dtype=torch.float32)          # (B,K)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        next_mask_t = torch.as_tensor(next_mask, dtype=torch.bool)
        future_t = torch.as_tensor(future_avail, dtype=torch.float32)
        gini_t = torch.as_tensor(gini_now, dtype=torch.float32).unsqueeze(-1)
        done_t = torch.as_tensor(done, dtype=torch.float32).unsqueeze(-1)    # (B,1)

        # --- Kritik (K kepala): y^k = r^k + gamma*(1-done)*Q^k_target(s',a',p') ---
        with torch.no_grad():
            next_action = self.actor_target(next_obs_t)
            next_future = future_t   # future_avail milik TRANSISI SAAT INI (privileged,
                                     # sama d-menit ke depan dari s -- konsisten dgn s' krn
                                     # s' == step berikutnya, bukan +d lagi)
            q_next, _ = self.critic_target(next_obs_t, next_action, next_mask_t, next_future, gini_t)
            target = reward_t + self.gamma * (1.0 - done_t) * q_next        # (B,K)
        q_sa, attn_w = self.critic(obs_t, action_t, mask_t, future_t, gini_t)
        critic_loss = nn.functional.mse_loss(q_sa, target)   # semua kepala, bobot SERAGAM
        self.opt_critic.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_critic.step()

        # --- Aktor: deterministic policy gradient, GABUNGAN K kepala via beta dinamis ---
        beta = self._compute_beta_ddpg(target)
        self._last_beta = beta
        beta_t = torch.as_tensor(beta, dtype=torch.float32)
        actor_action = self.actor(obs_t)
        q_pi, _ = self.critic(obs_t, actor_action, mask_t, future_t, gini_t)   # (B,K)
        actor_loss = -(q_pi @ beta_t).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_actor.step()

        self._n_updates += 1
        if self.tau is not None:
            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                    pt.mul_(1.0 - self.tau).add_(self.tau * p)
                for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                    pt.mul_(1.0 - self.tau).add_(self.tau * p)
        elif self._n_updates % self.target_update_every == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()),
               "q_mean": float(q_sa.detach().mean()), "critic_grad": float(critic_grad_norm),
               "actor_grad": float(actor_grad_norm), "beta": beta.tolist()}

    def _run_one_chunk(self, sim, agent, step: int, chunk: int, do_update: bool):
        boundary = False
        for _ in range(chunk):
            sim.step_once(step, agent=agent)
            self._slot_log.record(step, sim)   # Delayed Access: log SETIAP langkah
            step += 1
            if step >= sim.max_steps:
                boundary = True
                break
        if boundary:
            for t in agent.transitions:
                t.resolved = True
            if agent.transitions:
                agent.transitions[-1].done = True

        pre_push = list(agent.transitions)
        n_new = self._push_ready_pairs(agent, current_step=step, boundary=boundary)
        stats = None
        if do_update:
            for _ in range(min(n_new, self.updates_per_chunk)):
                s = self._ddpg_update()
                if s is not None:
                    stats = s

        info = None
        if n_new:
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            utils = np.array([s.get_utilization() for s in sim.spklus.values()])
            trusts = np.array([u.trust for u in sim.users], dtype=float)
            pushed = [t for t in pre_push if t.pushed]
            acceptance = float(np.mean([t.complied for t in pushed])) if pushed else None
            info = {"n_new": n_new, "n_backlog": len(agent.transitions),
                    "noise_std": agent.noise_std, "buffer_size": len(self.buffer),
                    "gini_served": _gini(served), "gini_util": _gini(utils),
                    "trust_mean": float(trusts.mean()) if trusts.size else 0.5,
                    "acceptance_rate": acceptance, **(stats or {})}

        if boundary:
            sim = self._carry_forward(sim, agent)
            step = 0
        return sim, agent, step, boundary, info

    def train(self, n_updates: int):
        chunk = self.rollout_steps
        self._total_chunks = int(n_updates)
        sim = self._fresh_sim()
        agent = MasterDDPGRolloutAgent(self.actor, sim, self.rc, noise_std=self._noise_std(),
                                       equity_calc=self.equity_calc)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            agent.noise_std = self._noise_std()
            sim, agent, step, boundary, info = self._run_one_chunk(sim, agent, step, chunk, do_update=True)
            self._it_global += 1
            if info is not None:
                self.history.append({"iter": it, **info})
                if self.verbose:
                    print(f"[chunk {it:3d}] new={info['n_new']} backlog={info['n_backlog']} "
                         f"noise={info['noise_std']:.2f} | critic_loss={info.get('critic_loss', 0):.4f} "
                         f"actor_loss={info.get('actor_loss', 0):+.4f} beta={info.get('beta')} | "
                         f"gini_util={info['gini_util']:.3f} trust={info['trust_mean']:.3f} "
                         f"buf={info['buffer_size']}" + (" |PASS-BARU(carry-fwd)" if boundary else ""))
        return self.actor, self.critic
