"""Trainer Master-PPO (2026-08-28) -- kelas BARU, tak mewarisi `MasterDDPGRolloutAgent`/
`MasterPureTrainer`/`PPOTrainer` generik (interface `hist_b`/`chosen_b`/`n_rec_b` generik
tak cocok tanpa rekayasa tambahan yg rawan-salah -- lih. diskusi). REUSE `compute_gae`
dari `ppo.py` (sudah teruji, ada perbaikan time-distance-gating C4) sbg satu-satunya
komponen dipakai bersama.

SATU mode (bukan 3 spt DDPG) -- konsekuensi V(s): gap-ratio DGR genuine-spesialis
(Pers.13) tak terdefinisi utk V(s) (lih. `master_pure_ppo_policy.py`), diganti gap-ratio
berbasis return (pola `ppo.py::_compute_beta`), TAK butuh pra-latih spesialis terpisah.

Delayed Access Strategy (§3.2.2) DIPERTAHANKAN identik versi DDPG (`_SlotRawLog`,
`snapshot_slots_raw`, penghapusan pengaruh q_t sendiri) -- direuse LANGSUNG dari
`master_pure_trainer.py` (bukan properti tulang-punggung latih)."""
import random

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.rollout import RLRolloutAgent, STREAM_INDIVIDUAL, STREAM_GLOBAL, N_REWARD_STREAMS, _gini
from marl_spklu.rl.master_paper_obs import build_joint_obs_master, STATION_FEAT_DIM_MASTER
from marl_spklu.rl.master_pure_ppo_policy import MasterPurePPOActor, MasterPurePPOCritic
from marl_spklu.rl.master_pure_trainer import snapshot_slots_raw, _SlotRawLog
from marl_spklu.rl.ppo import compute_gae
from marl_spklu.rl.rewards import RewardCalculator


class MasterPurePPOTransition:
    """`value`=(K,) DIISI saat rollout (act(), BUKAN belakangan spt DDPG -- PPO butuh
    V(s) SAAT KEPUTUSAN utk GAE). `logp`=log-prob bid tersampel (stasiun feasible saja,
    pola `_BiddingMixin`)."""

    def __init__(self, obs, mask, bids, logp, value, step, primary_idx):
        self.obs = obs; self.mask = mask; self.bids = bids
        self.logp = float(logp); self.value = value
        self.step = step
        self.chosen_indices = [int(primary_idx)]
        self.reward_streams = np.zeros(N_REWARD_STREAMS, dtype=np.float64)
        self.done = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.resolved = False; self.pushed = False; self.flock_penalty = 0.0
        self.I_raw = None

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


class MasterPurePPORolloutAgent(RLRolloutAgent):
    """Observasi §3.1 MURNI (7 fitur). Bid disampel dari Normal(mean,std) TIAP langkah
    (eksplorasi PPO bawaan -- TANPA noise OU eksternal spt DDPG). Pemenang = bid
    TERTINGGI (argmax, §3.1) -- sama arah versi DDPG."""

    def __init__(self, actor, critic, sim, reward_calc, forecaster=None, k: int = 3,
                equity_calc=None):
        super().__init__(actor, sim, reward_calc, forecaster, k=k, equity_calc=equity_calc)
        self.actor = actor
        self.critic = critic

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        _rich_obs_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)
        joint_obs = build_joint_obs_master(self.sim, self.sids, time_now)
        mask = self._feasible_mask(feasible_ids)

        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            bid_mean = self.actor(obs_t)
            dist = self.actor.dist(bid_mean)
            bids_t = dist.sample()
            logp = (dist.log_prob(bids_t) * mask_t.float()).sum().item()
            # I_raw BELUM tersedia saat keputusan (privileged, Delayed Access) -- V(s)
            # dihitung dgn I_raw=0 SEMENTARA saat act(), lih. catatan `_push_ready_pairs`
            # trainer utk kenapa ini aman (value dipakai HANYA sbg baseline GAE, bukan
            # bagian gradien aktor langsung -- beda dari DDPG yg butuh Q eksplisit).
            zero_I = torch.zeros_like(mask_t, dtype=torch.float32)
            value, _ = self.critic(obs_t, mask_t, zero_I)
        bids = bids_t.squeeze(0).numpy().astype(np.float32)

        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            chosen_order = []
        else:
            k_eff = max(1, min(self.k, int(feasible_idx.size)))
            order = feasible_idx[np.argsort(-bids[feasible_idx], kind="stable")]
            chosen_order = [int(i) for i in order[:k_eff]]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_order}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf")
                           for sid in feasible_ids}

        primary_idx = chosen_order[0] if chosen_order else int(feasible_idx[0]) if feasible_idx.size else 0
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)
        recs = [self.sids[i] for i in chosen_order]

        tr = MasterPurePPOTransition(joint_obs, mask, bids, logp,
                                     value.squeeze(0).numpy().astype(np.float64),
                                     self.sim.current_step, primary_idx)
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterPurePPOInferenceAgent:
    """Evaluasi bersih (mean bid, TANPA sampling) -- pola sama `MasterPureInferenceAgent`."""

    def __init__(self, actor, forecaster=None, k: int = 3):
        self.actor = actor
        self.actor.eval()
        from marl_spklu.rl.forecaster import FormulaForecaster
        self.forecaster = forecaster or FormulaForecaster()
        self.k = int(k)
        self.sids = None; self.sid_to_idx = None; self.N = None

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
            bid_mean = self.actor(obs_t).squeeze(0).numpy()   # mean = keputusan deterministik
        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            return []
        k_eff = max(1, min(self.k, int(feasible_idx.size)))
        order = feasible_idx[np.argsort(-bid_mean[feasible_idx], kind="stable")]
        return [self.sids[int(i)] for i in order[:k_eff]]

    def predict_waits(self, feasible_spklus: dict):
        time_now = self.sim.current_step * self.sim.dt_minutes
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        return self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)


class MasterPurePPOTrainer:
    """SATU mode (bukan pretrain_specialist/dgr spt DDPG -- lih. docstring modul).
    K=2 (wait,gini) sejak awal, gap-ratio berbasis return (running-max, pola
    `ppo.py::_compute_beta`)."""

    def __init__(self, dataset_path, rollout_steps: int = 96, gamma: float = 0.99,
                lam: float = 0.95, lr: float = 5e-4, clip: float = 0.2, epochs: int = 10,
                minibatch: int = 32, ent_coef: float = 0.01, vf_coef: float = 0.5,
                max_grad_norm: float = 0.5, target_kl: float = 0.03,
                delay_minutes: float = 30.0, hidden: int = 64,
                beta_mode: str = "gap_ratio", beta_sigma: float = 0.2,
                reward_calc=None, seed: int = 0, verbose: bool = True,
                equity_calc=None, max_step_gap: int = 4):
        self.dataset_path = dataset_path
        self.equity_calc = equity_calc
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma); self.lam = float(lam); self.clip = float(clip)
        self.epochs = int(epochs); self.minibatch = int(minibatch)
        self.ent_coef = float(ent_coef); self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm); self.target_kl = target_kl
        self.max_step_gap = max_step_gap
        self.n_critics = N_REWARD_STREAMS   # 2: wait(individual)+gini(global), tetap
        self.beta_mode = str(beta_mode); self.beta_sigma = float(beta_sigma)
        self._ret_best = np.full(self.n_critics, -np.inf, dtype=np.float64)
        self._last_beta = np.full(self.n_critics, 1.0 / self.n_critics, dtype=np.float64)
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim0 = self._fresh_sim()
        self.dt_minutes = sim0.dt_minutes
        self.delay_steps = max(1, round(float(delay_minutes) / self.dt_minutes))
        self._slot_log = _SlotRawLog(maxlen=self.rollout_steps + self.delay_steps + 4)
        self.N = len(sim0.spklus)

        self.actor = MasterPurePPOActor(STATION_FEAT_DIM_MASTER, hidden=hidden)
        self.critic = MasterPurePPOCritic(STATION_FEAT_DIM_MASTER, hidden=hidden,
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
        """Sama persis `MasterPureTrainer._push_ready_pairs` (Delayed Access + hapus
        pengaruh q_t sendiri) -- BEDA: hasil dikembalikan sbg LIST transisi (utk PPO
        `update()` langsung), bukan didorong ke replay buffer."""
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
        """Gap-ratio berbasis RETURN (running-max proksi R*_optimal) -- pola IDENTIK
        `ppo.py::PPOTrainer._compute_beta`. Satu-satunya adaptasi DGR yg koheren utk
        V(s) (lih. docstring modul: Pers.13 asli butuh Q bersyarat-aksi, tak berlaku)."""
        K = self.n_critics
        ret_mean = np.asarray(returns, dtype=np.float64).mean(axis=0)
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
        bids_b = torch.as_tensor(np.stack([t.bids for t in transitions]), dtype=torch.float32)
        I_b = torch.as_tensor(np.stack([t.I_raw for t in transitions]), dtype=torch.float32)
        old_logp = torch.as_tensor(np.array([t.logp for t in transitions]), dtype=torch.float32)
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
                bid_mean = self.actor(obs_b[mb])
                dist = self.actor.dist(bid_mean)
                m = mask_b[mb].float()
                logp = (dist.log_prob(bids_b[mb]) * m).sum(dim=-1)
                ent = (dist.entropy() * m).sum(dim=-1)
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
                        "grad_norm": grad_norm, "beta": beta.tolist()}
            if self.target_kl is not None:
                with torch.no_grad():
                    bid_mean = self.actor(obs_b)
                    dist = self.actor.dist(bid_mean)
                    m = mask_b.float()
                    nlp = (dist.log_prob(bids_b) * m).sum(dim=-1)
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
        agent = MasterPurePPORolloutAgent(self.actor, self.critic, sim, self.rc,
                                          equity_calc=self.equity_calc)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            sim, agent, step, boundary, info = self._run_one_chunk(sim, agent, step, chunk)
            self._it_global += 1
            if info is not None:
                self.history.append({"iter": it, **info})
                if self.verbose:
                    print(f"[ppo chunk {it:3d}] ready={info['n_ready']} backlog={info['n_backlog']} "
                         f"pi_loss={info.get('pi_loss', 0):+.4f} v_loss={info.get('v_loss', 0):.4f} "
                         f"beta={info.get('beta')} | gini_util={info['gini_util']:.3f}"
                         + (" |PASS-BARU" if boundary else ""))
        return self.actor, self.critic
