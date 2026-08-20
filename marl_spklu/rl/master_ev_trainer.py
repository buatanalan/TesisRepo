"""Lengan MASTER perspektif-EV: aktor DESENTRALISASI per-PERMINTAAN (rekomendasi
langsung, kategorikal) + Centralized Multi-Head Attentive Critic yang MENGUMPULKAN
seluruh keputusan pada satu TIMESTEP simulasi sekaligus (lihat `master_ev_policy.py`
utk arsitektur jaringan & diagram data-flow yang disepakati).

BEDA dgn `master_ddpg_trainer.py` (MASTER "asli", stasiun-sbg-agen, DDPG):
    - Backbone PPO on-policy (aksi kategorikal, bukan bid kontinu -- DDPG tak cocok).
    - Kritik dibatch per TIMESTEP (semua EV yg berkeputusan sama saat itu), bukan
      per KEPUTUSAN TUNGGAL (semua stasiun pada satu keputusan).
    - Delayed Access Strategy (mekanisme 3 MASTER) DIPERTAHANKAN identik secara
      prinsip: `future_avail` stasiun yang DIREKOMENDASIKAN hanya boleh dipakai
      kritik setelah simulasi BENAR2 mencapai t+d (`_SlotAvailLog`, diimpor ulang
      dari `master_ddpg_trainer.py` -- bukan duplikasi, mekanisme identik).
    - Multi-Critics + Dynamic Gradient Re-weighting (mekanisme 4) DIPERTAHANKAN:
      `n_critics` kepala Q output (default `N_REWARD_STREAMS`=2), digabung ke
      advantage lewat bobot gap-ratio Boltzmann (identik `ppo.py::_compute_beta`).

CATATAN ENGINEERING (batas desain, baca sebelum mengutip sbg "MAPPO penuh"):
    - GAE/advantage per-EV dihitung dgn nilai LAMA (`_value_pass`, no-grad) SEBELUM
      epoch update -- standar PPO (nilai tak diperbarui di tengah epoch yang sama
      dipakai menghitung advantagenya sendiri). Kritik lalu diperbarui per-epoch via
      forward BERGRADIEN yg SAMA (dikelompokkan ulang per timestep tiap epoch),
      SEMENTARA aktor diperbarui lewat minibatch token-individu (tak perlu
      pengelompokan krn aktor tak melihat EV lain, murni desentralisasi) -- kompromi
      yg diambil krn kritik butuh SELURUH anggota kelompok-timestep hadir bersama utk
      forward-nya (tak bisa diminibatch sembarang spt token independen).
"""
import random

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.rollout import RLRolloutAgent, STREAM_INDIVIDUAL, STREAM_GLOBAL, N_REWARD_STREAMS, _gini
from marl_spklu.rl.master_paper_obs import build_joint_obs_master_ev, STATION_FEAT_DIM_MASTER, STATION_FEAT_DIM_MASTER_EV
from marl_spklu.rl.master_ev_policy import MasterEVActor, MasterEVJointCritic
from marl_spklu.rl.master_ddpg_trainer import _SlotAvailLog   # DAS -- mekanisme SAMA, tak diduplikasi
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.ppo import compute_gae


class MasterEVTransition:
    """Duck-type `Transition` (rollout.py) -- REPLIKA field yg dibaca warisan
    RLRolloutAgent (on_decision/on_step_end/on_charge_complete, TAK diubah di sini,
    sama pola `MasterDDPGTransition`). `value` diisi BELAKANGAN saat update (lewat
    forward kritik terkelompok per-timestep), bukan saat keputusan diambil -- beda
    dgn `Transition` biasa yg mengisi value LANGSUNG (kritik biasa tak butuh
    menunggu anggota kelompok lain)."""

    def __init__(self, obs, mask, primary_idx, logp, ev_state, action_feat, step, gini_now):
        self.obs = obs; self.mask = mask
        self.chosen_indices = [int(primary_idx)]
        self.primary_idx = int(primary_idx)
        self.logp = float(logp)
        self.ev_state = ev_state          # (4,) token kritik: soc,baterai,urgency,jarak
        self.action_feat = action_feat    # (7,) fitur §3.1 stasiun terpilih
        self.step = step
        self.reward_streams = np.zeros(N_REWARD_STREAMS, dtype=np.float64)
        self.done = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.resolved = False; self.pushed = False; self.flock_penalty = 0.0
        self.gini_now = float(gini_now)
        self.future_avail = 0.0   # DAS -- diisi trainer saat digerbangkan siap (_push_ready)
        self.value = None         # diisi trainer saat update (_value_pass)

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


class MasterEVRolloutAgent(RLRolloutAgent):
    """Eksekusi DESENTRALISASI per-permintaan: tiap EV yang muncul menghitung
    rekomendasinya SENDIRI dari obs §3.1+state-EV (`build_joint_obs_master_ev`) via
    `MasterEVActor` (bobot dibagi). Primer disampel STOKASTIK dari distribusi
    kategorikal (eksplorasi PPO via entropi, BUKAN noise OU spt DDPG); sisa slot
    rekomendasi (k-1) diisi deterministik dari peringkat logit berikutnya."""

    def __init__(self, actor, sim, reward_calc, forecaster=None, k: int = 3, equity_calc=None):
        super().__init__(actor, sim, reward_calc, forecaster, k=k, equity_calc=equity_calc)
        self.actor = actor

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        # default_idx/wait_hat: bookkeeping REWARD/CTDE saja (sama batas MasterDDPGRolloutAgent).
        _rich_obs_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)
        joint_obs = build_joint_obs_master_ev(self.sim, self.sids, time_now, user, soc)   # (N,10)
        mask = self._feasible_mask(feasible_ids)
        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            return []   # tak ada kandidat -- tak ada keputusan utk dicatat (beda MasterDDPGRolloutAgent
                        # yg tetap mencatat transisi dummy; di sini softmax atas mask-kosong = NaN)

        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t, mask_t).squeeze(0)     # (N,)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        primary_idx_t = dist.sample()
        primary_idx = int(primary_idx_t.item())
        logp = float(dist.log_prob(primary_idx_t).item())

        logits_np = logits.numpy()
        rest = [int(i) for i in feasible_idx if i != primary_idx]
        rest.sort(key=lambda i: -logits_np[i])
        k_eff = max(1, min(self.k, int(feasible_idx.size)))
        chosen_order = [primary_idx] + rest[:k_eff - 1]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_order}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf") for sid in feasible_ids}
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)
        recs = [self.sids[i] for i in chosen_order]

        soc_norm = float(joint_obs[primary_idx, 8]); battery_norm = float(joint_obs[primary_idx, 9])
        dist_norm = float(joint_obs[primary_idx, 7])
        urgency_norm = (1.0 - soc_norm) * dist_norm   # proksi soc_urgency (Model_Simulasi_Inti §3.1)
        ev_state = np.array([soc_norm, battery_norm, urgency_norm, dist_norm], dtype=np.float32)
        action_feat = joint_obs[primary_idx, :STATION_FEAT_DIM_MASTER].astype(np.float32)

        utils = np.array([s.get_utilization() for s in self.sim.spklus.values()])
        tr = MasterEVTransition(joint_obs, mask, primary_idx, logp, ev_state, action_feat,
                                self.sim.current_step, gini_now=_gini(utils))
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterEVInferenceAgent:
    """Evaluasi BERSIH: argmax logit (bukan sampel) atas kandidat feasible, tanpa
    akumulasi transisi/kritik -- dipasang langsung ke `Simulator`."""

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
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        joint_obs = build_joint_obs_master_ev(self.sim, self.sids, time_now, user, soc)
        mask = np.zeros(self.N, dtype=bool)
        for sid in feasible_spklus:
            if sid in self.sid_to_idx:
                mask[self.sid_to_idx[sid]] = True
        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            return []

        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t, mask_t).squeeze(0).numpy()
        k_eff = max(1, min(self.k, int(feasible_idx.size)))
        order = feasible_idx[np.argsort(-logits[feasible_idx], kind="stable")]
        chosen = [int(i) for i in order[:k_eff]]
        return [self.sids[i] for i in chosen]

    def predict_waits(self, feasible_spklus: dict):
        time_now = self.sim.current_step * self.sim.dt_minutes
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        return self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)


class MasterEVTrainer:
    """Melatih `MasterEVActor` + `MasterEVJointCritic` via PPO on-policy, kritik
    dibatch PER TIMESTEP (lihat docstring modul). Menahan Delayed Access Strategy
    (mekanisme 3) & Multi-Critics+Dynamic Gradient Re-weighting (mekanisme 4);
    TIDAK memakai Bidding Game (mekanisme 2 asli) -- diganti rekomendasi kategorikal
    langsung sesuai permintaan eksplisit user (bukan lelang)."""

    def __init__(self, dataset_path, rollout_steps: int = 96, gamma: float = 0.99,
                lam: float = 0.95, clip: float = 0.2, epochs: int = 4, minibatch: int = 64,
                ent_coef: float = 0.01, lr: float = 3e-4, max_grad_norm: float = 0.5,
                avg_reward: bool = True, r_bar_lr: float = 0.01, max_step_gap: int = 4,
                reward_calc=None, seed: int = 0, verbose: bool = True, equity_calc=None,
                delay_minutes: float = 15.0, k: int = 3, n_critics: int = N_REWARD_STREAMS,
                beta_mode: str = "gap_ratio", beta_sigma: float = 0.1,
                hidden: int = 64, d_model: int = 128, n_heads: int = 4):
        self.dataset_path = dataset_path
        self.equity_calc = equity_calc
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma); self.lam = float(lam); self.clip = float(clip)
        self.epochs = int(epochs); self.minibatch = int(minibatch)
        self.ent_coef = float(ent_coef); self.max_grad_norm = float(max_grad_norm)
        self.avg_reward = bool(avg_reward); self.r_bar_lr = float(r_bar_lr)
        self.max_step_gap = max_step_gap
        self.k = int(k)
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim0 = self._fresh_sim()
        self.dt_minutes = sim0.dt_minutes
        self.delay_steps = max(1, round(float(delay_minutes) / self.dt_minutes))
        self._slot_log = _SlotAvailLog(maxlen=self.rollout_steps + self.delay_steps + 4)
        self.N = len(sim0.spklus)

        self.actor = MasterEVActor(STATION_FEAT_DIM_MASTER_EV, hidden=hidden)
        self.n_critics = int(n_critics)
        self.beta_mode = str(beta_mode); self.beta_sigma = float(beta_sigma)
        self._ret_best = np.full(self.n_critics, -np.inf, dtype=np.float64)
        self._last_beta = np.full(self.n_critics, 1.0 / self.n_critics, dtype=np.float64)
        self.critic = MasterEVJointCritic(d_model=d_model, n_heads=n_heads, n_critics=self.n_critics)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.r_bar = np.zeros(self.n_critics, dtype=np.float64)

        self.history = []
        self._n_updates = 0

    def _fresh_sim(self):
        from marl_spklu.rl.training import _fresh_sim
        return _fresh_sim(self.dataset_path)

    def _carry_forward(self, sim, agent):
        """Identik pola `MasterDDPGTrainer._carry_forward` (trust_alpha/trust_beta,
        BUKAN `trust` -- lihat komentar bug di sana)."""
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

    def _push_ready(self, agent, current_step: int, boundary: bool = False) -> list:
        """Gerbang Delayed Access Strategy: transisi siap dipakai update HANYA bila
        (a) reward wait sudah diketahui (`resolved`) DAN (b) `future_avail` di
        step+delay_steps sudah TERCATAT `_slot_log` (BENAR, bukan diperkirakan) --
        identik prinsip `MasterDDPGTrainer._push_ready_pairs`, hanya mengembalikan
        LIST siap-pakai (bukan pasangan replay-buffer, krn PPO on-policy)."""
        trs = agent.transitions
        ready = []
        for t in trs:
            if t.pushed or not t.resolved:
                continue
            target_step = t.step + self.delay_steps
            future = self._slot_log.get(target_step)
            if future is None and not boundary:
                continue
            if future is None:
                future = self._slot_log.get(current_step) or {}
            sid = agent.sids[t.primary_idx]
            t.future_avail = float(future.get(sid, 0.0))
            t.pushed = True
            ready.append(t)
        if boundary:
            for t in trs:
                if t.resolved and not t.pushed:
                    future = self._slot_log.get(current_step) or {}
                    sid = agent.sids[t.primary_idx]
                    t.future_avail = float(future.get(sid, 0.0))
                    t.pushed = True
                    ready.append(t)
        agent.transitions = [t for t in trs if not t.pushed]
        return ready

    @staticmethod
    def _group_by_step(transitions):
        groups = {}
        for t in transitions:
            groups.setdefault(t.step, []).append(t)
        return groups

    def _critic_forward_group(self, group, requires_grad: bool):
        tokens = torch.as_tensor(
            np.stack([np.concatenate([t.ev_state, t.action_feat]) for t in group]),
            dtype=torch.float32).unsqueeze(0)                                   # (1,Nt,11)
        priv = torch.as_tensor(
            np.stack([[t.future_avail, t.gini_now] for t in group]),
            dtype=torch.float32).unsqueeze(0)                                   # (1,Nt,2)
        pad_mask = torch.zeros(1, len(group), dtype=torch.bool)                  # tanpa padding
        if requires_grad:
            out, _ = self.critic(tokens, pad_mask, priv)
        else:
            with torch.no_grad():
                out, _ = self.critic(tokens, pad_mask, priv)
        return out.squeeze(0)   # (Nt, n_critics)

    def _value_pass(self, transitions):
        """Nilai LAMA (no-grad), dipakai GAE -- standar PPO (advantage dihitung
        SEBELUM epoch update, bukan diperbarui di tengah jalan)."""
        for step, group in self._group_by_step(transitions).items():
            out = self._critic_forward_group(group, requires_grad=False).numpy().astype(np.float64)
            for i, t in enumerate(group):
                t.value = out[i]

    def _critic_loss(self, transitions, returns):
        idx_of = {id(t): i for i, t in enumerate(transitions)}
        total = 0.0
        n = 0
        for step, group in self._group_by_step(transitions).items():
            out = self._critic_forward_group(group, requires_grad=True)              # (Nt,K) grad
            tgt = torch.as_tensor(np.stack([returns[idx_of[id(t)]] for t in group]),
                                  dtype=torch.float32)
            total = total + nn.functional.mse_loss(out, tgt, reduction="sum")
            n += len(group)
        return total / max(1, n)

    def _compute_beta(self, returns):
        """IDENTIK rumus `ppo.py::PPOTrainer._compute_beta` / `master_ddpg_trainer.py
        ::_compute_beta_ddpg` (gap-ratio Boltzmann) -- Dynamic Gradient Re-weighting."""
        K = self.n_critics
        if K == 1:
            return np.ones(1, dtype=np.float64)
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

    def _update(self, transitions):
        transitions = sorted(transitions, key=lambda t: t.step)
        self._value_pass(transitions)
        if self.avg_reward:
            batch_mean_r = np.mean(
                np.stack([t.reward_vec(self.n_critics) for t in transitions]), axis=0)
            self.r_bar += self.r_bar_lr * (batch_mean_r - self.r_bar)
            returns, adv = compute_gae(transitions, self.gamma, self.lam, avg_reward=True,
                                       r_bar=self.r_bar, max_step_gap=self.max_step_gap)
        else:
            returns, adv = compute_gae(transitions, self.gamma, self.lam,
                                       max_step_gap=self.max_step_gap)

        adv = (adv - adv.mean(axis=0, keepdims=True)) / (adv.std(axis=0, keepdims=True) + 1e-8)
        beta = self._compute_beta(returns)
        self._last_beta = beta
        adv_combined = adv @ beta.astype(np.float32)

        obs_b = torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32)
        mask_b = torch.as_tensor(np.stack([t.mask for t in transitions]), dtype=torch.bool)
        primary_b = torch.as_tensor(np.array([t.primary_idx for t in transitions]), dtype=torch.long)
        old_logp = torch.as_tensor(np.array([t.logp for t in transitions]), dtype=torch.float32)
        adv_b = torch.as_tensor(adv_combined, dtype=torch.float32)

        B = len(transitions)
        idx = np.arange(B)
        last = {}
        for _ in range(self.epochs):
            critic_loss = self._critic_loss(transitions, returns)
            self.opt_critic.zero_grad()
            critic_loss.backward()
            critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.opt_critic.step()

            np.random.shuffle(idx)
            for start in range(0, B, self.minibatch):
                mb = idx[start:start + self.minibatch]
                logits = self.actor(obs_b[mb], mask_b[mb])
                logp_all = torch.log_softmax(logits, dim=-1)
                probs_all = torch.softmax(logits, dim=-1)
                logp = logp_all.gather(1, primary_b[mb].unsqueeze(-1)).squeeze(-1)
                ent = -(probs_all * logp_all).sum(-1)
                ratio = torch.exp(logp - old_logp[mb])
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b[mb]
                pi_loss = -torch.min(s1, s2).mean()
                ent_loss = -ent.mean()
                loss = pi_loss + self.ent_coef * ent_loss
                self.opt_actor.zero_grad()
                loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()
                last = {"pi_loss": float(pi_loss.item()), "critic_loss": float(critic_loss.item()),
                       "entropy": float(ent.mean().item()), "actor_grad": float(actor_grad),
                       "critic_grad": float(critic_grad), "beta": beta.tolist()}
        self._n_updates += 1
        return last

    def train(self, n_updates: int):
        chunk = self.rollout_steps
        sim = self._fresh_sim()
        agent = MasterEVRolloutAgent(self.actor, sim, self.rc, k=self.k, equity_calc=self.equity_calc)
        step = 0
        for it in range(n_updates):
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

            ready = self._push_ready(agent, current_step=step, boundary=boundary)
            info = None
            if len(ready) >= 2:
                stats = self._update(ready)
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                utils = np.array([s.get_utilization() for s in sim.spklus.values()])
                trusts = np.array([u.trust for u in sim.users], dtype=float)
                acceptance = float(np.mean([t.complied for t in ready]))
                info = {"n_tr": len(ready), "gini_served": _gini(served), "gini_util": _gini(utils),
                       "trust_mean": float(trusts.mean()) if trusts.size else 0.5,
                       "acceptance_rate": acceptance, **stats}
                self.history.append({"iter": it, **info})
                if self.verbose:
                    print(f"[chunk {it:3d}] n={info['n_tr']} acc={info['acceptance_rate']:.2f} "
                         f"gini_util={info['gini_util']:.3f} trust={info['trust_mean']:.3f} "
                         f"pi_loss={info.get('pi_loss', 0):+.4f} critic_loss={info.get('critic_loss', 0):.4f} "
                         f"ent={info.get('entropy', 0):.3f} beta={info.get('beta')}"
                         + (" |PASS-BARU(carry-fwd)" if boundary else ""), flush=True)

            if boundary:
                sim = self._carry_forward(sim, agent)
                step = 0
        return self.actor, self.critic
