"""PDQN KONTINU -- versi DQN SUNGGUHAN (bukan PPO). Arsitektur DDPG (Deep Deterministic
Policy Gradient): varian DQN klasik untuk ruang aksi kontinu -- actor+critic, TARGET
NETWORK, replay buffer, TD bootstrap -- PERSIS mekanisme PDQN diskrit (dqn_trainer.py),
bukan clipped-surrogate/GAE ala PPO (pdqn_continuous_policy.py, DIPERTAHANKAN terpisah,
tidak dihapus/diganti).

Kenapa DDPG, bukan Q-network diskrit biasa: aksi di sini KONTINU (janji EstWait per
SPKLU feasible sekaligus, delta menit) -- max_a Q(s,a) tak bisa dihitung dgn enumerasi
aksi spt DQN diskrit (ruang aksi tak berhingga). DDPG mengganti max_a Q(s,a) dgn Q(s,
actor(s)) -- actor DILATIH mengarah ke aksi yg MEMAKSIMALKAN Q kritik (deterministic
policy gradient), kritik DILATIH via TD bootstrap spt Q-learning biasa:
    y          = r + gamma*(1-done)*Q_target(s', actor_target(s'))   (kritik, TD)
    L_critic   = (Q(s,a) - y)^2                                       (MSE, spt PDQN diskrit)
    L_actor    = -mean(Q(s, actor(s)))                                (deterministic PG)
Target network (actor & kritik) disinkron PERIODIK (hard sync tiap N langkah gradien,
KONSISTEN dgn konvensi PDQN diskrit -- bukan soft/Polyak update spt DDPG asli).

Modul preferensi PDQN (LSTM atas pasangan (a_hat,a) + attention) DIPERTAHANKAN identik
dgn pdqn_policy.py -- a_hat di sini didefinisikan sbg SPKLU dgn janji EstWait TERENDAH
dari actor (proksi "rekomendasi utama" karena aksi aslinya vektor, bukan skalar).

Trust DINAMIS (tidak dibekukan) -- performatif spt PDQNContinuousPolicy (PPO), tapi
mekanisme belajarnya sekarang benar-benar off-policy Q-learning, bukan on-policy PPO.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.pdqn_policy import PreferenceAttention, ResidualBlock, hist_feat_dim, hist_feat_dim_feature
from marl_spklu.rl.policy import STATION_FEAT_DIM
from marl_spklu.rl.rollout import RLRolloutAgent, _gini
from marl_spklu.rl.rewards import RewardCalculator

D_LSTM = 64
D_ATTN = 64
FC_WIDTH = 128
N_RESIDUAL_BLOCKS = 3


class _PDQNBackbone(nn.Module):
    """Tulang punggung BERSAMA actor & kritik: modul preferensi PDQN (LSTM+attention)
    + FC 6-lapis+residual, identik desain PDQNQNetwork (pdqn_policy.py) tapi dipisah
    jadi modul dipakai ulang oleh actor DAN kritik (beda kepala output saja)."""

    def __init__(self, obs_dim: int, n_spklu: int, extra_in_dim: int = 0,
                d_lstm: int = D_LSTM, d_attn: int = D_ATTN, width: int = FC_WIDTH,
                n_blocks: int = N_RESIDUAL_BLOCKS, pref_feature_mode: bool = False):
        super().__init__()
        self.n_spklu = int(n_spklu)
        self.scalar_dim = obs_dim - STATION_FEAT_DIM * n_spklu
        assert self.scalar_dim >= 0

        # pref_feature_mode=True: pasangan (a_hat,a) sbg VEKTOR FITUR stasiun, bukan
        # one-hot identitas -- lihat pdqn_policy.py::hist_feat_dim_feature.
        self.pref_feature_mode = bool(pref_feature_mode)
        self.hist_feat_dim = (hist_feat_dim_feature() if pref_feature_mode
                              else hist_feat_dim(n_spklu))
        self.pref_lstm = nn.LSTM(self.hist_feat_dim, d_lstm, batch_first=True)
        self.attn = PreferenceAttention(STATION_FEAT_DIM, d_lstm, d_attn)

        in_dim = obs_dim + d_lstm + d_attn + extra_in_dim
        self.input_proj = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU())
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(n_blocks)])
        self.width = width

    def _split_station_block(self, obs):
        scalars = obs[:, :self.scalar_dim]
        block = obs[:, self.scalar_dim:]
        n = block.shape[1] // STATION_FEAT_DIM
        station_feats = block.view(-1, STATION_FEAT_DIM, n).transpose(1, 2)
        return scalars, station_feats

    def forward(self, obs, hist, extra=None):
        """obs:(B,obs_dim) hist:(B,K,2N) extra:(B,extra_in_dim)|None -> h:(B,width)."""
        _, (h_n, _) = self.pref_lstm(hist)
        c_t = h_n[-1]
        _, station_feats = self._split_station_block(obs)
        attended, _ = self.attn(station_feats, c_t)
        x = torch.cat([obs, c_t, attended] + ([extra] if extra is not None else []), dim=-1)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return h


class PDQNContinuousActor(nn.Module):
    """pi(s) -> a in [-delta_max,delta_max]^N (janji EstWait delta per SPKLU feasible)."""

    def __init__(self, obs_dim: int, n_spklu: int, delta_max: float = 10.0,
                pref_feature_mode: bool = False):
        super().__init__()
        self.n_spklu = int(n_spklu)
        self.delta_max = float(delta_max)
        self.backbone = _PDQNBackbone(obs_dim, n_spklu, pref_feature_mode=pref_feature_mode)
        self.head = nn.Linear(self.backbone.width, n_spklu)

    def forward(self, obs, hist):
        h = self.backbone(obs, hist)
        return torch.tanh(self.head(h)) * self.delta_max   # (B,N)

    @property
    def pref_lstm(self):
        """Alias supaya RLRolloutAgent._use_pref (hasattr check) mendeteksi modul
        preferensi -- LSTM sesungguhnya ada di self.backbone.pref_lstm."""
        return self.backbone.pref_lstm


class PDQNContinuousCritic(nn.Module):
    """Q(s,a) -> skalar, a = vektor aksi PENUH (N,) dari actor (bukan per-stasiun)."""

    def __init__(self, obs_dim: int, n_spklu: int, pref_feature_mode: bool = False):
        super().__init__()
        self.n_spklu = int(n_spklu)
        self.backbone = _PDQNBackbone(obs_dim, n_spklu, extra_in_dim=n_spklu,
                                      pref_feature_mode=pref_feature_mode)
        self.head = nn.Linear(self.backbone.width, 1)

    def forward(self, obs, hist, action):
        h = self.backbone(obs, hist, extra=action)
        return self.head(h).squeeze(-1)   # (B,)


class DDPGTransition:
    """Transisi DQN-kontinu utk SATU keputusan. Atribut `chosen_indices`/`resolved`/
    `flock_penalty`/dst SENGAJA meniru rollout.Transition supaya method WARISAN
    RLRolloutAgent (on_decision/on_step_end/on_charge_complete -- akumulasi reward
    identik jalur H-PPO/PPO) bisa dipakai TANPA modifikasi (duck typing, bukan
    pewarisan class) -- hanya get_recommendation yg diganti mekanisme aksinya."""

    def __init__(self, obs, pref_hist, action, mask, step, primary_idx):
        self.obs = obs; self.pref_hist = pref_hist; self.action = action; self.mask = mask
        self.chosen_indices = [int(primary_idx)]   # dibaca on_step_end (flocking) & on_decision
        self.step = step
        self.reward = 0.0; self.done = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.resolved = False; self.pushed = False; self.flock_penalty = 0.0
        # Diisi trainer saat dipasangkan (s,a,r,s') menurut urutan keputusan (event-driven,
        # sama spt DQNContinuingTrainer._push_ready_pairs) -- next_* None sampai penerusnya ada.
        self.next_obs = None; self.next_pref_hist = None


class PDQNDDPGRolloutAgent(RLRolloutAgent):
    """Reuse SELURUH mesin observasi/reward RLRolloutAgent (obs kaya-fitur, critic_obs,
    pref_hist, akumulasi reward prox+wait+honesty+global) -- HANYA get_recommendation
    diganti: actor DDPG (kontinu, + noise eksplorasi) menggantikan policy.act() PPO.
    Trust DINAMIS (estwait, sama spt PDQNContinuousPolicy) -- BUKAN mode binary_utility
    PDQN diskrit."""

    def __init__(self, actor, sim, reward_calc, forecaster=None, noise_std: float = 2.0,
                honest_estwait: bool = False, k: int = 3, equity_calc=None,
                pref_feature_mode: bool = False):
        # k=3 (default): rekomendasikan hanya TOP-k stasiun (janji EstWait terendah dari
        # aktor), BUKAN k=n_feasible ("kontinu murni" versi awal) -- itu membuat kepatuhan
        # (`recs`=SEMUA feasible) trivial 1,0 (dikonfirmasi §diskusi: satu-satunya di antara
        # 9 kombinasi metode/rezim yg diuji). Sekarang setara PPO (PDQNContinuousTrainer
        # jg dipanggil k=3) -- perbandingan kepatuhan jadi bermakna & adil.
        super().__init__(actor, sim, reward_calc, forecaster, k=k,
                         honest_estwait=honest_estwait, equity_calc=equity_calc,
                         pref_feature_mode=pref_feature_mode)
        self.actor = actor
        self.noise_std = float(noise_std)   # menit, skala noise eksplorasi (spt epsilon)
        # Noise Ornstein-Uhlenbeck (standar DDPG asli, BUKAN Gaussian independen per-
        # keputusan) -- terkorelasi temporal: theta menarik balik ke 0, sigma dorongan
        # acak. Menghasilkan eksplorasi yg "menjelajah" (jalan beberapa langkah ke arah
        # yg sama) alih-alih melompat acak tiap keputusan -- lebih sesuai utk aksi
        # kontinu multi-dimensi (paper DDPG asli, Lillicrap dkk. 2016).
        self.use_ou_noise = True
        self.ou_theta = 0.15
        self._ou_state = np.zeros(self.N, dtype=np.float32)

    def _sample_noise(self):
        if not self.use_ou_noise:
            return np.random.normal(0.0, self.noise_std, size=self.N)
        self._ou_state += (self.ou_theta * (0.0 - self._ou_state)
                           + self.noise_std * np.random.normal(size=self.N))
        return self._ou_state.copy()

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        obs, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)
        mask = self._feasible_mask(feasible_ids)
        pref_hist = self._build_pref_hist(user)

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        pref_hist_t = torch.as_tensor(pref_hist, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t, pref_hist_t).squeeze(0).numpy()
        if self.noise_std > 0:
            action = action + self._sample_noise()
            action = np.clip(action, -self.actor.delta_max, self.actor.delta_max)

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        feasible_idx = [self.sid_to_idx[s] for s in feasible_ids if s in self.sid_to_idx]
        all_disps = {}
        for i in feasible_idx:
            sid = self.sids[i]
            base_w = float(baseline.get(sid, 0.0))
            all_disps[sid] = base_w if self.honest_estwait else max(0.0, base_w + float(action[i]))

        # TOP-k (janji terendah = paling menarik) yg BENAR-BENAR direkomendasikan --
        # sisanya dapat proksi "2x janji tertinggi di antara top-k" (SAMA pola
        # RLRolloutAgent/PPO) supaya P_rec (user.py) punya insentif jelas memilih di
        # antara yg direkomendasikan, dan `recs` TIDAK trivial = semua feasible.
        k_eff = min(self.k, len(all_disps)) if all_disps else 0
        topk_sids = sorted(all_disps, key=all_disps.get)[:k_eff]
        rec_disps = {sid: all_disps[sid] for sid in topk_sids}
        max_rec_disp = max(rec_disps.values()) if rec_disps else 0.0
        estimated_waits = {sid: rec_disps.get(sid, 2.0 * max_rec_disp) for sid in feasible_ids}

        primary_sid = topk_sids[0] if topk_sids else feasible_ids[0]
        primary_idx = self.sid_to_idx[primary_sid]
        primary_disp = rec_disps.get(primary_sid, 0.0)
        recs = list(topk_sids)   # HANYA top-k -- kepatuhan sekarang bermakna (bukan trivial 1,0)

        tr = DDPGTransition(obs, pref_hist, action.astype(np.float32), mask,
                            self.sim.current_step, primary_idx)
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(primary_sid, 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class DDPGReplayBuffer:
    """(s,a,r,s',done) off-policy -- identik pola ReplayBuffer PDQN diskrit (dqn_trainer.py),
    aksi di sini VEKTOR kontinu (N,) bukan indeks diskrit."""

    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, pref_hist, action, reward, next_obs, next_pref_hist, done):
        self.buf.append((obs, pref_hist, action, reward, next_obs, next_pref_hist, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        obs, pref_hist, action, reward, next_obs, next_pref_hist, done = zip(*batch)
        return (np.stack(obs), np.stack(pref_hist), np.stack(action),
                np.array(reward, dtype=np.float32), np.stack(next_obs),
                np.stack(next_pref_hist), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class PDQNContinuousDDPGTrainer:
    """Melatih PDQNContinuousActor/Critic via DDPG -- sim PERSISTEN di-step per potongan
    (pola sama dgn PDQNContinuousTrainer/DQNContinuingTrainer), trust DINAMIS + CARRY-
    FORWARD lintas batas horizon (state fisik direset, trust/riwayat TIDAK -- lihat
    pdqn_continuous_trainer.py utk motivasi lengkap pola ini)."""

    def __init__(self, dataset_path, rollout_steps: int = 96, gamma: float = 0.95,
                lr: float = 1e-3, batch_size: int = 64, buffer_capacity: int = 100_000,
                target_update_every: int = 100, tau: float = None, noise_start: float = 4.0,
                noise_end: float = 0.5, noise_decay_frac: float = 0.6,
                reward_calc=None, delta_max: float = 10.0, seed: int = 0,
                verbose: bool = True, updates_per_chunk: int = 20,
                honest_estwait: bool = False, equity_calc=None,
                pref_feature_mode: bool = False):
        self.dataset_path = dataset_path
        self.honest_estwait = bool(honest_estwait)
        self.equity_calc = equity_calc
        self.pref_feature_mode = bool(pref_feature_mode)
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        # tau (bukan None) -> SOFT update Polyak tiap langkah gradien (standar DDPG asli,
        # theta_target = tau*theta + (1-tau)*theta_target), MENGGANTIKAN hard-sync periodik
        # -- katanya lebih stabil krn target bergerak halus, bukan melompat tiap N langkah.
        self.tau = tau
        # Cap langkah gradien/chunk -- TANPA ini, jumlah update = n_new (bisa ratusan/chunk
        # pada dataset padat), terlalu mahal. 20 dipilih sbg titik-tengah (masih signifikan
        # dibanding target_update_every=100) -- TIDAK dikalibrasi/disapu, catat sbg keterbatasan.
        self.updates_per_chunk = int(updates_per_chunk)
        self.noise_start = float(noise_start)
        self.noise_end = float(noise_end)
        self.noise_decay_frac = float(noise_decay_frac)
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim = self._fresh_sim()
        self.N = len(sim.spklus)
        self.obs_dim = 6 + 5 * self.N   # sama dgn RLRolloutAgent (STATION_FEAT_DIM=5)

        self.actor = PDQNContinuousActor(self.obs_dim, self.N, delta_max=delta_max,
                                         pref_feature_mode=self.pref_feature_mode)
        self.actor_target = PDQNContinuousActor(self.obs_dim, self.N, delta_max=delta_max,
                                                pref_feature_mode=self.pref_feature_mode)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = PDQNContinuousCritic(self.obs_dim, self.N, pref_feature_mode=self.pref_feature_mode)
        self.critic_target = PDQNContinuousCritic(self.obs_dim, self.N, pref_feature_mode=self.pref_feature_mode)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = DDPGReplayBuffer(buffer_capacity)

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
        return new_sim

    def _push_ready_pairs(self, agent, boundary: bool = False) -> int:
        """Pasangkan (s,a,r,s') menurut URUTAN KEPUTUSAN (event-driven, sama semantik
        DQNContinuingTrainer._push_ready_pairs) -- transisi i baru bisa dimasukkan ke
        buffer setelah i.resolved DAN transisi i+1 sudah ADA (jadi next_obs tersedia)."""
        trs = agent.transitions
        n_new = 0
        for i in range(len(trs) - 1):
            t = trs[i]
            if t.resolved and not t.pushed:
                nx = trs[i + 1]
                self.buffer.push(t.obs, t.pref_hist, t.action, t.reward,
                                 nx.obs, nx.pref_hist, 1.0 if t.done else 0.0)
                t.pushed = True
                n_new += 1
        if boundary and trs:
            last = trs[-1]
            if last.resolved and not last.pushed:
                self.buffer.push(last.obs, last.pref_hist, last.action, last.reward,
                                 last.obs, last.pref_hist, 1.0)
                last.pushed = True
                n_new += 1
        k = 0
        while k < len(trs) and trs[k].pushed:
            k += 1
        agent.transitions = trs[k:]
        return n_new

    def _ddpg_update(self):
        if len(self.buffer) < self.batch_size:
            return None
        obs, pref_hist, action, reward, next_obs, next_pref_hist, done = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        pref_hist_t = torch.as_tensor(pref_hist, dtype=torch.float32)
        action_t = torch.as_tensor(action, dtype=torch.float32)
        reward_t = torch.as_tensor(reward, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        next_pref_hist_t = torch.as_tensor(next_pref_hist, dtype=torch.float32)
        done_t = torch.as_tensor(done, dtype=torch.float32)

        # --- Kritik: TD bootstrap, y = r + gamma*(1-done)*Q_target(s', actor_target(s')) ---
        with torch.no_grad():
            next_action = self.actor_target(next_obs_t, next_pref_hist_t)
            q_next = self.critic_target(next_obs_t, next_pref_hist_t, next_action)
            target = reward_t + self.gamma * (1.0 - done_t) * q_next
        q_sa = self.critic(obs_t, pref_hist_t, action_t)
        critic_loss = nn.functional.mse_loss(q_sa, target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_critic.step()

        # --- Aktor: deterministic policy gradient, maksimalkan Q(s, actor(s)) ---
        actor_action = self.actor(obs_t, pref_hist_t)
        actor_loss = -self.critic(obs_t, pref_hist_t, actor_action).mean()
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
               "actor_grad": float(actor_grad_norm)}

    def _run_one_chunk(self, sim, agent, step: int, chunk: int, do_update: bool):
        """do_update=False -> fase FREEZE RRM: transisi tetap terkumpul/di-push ke
        buffer (trust/reward berjalan wajar) TAPI TIDAK ADA `_ddpg_update` -- bobot
        actor/critic TIDAK berubah. Return (sim, agent, step, boundary, info|None)."""
        boundary = False
        for _ in range(chunk):
            sim.step_once(step, agent=agent)
            step += 1
            if step >= sim.max_steps:
                boundary = True
                break
        if boundary:
            for t in agent.transitions:
                t.resolved = True
            if agent.transitions:
                agent.transitions[-1].done = True

        pre_push_transitions = list(agent.transitions)   # snapshot -- kepatuhan dihitung sebelum disunting
        n_new = self._push_ready_pairs(agent, boundary=boundary)
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
            pushed_this_chunk = [t for t in pre_push_transitions if t.pushed]
            acceptance = (float(np.mean([t.complied for t in pushed_this_chunk]))
                         if pushed_this_chunk else None)
            disp_waits = [t.disp_estwait for t in pushed_this_chunk]
            info = {"n_new": n_new, "n_backlog": len(agent.transitions),
                    "noise_std": agent.noise_std, "buffer_size": len(self.buffer),
                    "gini_served": _gini(served), "gini_util": _gini(utils),
                    "trust_mean": float(trusts.mean()) if trusts.size else 0.5,
                    "trust_std": float(trusts.std()) if trusts.size else 0.0,
                    "acceptance_rate": acceptance,
                    "mean_disp_estwait": float(np.mean(disp_waits)) if disp_waits else None,
                    **(stats or {})}

        if boundary:
            sim = self._carry_forward(sim, agent)
            step = 0
        return sim, agent, step, boundary, info

    def train(self, n_updates: int):
        chunk = self.rollout_steps
        self._total_chunks = int(n_updates)
        sim = self._fresh_sim()
        agent = PDQNDDPGRolloutAgent(self.actor, sim, self.rc, noise_std=self._noise_std(),
                                    honest_estwait=self.honest_estwait, equity_calc=self.equity_calc,
                                    pref_feature_mode=self.pref_feature_mode)
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
                         f"actor_loss={info.get('actor_loss', 0):+.4f} q={info.get('q_mean', 0):+.3f} | "
                         f"gini_util={info['gini_util']:.3f} trust={info['trust_mean']:.3f} "
                         f"buf={info['buffer_size']}" + (" |PASS-BARU(carry-fwd)" if boundary else ""))
        return self.actor, self.critic

    def train_rrm(self, n_rounds: int = 5, freeze_chunks: int = 15, retrain_chunks: int = 10,
                 flush_buffer_each_round: bool = False, retrain_updates_per_chunk: int = None):
        """RRM (Repeated Risk Minimization -- lebih presisi disebut RPO/Repeated Policy
        Optimization utk konteks RL, lihat diskusi) versi DDPG: fase FREEZE (bobot beku,
        ukur D(pi_t) setelah menyetimbang) -> fase RETRAIN (`_ddpg_update` aktif lagi).

        `flush_buffer_each_round=True` (PERBAIKAN #1): kosongkan replay buffer di AWAL
        tiap ronde -- retrain HANYA belajar dari transisi ronde INI (freeze+retrain),
        tak pernah tercampur data ronde lampau yg distribusinya (trust) sudah beda.
        Menyelaraskan DDPG dgn premis RRM/RPO ("retrain dari D(pi_t) bersih") yg secara
        struktural dilanggar replay buffer biasa (lihat diskusi kelayakan).

        `retrain_updates_per_chunk` (PERBAIKAN #2): anggaran langkah gradien KHUSUS fase
        retrain, TERPISAH dari `self.updates_per_chunk` (dipakai `train()` biasa) --
        default None -> pakai `self.updates_per_chunk`. Naikkan (mis. 100) utk menguji
        apakah DDPG cuma kurang anggaran, bukan cacat struktural.

        Return (actor, critic, rrm_trace)."""
        chunk = self.rollout_steps
        self._total_chunks = n_rounds * (freeze_chunks + retrain_chunks)
        retrain_upc = retrain_updates_per_chunk if retrain_updates_per_chunk is not None else self.updates_per_chunk
        sim = self._fresh_sim()
        agent = PDQNDDPGRolloutAgent(self.actor, sim, self.rc, noise_std=self._noise_std(),
                                    honest_estwait=self.honest_estwait, equity_calc=self.equity_calc,
                                    pref_feature_mode=self.pref_feature_mode)
        step = 0
        rrm_trace = []
        for r in range(n_rounds):
            if flush_buffer_each_round:
                self.buffer.buf.clear()
            freeze_last = None
            for _ in range(freeze_chunks):
                agent.noise_std = self.noise_end   # noise minimal saat freeze (ukur D(pi_t) bersih)
                sim, agent, step, boundary, info = self._run_one_chunk(sim, agent, step, chunk, do_update=False)
                if info is not None:
                    freeze_last = info
            retrain_last = None
            _saved_upc = self.updates_per_chunk
            self.updates_per_chunk = retrain_upc
            for _ in range(retrain_chunks):
                self._it_global += 1
                agent.noise_std = self._noise_std()
                sim, agent, step, boundary, info = self._run_one_chunk(sim, agent, step, chunk, do_update=True)
                if info is not None:
                    retrain_last = info
            self.updates_per_chunk = _saved_upc
            rec = {"round": r,
                  "freeze_gini_util": freeze_last["gini_util"] if freeze_last else None,
                  "freeze_gini_served": freeze_last["gini_served"] if freeze_last else None,
                  "freeze_trust": freeze_last["trust_mean"] if freeze_last else None,
                  "freeze_acceptance": freeze_last["acceptance_rate"] if freeze_last else None,
                  "retrain_gini_util": retrain_last["gini_util"] if retrain_last else None,
                  "retrain_gini_served": retrain_last["gini_served"] if retrain_last else None,
                  "retrain_trust": retrain_last["trust_mean"] if retrain_last else None,
                  "retrain_acceptance": retrain_last["acceptance_rate"] if retrain_last else None}
            rrm_trace.append(rec)
            if self.verbose:
                print(f"[RPO ronde {r:2d}] FREEZE: gini_util={rec['freeze_gini_util'] or -1:.4f} "
                     f"trust={rec['freeze_trust'] or -1:.3f}  ->  RETRAIN: "
                     f"gini_util={rec['retrain_gini_util'] or -1:.4f} trust={rec['retrain_trust'] or -1:.3f}")
        return self.actor, self.critic, rrm_trace
