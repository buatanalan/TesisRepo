"""MASTER perspektif-EV, DILATIH via `ppo.py::PPOTrainer` LANGSUNG (tanpa modifikasi)
-- pengganti trainer custom `master_ev_trainer.py::MasterEVTrainer` yang kritiknya
terbukti TAK PERNAH stabil sepanjang 300 chunk (critic_loss tetap orde ribuan-puluhan
ribu hampir sepanjang training, gini_util per-chunk melompat 0,04-0,57 tanpa tren --
lihat riwayat `master_ev_training_results.json`).

SATU PERUBAHAN UTAMA (agar dapat diatribusikan): kritik "kolektif per-timestep"
(`MasterEVJointCritic`, self-attention antar-EV yang berkeputusan sama saat itu)
DIGANTI kritik V(s) TUNGGAL ber-atensi HANYA atas kandidat stasiun EV itu SENDIRI
(`AttentionPooling`, `policy.py`) -- pola PERSIS `MasterStationPPOPolicy`, satu-satunya
lengan keluarga MASTER-PPO yang terbukti stabil (giniSD 0,005-0,010 lintas 3 seed,
`[[tahap2-gae-delayed-reward-bug]]` & diskusi perbandingan MASTER-EV). Trainer custom
(_value_pass/_critic_loss/_group_by_step per-timestep) DIHAPUS SELURUHNYA -- memakai
`ppo.py::PPOTrainer.update()` apa adanya (KL early-stop, minibatch epoch, gap-ratio DGR,
normalisasi advantage per-aliran -- SEMUA mesin yang sudah terbukti stabil di H-PPO/
P-PPO/MASTER-bid/MASTER-stasiun-PPO, TAK diduplikasi/ditulis ulang).

DIPERTAHANKAN dari `master_ev_policy.py` (unit agen = permintaan EV, BUKAN stasiun)
------------------------------------------------------------------------------------
  - Observasi §3.1+state-EV (`build_joint_obs_master_ev`, 10 fitur/stasiun) -- EV native
    di observasi aktor, bukan tempelan.
  - Aksi = rekomendasi kategorikal LANGSUNG (top-K/threshold+epsilon-greedy, IDENTIK
    `HPPOPolicy.act()`/`evaluate()` -- disalin apa adanya, bukan re-implementasi, supaya
    perbedaan hasil tak bisa dijelaskan oleh perbedaan logika seleksi).

DIHILANGKAN scr EKSPLISIT (sama pola & alasan `MasterStationPPOPolicy`)
-------------------------------------------------------------------------
  - Kritik kolektif per-timestep (evaluasi "dikumpulkan per timestep") -- diganti V(s)
    tunggal, krn diagnosis kuat (`critic_loss` liar 300 chunk penuh) menunjuk ke SINI
    sbg biang ketidakstabilan, bukan ke tulang-punggung DDPG (sudah PPO dari awal) atau
    reward (preset `seimbang4x` sudah diuji & DITOLAK di kelas lama).
  - Delayed Access Strategy (privileged future_avail) -- kritik V(s) di sini HANYA
    melihat state SAAT INI, sama seperti H-PPO/P-PPO/MASTER-bid/MASTER-stasiun-PPO.

BILA hasil membaik tajam & stabil, itu ISYARAT KUAT (bukan bukti definitif -- kritik
kolektif-per-timestep tetap gagasan yang beralasan dari sisi teori CTDE, hanya
implementasinya di sini yang terbukti tak stabil) bahwa arsitektur kritiknya-lah yang
perlu didesain ulang lebih hati-hati, BUKAN unit-agen-EV atau bentuk-aksi-kategorikalnya."""
import random
import warnings

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.policy import StationEncoder, AttentionPooling, NEG_INF
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER_EV, build_joint_obs_master_ev
from marl_spklu.rl.rollout import RLRolloutAgent, Transition, RewardCalculatorStub, _gini
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.ppo import PPOTrainer, _make_logger
from marl_spklu.rl.pdqn_policy import (PreferenceAttention, hist_feat_dim,
                                       hist_feat_dim_feature, PREF_STATION_FEAT_DIM)
from marl_spklu.rl.p_ppo_policy import PREF_D_LSTM, PREF_D_ATTN


class MasterEVPPOPolicy(nn.Module):
    """Aktor per-PERMINTAAN (`station_encoder`+`disc_head`, TANPA konteks tambahan --
    state EV SUDAH di-broadcast ke tiap baris kandidat via `build_joint_obs_master_ev`,
    beda dgn `HPPOPolicy` yg butuh blok skalar+LSTM riwayat terpisah) + kritik V(s)
    "buta-aksi" ber-atensi (`AttentionPooling`, sama kelas dipakai `MasterStationPPOPolicy`
    -- bukan implementasi baru, supaya beda hasil tak bisa dijelaskan beda encoder)."""

    def __init__(self, n_spklu: int, hidden: int = 64, critic_hidden: int = 128,
                n_critics: int = 1):
        super().__init__()
        self.n_spklu = int(n_spklu)
        self.n_critics = int(n_critics)
        self.scalar_dim = 0   # state EV sudah masuk tiap baris kandidat, tak perlu blok skalar

        self.station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER_EV, 0, hidden)
        self.disc_head = nn.Linear(hidden, 1)

        self.critic_station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER_EV, 0, critic_hidden)
        self.critic_pool = AttentionPooling(critic_hidden)
        self.critic_head = nn.Sequential(
            nn.Linear(critic_hidden, critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, self.n_critics),
        )

    def _split_station_block(self, flat, feat_dim, scalar_dim):
        """flat: (B, feat_dim*N) FEATURE-MAJOR -- lihat `MasterEVPPORolloutAgent` utk
        konversi dari (N,feat_dim) station-major (sama konvensi `MasterStationPPOPolicy`)."""
        scalars = flat[:, :scalar_dim]
        block = flat[:, scalar_dim:]
        n = block.shape[1] // feat_dim
        station_feats = block.view(-1, feat_dim, n).transpose(1, 2)
        return scalars, station_feats

    def forward(self, obs, hist=None, critic_obs=None):
        """`hist` diabaikan (tak ada modul riwayat di sini). `critic_obs` diabaikan bila
        None -- kritik memakai `obs` YANG SAMA dgn aktor (V(s) sederhana, bukan CTDE
        penuh -- sesuai penyempitan cakupan yg dinyatakan di docstring modul)."""
        _, station_feats = self._split_station_block(obs, STATION_FEAT_DIM_MASTER_EV, 0)
        zero_ctx = torch.zeros(obs.shape[0], 0, device=obs.device, dtype=obs.dtype)

        emb = self.station_encoder(station_feats, zero_ctx)
        logits = self.disc_head(emb).squeeze(-1)             # (B,N)

        c_emb = self.critic_station_encoder(station_feats, zero_ctx)
        pooled = self.critic_pool(c_emb)
        value = self.critic_head(pooled)                     # (B,n_critics)
        return logits, value

    def _fwd(self, obs, hist, critic_obs, pref_hist):
        """Panggil `forward()` polimorfik -- teruskan `pref_hist` HANYA bila subclass
        (`MasterEVPPOPrefPolicy`) benar2 punya modul preferensi. Pola SAMA PERSIS
        `_BiddingMixin._fwd` (master_bidding_policy.py) -- disalin, bukan diwarisi,
        krn kelas ini basisnya nn.Module langsung (bukan _BiddingMixin)."""
        if pref_hist is not None and hasattr(self, "pref_lstm"):
            return self.forward(obs, hist, critic_obs, pref_hist=pref_hist)
        return self.forward(obs, hist, critic_obs)

    # ---------------- act()/evaluate(): IDENTIK `HPPOPolicy` (policy.py), disalin ----
    # apa adanya (bukan re-implementasi) -- top-K/threshold+epsilon-greedy, log-prob
    # Categorical berurutan-tanpa-pengembalian. Lihat `policy.py::HPPOPolicy.act` utk
    # penjelasan lengkap tiap cabang; komentar di sini dipangkas (tak diduplikasi).
    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, k: int = 3, critic_obs_np=None,
           epsilon: float = 0.0, threshold: float = 0.20, pref_hist_np=None, **_abaikan):
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        critic_obs = (torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
                     if critic_obs_np is not None else None)

        pref_hist = (torch.as_tensor(pref_hist_np, dtype=torch.float32).unsqueeze(0)
                    if pref_hist_np is not None else None)
        logits, value = self._fwd(obs, hist, critic_obs, pref_hist)
        if not torch.isfinite(logits).all():
            warnings.warn("MasterEVPPOPolicy.act: logits non-finite (NaN/Inf).", RuntimeWarning)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=NEG_INF)

        feasible_idx = torch.nonzero(mask[0], as_tuple=True)[0]
        n_feasible = int(feasible_idx.numel())
        k_eff = max(1 if n_feasible > 0 else 0, min(int(k), n_feasible))

        import random as _random
        if n_feasible > 0 and _random.random() < epsilon:
            explore_size = _random.randint(1, k_eff)
            perm = feasible_idx[torch.randperm(n_feasible)][:explore_size]
            chosen_order = perm.tolist()
        else:
            masked_logits = logits[0].masked_fill(~mask[0], NEG_INF)
            probs = torch.softmax(masked_logits, dim=-1)
            above = feasible_idx[probs[feasible_idx] > threshold]
            if above.numel() == 0:
                above = feasible_idx[probs[feasible_idx].argmax().unsqueeze(0)]
            if above.numel() > k_eff:
                top = torch.topk(probs[above], k_eff).indices
                above = above[top]
            order = torch.argsort(probs[above], descending=True)
            chosen_order = above[order].tolist()

        remaining_mask = mask[0].clone()
        logp_total = 0.0
        for idx in chosen_order:
            masked_logits_j = logits[0].masked_fill(~remaining_mask, NEG_INF)
            dist_j = torch.distributions.Categorical(logits=masked_logits_j)
            idx_t = torch.as_tensor(idx)
            logp_total = logp_total + dist_j.log_prob(idx_t)
            remaining_mask[idx] = False

        return {
            "chosen_indices": [int(i) for i in chosen_order],
            "n_rec": len(chosen_order),
            "logp": float(logp_total) if isinstance(logp_total, float) else float(logp_total.item()),
            "value": value[0].detach().cpu().numpy().astype("float64"),
        }

    def evaluate(self, obs_b, mask_b, chosen_indices_b, n_rec_b, hist_b, critic_obs_b=None,
                pref_hist_b=None, **_abaikan):
        logits, value = self._fwd(obs_b, hist_b, critic_obs_b, pref_hist_b)
        logits = logits.masked_fill(~mask_b, NEG_INF)

        B, k = chosen_indices_b.shape
        remaining_mask = mask_b.clone()
        logp = torch.zeros(B, device=logits.device)
        entropy = torch.zeros(B, device=logits.device)
        for j in range(k):
            valid_j = j < n_rec_b
            masked_logits = logits.masked_fill(~remaining_mask, NEG_INF)
            dist_disc = torch.distributions.Categorical(logits=masked_logits)
            idx_j = chosen_indices_b[:, j]
            logp_disc_j = dist_disc.log_prob(idx_j)
            ent_disc_j = dist_disc.entropy()

            logp = logp + torch.where(valid_j, logp_disc_j, torch.zeros_like(logp_disc_j))
            entropy = entropy + torch.where(valid_j, ent_disc_j, torch.zeros_like(ent_disc_j))
            remove = torch.zeros_like(remaining_mask)
            remove.scatter_(1, idx_j.unsqueeze(1), True)
            remaining_mask = remaining_mask & ~(remove & valid_j.unsqueeze(1))

        return logp, entropy, value


class MasterEVPPOPrefPolicy(MasterEVPPOPolicy):
    """`MasterEVPPOPolicy` + modul ekstraksi preferensi PDQN (`pref_lstm` +
    `PreferenceAttention` + `pref_gate`) -- pola SAMA PERSIS `MasterStationPPOPrefPolicy`/
    `PPPOPolicy`, disuntikkan sbg konteks tambahan yg di-broadcast ke seluruh kandidat
    stasiun saat encoding.

    BEDA PENTING dgn `MasterStationPPOPrefPolicy`: di sana P dianggap "melanggar" §3.1
    murni krn stasiun SEHARUSNYA buta terhadap pemohon (Pers. 11 MASTER). Di sini TIDAK
    ADA tegangan itu -- unit agen MEMANG permintaan EV itu sendiri, jadi riwayat
    (a_hat,a) MILIK SATU EV YANG SAMA yang sedang membuat keputusan adalah identitas
    yang alami, bukan tempelan. Ini pengujian ULANG hipotesis "P gagal krn identitas-
    ambigu di perspektif stasiun" (5x gagal sebelumnya, SEMUA di arsitektur stasiun-
    sbg-agen atau kritik-per-timestep yg belum stabil) pada kondisi yg sudah dibersihkan
    dari KEDUA confound itu sekaligus."""

    def __init__(self, n_spklu: int, hidden: int = 64, critic_hidden: int = 128,
                n_critics: int = 1, pref_d_lstm: int = PREF_D_LSTM, pref_d_attn: int = PREF_D_ATTN,
                pref_feature_mode: bool = False, use_preference: bool = True):
        super().__init__(n_spklu, hidden=hidden, critic_hidden=critic_hidden, n_critics=n_critics)
        self.pref_feature_mode = bool(pref_feature_mode)
        self.use_preference = bool(use_preference)
        self.pref_d_attn = int(pref_d_attn)
        self.pref_hist_feat_dim = (hist_feat_dim_feature(PREF_STATION_FEAT_DIM)
                                   if self.pref_feature_mode else hist_feat_dim(n_spklu))
        self.pref_lstm = nn.LSTM(self.pref_hist_feat_dim, pref_d_lstm, batch_first=True)
        self.pref_attn = PreferenceAttention(STATION_FEAT_DIM_MASTER_EV, pref_d_lstm, pref_d_attn)
        # Gerbang nol-awal (GTrXL/Parisotto dkk. 2019) -- modul baru diinisialisasi acak,
        # disuntik penuh langsung akan jadi derau besar bagi station_encoder yg sedianya
        # belajar lancar tanpa itu (sama alasan seluruh kelas +P lain di repo ini).
        self.pref_gate = nn.Parameter(torch.tensor(0.0))
        self.station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER_EV, pref_d_attn, hidden)

    def _encode_pref(self, pref_hist):
        if not self.use_preference:
            return torch.zeros(pref_hist.shape[0], self.pref_lstm.hidden_size,
                               device=pref_hist.device, dtype=pref_hist.dtype)
        _, (h_n, _) = self.pref_lstm(pref_hist)
        return h_n[-1]

    def forward(self, obs, hist=None, critic_obs=None, pref_hist=None):
        _, station_feats = self._split_station_block(obs, STATION_FEAT_DIM_MASTER_EV, 0)

        if pref_hist is not None and self.use_preference:
            c_pref = self._encode_pref(pref_hist)
            attended_pref, _ = self.pref_attn(station_feats, c_pref)
            attended_pref = self.pref_gate * attended_pref
        else:
            attended_pref = torch.zeros(obs.shape[0], self.pref_d_attn, device=obs.device)

        emb = self.station_encoder(station_feats, attended_pref)
        logits = self.disc_head(emb).squeeze(-1)

        c_emb = self.critic_station_encoder(station_feats,
                                            torch.zeros(obs.shape[0], 0, device=obs.device))
        pooled = self.critic_pool(c_emb)
        value = self.critic_head(pooled)
        return logits, value


class MasterEVPPORolloutAgent(RLRolloutAgent):
    """Override HANYA `get_recommendation` -- observasi §3.1+state-EV via
    `build_joint_obs_master_ev`. `on_decision`/`on_step_end`/`on_charge_complete`
    DIWARISI APA ADANYA (resep reward 2-aliran SAMA PERSIS lengan PPO lain)."""

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        _rich_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)

        joint_obs = build_joint_obs_master_ev(self.sim, self.sids, time_now, user, soc)   # (N,10)
        obs_flat = joint_obs.T.reshape(-1).astype(np.float32)   # feature-major, konvensi _split_station_block
        mask = self._feasible_mask(feasible_ids)
        dummy_hist = np.zeros((1, 1), dtype=np.float32)

        # Modul preferensi (opsional): _use_pref/_build_pref_hist DIWARISI dari
        # RLRolloutAgent.__init__ -- pola sama `MasterStationPPORolloutAgent`.
        if self._use_pref:
            pref_hist = self._build_pref_hist(user)
            act = self.policy.act(obs_flat, mask, dummy_hist, k=self.k, critic_obs_np=obs_flat,
                                  epsilon=self.epsilon, threshold=self.threshold,
                                  pref_hist_np=pref_hist)
        else:
            pref_hist = None
            act = self.policy.act(obs_flat, mask, dummy_hist, k=self.k, critic_obs_np=obs_flat,
                                  epsilon=self.epsilon, threshold=self.threshold)

        chosen_indices = act["chosen_indices"]
        n_rec = act["n_rec"]
        recs = [self.sids[i] for i in chosen_indices]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_indices}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf")
                           for sid in feasible_ids}

        primary_idx = chosen_indices[0] if chosen_indices else int(np.nonzero(mask)[0][0])
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)

        idx_arr = np.zeros(self.k, dtype=np.int64)
        idx_arr[:n_rec] = chosen_indices

        tr = Transition(obs_flat, obs_flat, dummy_hist, mask, idx_arr, n_rec,
                        act["logp"], act["value"], self.sim.current_step,
                        pref_hist=pref_hist)
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterEVPPOInferenceAgent:
    """Evaluasi bersih -- pola sama `MasterStationPPOInferenceAgent`. `pref_feature_mode`
    DIAMBIL OTOMATIS dari `policy.pref_feature_mode` (bila ada) -- latih & uji tak pernah
    bisa berbeda mode encoding riwayat (kelas bug berulang, lihat docstring kelas itu)."""

    def __init__(self, policy, sim, forecaster=None, k: int = 3, epsilon: float = 0.0,
                threshold: float = 0.20):
        pref_feature_mode = bool(getattr(policy, "pref_feature_mode", False))
        self._roll = MasterEVPPORolloutAgent(policy, sim, RewardCalculatorStub(), forecaster, k=k,
                                             pref_feature_mode=pref_feature_mode)
        self._roll.epsilon = epsilon
        self._roll.threshold = threshold

    def get_recommendation(self, feasible_spklus):
        return self._roll.get_recommendation(feasible_spklus)

    def predict_waits(self, feasible_spklus):
        return self._roll.predict_waits(feasible_spklus)

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        self._roll.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        self._roll.transitions.clear()

    def on_charge_complete(self, user):
        self._roll.on_charge_complete(user)
        self._roll.transitions.clear()


class MasterEVPPOTrainer:
    """Trainer continuing-task PPO/GAE/clip -- pola SAMA PERSIS `MasterStationPPOTrainer`
    (satu-satunya penyimpangan dari `TorchContinuingTrainer` generik: carry-forward trust
    lintas pass, dipertahankan sengaja agar sepadan dgn seluruh keluarga MASTER lain)."""

    def __init__(self, dataset_path, rollout_steps: int = 96, seed: int = 0,
                verbose: bool = True, reward_calc=None, hidden: int = 64,
                critic_hidden: int = 128, k: int = 3, n_critics: int = 1,
                equity_calc=None, policy_cls=MasterEVPPOPolicy, policy_kw=None, **ppo_kw):
        self.dataset_path = dataset_path
        self.rollout_steps = int(rollout_steps)
        self.k = int(k)
        self.equity_calc = equity_calc
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim0 = self._fresh_sim()
        self.N = len(sim0.spklus)
        self.policy = policy_cls(self.N, hidden=hidden, critic_hidden=critic_hidden,
                                 n_critics=n_critics, **(policy_kw or {}))
        self.ppo = PPOTrainer(self.policy, avg_reward=True, **ppo_kw)
        self.history = []
        self._it_global = 0
        self._logger = _make_logger(verbose)

    def _fresh_sim(self):
        from marl_spklu.rl.training import _fresh_sim
        return _fresh_sim(self.dataset_path)

    def _reset_rc(self):
        if hasattr(self.rc, "reset_episode_state"):
            self.rc.reset_episode_state()

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
        return new_sim

    def train(self, forecaster, n_updates: int):
        chunk = self.rollout_steps
        sim = self._fresh_sim()
        self._reset_rc()
        pref_feature_mode = bool(getattr(self.policy, "pref_feature_mode", False))
        agent = MasterEVPPORolloutAgent(self.policy, sim, self.rc, forecaster, k=self.k,
                                        equity_calc=self.equity_calc,
                                        pref_feature_mode=pref_feature_mode)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            boundary = False
            for _ in range(chunk):
                sim.step_once(step, agent=agent)
                step += 1
                if step >= sim.max_steps:
                    boundary = True
                    break
            resolved = [t for t in agent.transitions if t.resolved]
            pending = [t for t in agent.transitions if not t.resolved]
            self._it_global += 1
            if resolved:
                resolved[-1].done = boundary
                stats = self.ppo.update(resolved)
                rewards = np.array([t.reward for t in resolved])
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                trusts = np.array([u.trust for u in sim.users], float)
                info = {"mean_reward": float(rewards.mean()),
                       "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                       "gini_served": _gini(served), "n_tr": len(resolved),
                       "n_pending": len(pending), "trust_mean": float(trusts.mean())}
                rec = {"iter": it, **info, **stats}
                self.history.append(rec)
                if self.verbose:
                    self._logger.info(
                        "[chunk %3d] R=%+.4f accept=%.2f | KL=%.4f EV=%.2f | gini=%.3f "
                        "trust=%.3f n=%d pend=%d%s", it, info["mean_reward"],
                        info["acceptance_rate"], stats.get("approx_kl", 0),
                        stats.get("explained_var", 0), info["gini_served"],
                        info["trust_mean"], info["n_tr"], info["n_pending"],
                        " |PASS-BARU(carry-fwd)" if boundary else "")
            if boundary:
                sim = self._carry_forward(sim, agent)
                self._reset_rc()
                step = 0
            else:
                agent.transitions = pending
        return self.policy
