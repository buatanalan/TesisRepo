"""MASTER stasiun-sebagai-agen + bidding, DILATIH PPO/GAE/clip ON-POLICY -- mengisolasi
apakah TULANG PUNGGUNG DDPG (bukan mekanisme koordinasinya) yang menyebabkan kegagalan
MASTER-DDPG (`master_ddpg_policy.py`/`master_ddpg_trainer.py`).

DITAHAN KONSTAN dari MASTER-DDPG (satu faktor berubah, agar hasil dapat diatribusikan)
------------------------------------------------------------------------------------
  - stasiun sbg agen desentralisasi, bobot dibagi lintas stasiun (Gupta dkk. 2017)
  - observasi §3.1 MURNI (`master_paper_obs.py` -- TANPA SoC/lokasi/baterai pemohon)
  - aksi = bid kontinu per-stasiun, k TERENDAH menang (`_BiddingMixin`, dipakai ulang
    apa adanya dari `master_bidding_policy.py`)
  - reward: 2 aliran (Prox+wait individual, Gini+antiherding global) via `RLRolloutAgent`
    hook bawaan (`on_decision`/`on_step_end`/`on_charge_complete`) -- TIDAK diduplikasi

DIGANTI (INI YANG DIUJI)
-------------------------
  - tulang punggung: DDPG off-policy + replay -> PPO on-policy + GAE + clip (`ppo.py`,
    TANPA modifikasi -- kelas ini hanya perlu memenuhi kontrak act()/evaluate() yang
    sudah dipakai `PPOTrainer`)
  - kritik: Q(o,a,p) atas AKSI GABUNGAN -> V(s) TANPA aksi (`AttentionPooling` dari
    `policy.py`, sama seperti SELURUH lengan PPO lain yang berhasil)

DIHILANGKAN scr EKSPLISIT (di luar cakupan ablasi ini)
--------------------------------------------------------
  - Delayed Access Strategy (privileged future_avail) -- kritik V(s) di sini HANYA
    melihat state SAAT INI, sama seperti H-PPO/P-PPO/MASTER-bid
  - Dynamic Gradient Re-weighting -- n_critics=1 (baku), sama seperti SELURUH lengan
    lain yang benar-benar dijalankan (audit 2026-08-19 #8: n_critics=2 tak pernah
    beroperasi di lengan mana pun KECUALI MASTER-DDPG)

Kedua penghilangan itu BUKAN kelalaian -- keduanya mekanisme KHUSUS-Q/DDPG tanpa padanan
bersih di V(s)/PPO tanpa re-arsitektur besar. Ablasi ini sengaja SEMPIT: satu faktor
berubah (tulang punggung), agar hasilnya dapat diatribusikan. Bila hasilnya membaik
tajam, itu ISYARAT KUAT (bukan bukti definitif -- 2 dari 4 mekanisme MASTER lain masih
tak beroperasi, sama seperti sebelumnya) bahwa DDPG-lah biang kegagalan MASTER-DDPG,
bukan koordinasi/bidding/stasiun-sebagai-agen itu sendiri (yang terbukti BEKERJA baik
di `MasterBiddingPolicy`, substrat PPO, agen PERMINTAAN -- lihat modul itu).

CARRY-FORWARD TRUST LINTAS PASS -- MENYIMPANG dari konvensi keluarga PPO lain
-------------------------------------------------------------------------------
Seluruh lengan `TorchContinuingTrainer` lain (H-PPO, P-PPO, MASTER, MASTER-bid, dst)
TIDAK membawa trust lintas batas horizon -- `sim` di-reset total tiap pass, trust
kembali ke 0,5 (lihat `ppo.py::TorchContinuingTrainer.train`, baris boundary). Trainer
di sini SENGAJA membawa `trust_alpha`/`trust_beta` lintas pass (spt `master_ddpg_
trainer.py::_carry_forward`), supaya perbandingan terhadap MASTER-DDPG tetap satu
faktor berubah. Ini berarti kelas ini TIDAK sepenuhnya sepadan dengan H-PPO/P-PPO/
MASTER-bid pada sumbu carry-forward -- laporkan sbg batas, jangan disamakan diam-diam.
"""
import random

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.policy import StationEncoder, AttentionPooling
from marl_spklu.rl.master_bidding_policy import _BiddingMixin
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER, build_joint_obs_master
from marl_spklu.rl.rollout import RLRolloutAgent, Transition, RewardCalculatorStub, _gini
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.ppo import PPOTrainer, _make_logger
from marl_spklu.rl.pdqn_policy import PreferenceAttention, hist_feat_dim
from marl_spklu.rl.p_ppo_policy import PREF_D_LSTM, PREF_D_ATTN


class MasterStationPPOPolicy(_BiddingMixin, nn.Module):
    """Aktor stasiun `b^i(o^i_t)` (TANPA konteks -- `scalar_dim=0`, persis Pers. 11 MASTER:
    o^i_t tak memuat fitur pemohon) + kritik V(s) invarian-permutasi atas HANYA fitur
    stasiun §3.1 (tanpa aksi, tanpa observasi privileged -- beda dgn `MasterAttentiveCritic`
    yg mengondisikan pada aksi gabungan; di sini kritik "buta-aksi" spt HPPOPolicy.V(s)).

    `station_encoder`/`disc_head`/`critic_station_encoder`/`critic_pool`/`critic_head`
    memakai KELAS YANG SAMA dari `policy.py` (StationEncoder, AttentionPooling) --
    bukan implementasi baru -- supaya perbedaan hasil tak dapat dijelaskan oleh
    perbedaan encoder, hanya oleh (tulang-punggung, observasi, bentuk-aksi)."""

    def __init__(self, n_spklu: int, hidden: int = 64, critic_hidden: int = 128,
                n_critics: int = 1, bid_log_std_init: float = 0.0):
        super().__init__(bid_log_std_init=bid_log_std_init)   # -> _BiddingMixin -> nn.Module
        self.n_spklu = int(n_spklu)
        self.n_critics = int(n_critics)
        self.scalar_dim = 0          # TANPA konteks skalar -- desentralisasi SESUNGGUHNYA
        self.station_hidden = hidden

        self.station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER, 0, hidden)
        self.disc_head = nn.Linear(hidden, 1)   # dipakai ulang sbg kepala rerata BID

        self.critic_station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER, 0, critic_hidden)
        self.critic_pool = AttentionPooling(critic_hidden)
        self.critic_head = nn.Sequential(
            nn.Linear(critic_hidden, critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, self.n_critics),
        )

    def _split_station_block(self, flat, feat_dim, scalar_dim):
        """Disalin dari `HPPOPolicy._split_station_block` (kelas ini TIDAK mewarisi
        HPPOPolicy -- basisnya nn.Module langsung, tak ada hist/scalar_dim sisa dari
        konvensi request-sebagai-agen). `flat`: (B, feat_dim*N) FEATURE-MAJOR -- lihat
        `MasterStationPPORolloutAgent` untuk konversi dari (N,feat_dim) station-major."""
        scalars = flat[:, :scalar_dim]
        block = flat[:, scalar_dim:]
        n = block.shape[1] // feat_dim
        station_feats = block.view(-1, feat_dim, n).transpose(1, 2)
        return scalars, station_feats

    def forward(self, obs, hist=None, critic_obs=None):
        """`hist`/`critic_obs` DIABAIKAN -- tak ada riwayat (stasiun tak mengingat
        pemohon), kritik memakai obs YANG SAMA dgn aktor (tak ada info privileged,
        konsisten desentralisasi §3.1 murni -- beda dgn HPPOPolicy yg kritiknya CTDE
        penuh melihat trust asli/oracle wait)."""
        _, station_feats = self._split_station_block(obs, STATION_FEAT_DIM_MASTER, 0)
        zero_ctx = torch.zeros(obs.shape[0], 0, device=obs.device, dtype=obs.dtype)

        emb = self.station_encoder(station_feats, zero_ctx)
        bid_mean = self.disc_head(emb).squeeze(-1)          # (B,N) -- ditafsir _BiddingMixin

        c_emb = self.critic_station_encoder(station_feats, zero_ctx)
        pooled = self.critic_pool(c_emb)
        value = self.critic_head(pooled)
        return bid_mean, value


class MasterStationPPOPrefPolicy(MasterStationPPOPolicy):
    """`MasterStationPPOPolicy` + modul ekstraksi preferensi PDQN (`pref_lstm` +
    `PreferenceAttention` + `pref_gate`), sbg KONTEKS TAMBAHAN yang di-broadcast ke
    seluruh stasiun saat encoding -- pola SAMA PERSIS `PPPOPolicy` (p_ppo_policy.py).

    ⚠️ TEGANGAN ARSITEKTURAL yang harus dinyatakan terbuka. Observasi §3.1 MASTER
    sengaja TIDAK memuat fitur pemohon -- `a^i_t = b^i(o^i_t)` (Pers. 11): bid stasiun
    TIDAK bergantung pada siapa yang bertanya. Menambahkan modul preferensi berarti bid
    kini bergantung pada riwayat (a_hat,a) PEMOHON YANG SEDANG MEMINTA -- ini melanggar
    "stasiun buta terhadap pemohon" itu sendiri. Kelas ini TIDAK LAGI dapat diklaim
    "setia observasi §3.1 murni MASTER" -- ia adalah hibrida sengaja: stasiun-sebagai-
    agen (peran) + kesadaran-preferensi-pemohon (informasi tambahan), diuji utk menjawab
    apakah preferensi (bukan sekadar informasi permintaan apa pun) yang menaikkan
    penerimaan -- lihat `RENCANA_...` /diskusi 2026-08-20 (hipotesis P pada MASTER-bid).

    `_record_pref`/`_build_pref_hist` (rollout.py, DIWARISI dari `RLRolloutAgent`)
    beroperasi murni atas INDEKS STASIUN (rekomendasi vs pilihan) + `user_id` -- tak
    bergantung apa pun pada observasi kaya/murni yang dilihat kebijakan, sehingga dapat
    dipakai ulang APA ADANYA di sini tanpa modifikasi.
    """

    def __init__(self, n_spklu: int, hidden: int = 64, critic_hidden: int = 128,
                n_critics: int = 1, bid_log_std_init: float = 0.0,
                pref_d_lstm: int = PREF_D_LSTM, pref_d_attn: int = PREF_D_ATTN,
                pref_feature_mode: bool = False, use_preference: bool = True):
        super().__init__(n_spklu, hidden=hidden, critic_hidden=critic_hidden,
                         n_critics=n_critics, bid_log_std_init=bid_log_std_init)
        self.pref_feature_mode = bool(pref_feature_mode)
        self.use_preference = bool(use_preference)
        self.pref_d_attn = int(pref_d_attn)
        self.pref_hist_feat_dim = hist_feat_dim(n_spklu)   # feature-mode tak didukung di sini
        self.pref_lstm = nn.LSTM(self.pref_hist_feat_dim, pref_d_lstm, batch_first=True)
        self.pref_attn = PreferenceAttention(STATION_FEAT_DIM_MASTER, pref_d_lstm, pref_d_attn)
        # Gerbang nol-awal (Parisotto dkk. 2019/GTrXL, sama alasan dgn PPPOPolicy) --
        # modul preferensi baru diinisialisasi acak, disuntik penuh langsung akan jadi
        # derau besar bagi station_encoder yg sedianya belajar lancar tanpa itu.
        self.pref_gate = nn.Parameter(torch.tensor(0.0))
        # GANTI station_encoder: context_dim bertambah pref_d_attn (sebelumnya 0).
        self.station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER, pref_d_attn, hidden)

    def _encode_pref(self, pref_hist):
        if not self.use_preference:
            return torch.zeros(pref_hist.shape[0], self.pref_lstm.hidden_size,
                               device=pref_hist.device, dtype=pref_hist.dtype)
        _, (h_n, _) = self.pref_lstm(pref_hist)
        return h_n[-1]

    def forward(self, obs, hist=None, critic_obs=None, pref_hist=None):
        _, station_feats = self._split_station_block(obs, STATION_FEAT_DIM_MASTER, 0)

        if pref_hist is not None and self.use_preference:
            c_pref = self._encode_pref(pref_hist)
            attended_pref, _ = self.pref_attn(station_feats, c_pref)
            attended_pref = self.pref_gate * attended_pref
        else:
            attended_pref = torch.zeros(obs.shape[0], self.pref_d_attn, device=obs.device)

        emb = self.station_encoder(station_feats, attended_pref)
        bid_mean = self.disc_head(emb).squeeze(-1)

        c_emb = self.critic_station_encoder(station_feats,
                                            torch.zeros(obs.shape[0], 0, device=obs.device))
        pooled = self.critic_pool(c_emb)
        value = self.critic_head(pooled)
        return bid_mean, value


class MasterStationPPORolloutAgent(RLRolloutAgent):
    """Override HANYA `get_recommendation` -- observasi §3.1 murni via
    `build_joint_obs_master`. `on_decision`/`on_step_end`/`on_charge_complete` DIWARISI
    APA ADANYA dari `RLRolloutAgent`: resep reward (2 aliran, anti-herding jendela
    bergulir, backfill realized_gap) SAMA PERSIS dgn seluruh lengan PPO lain -- supaya
    perbandingan terhadap MASTER-DDPG hanya berbeda pada (obs, aktor, kritik, backbone),
    BUKAN pada resep reward.
    """

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        # `_build_obs` dipanggil HANYA utk default_idx/wait_hat (bookkeeping trust/reward
        # bawaan RLRolloutAgent) -- observasi kaya-fitur yg dihasilkannya TIDAK diberikan
        # ke kebijakan (lihat obs_flat di bawah).
        _rich_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)

        joint_obs = build_joint_obs_master(self.sim, self.sids, time_now)   # (N,7) station-major
        # -> FEATURE-MAJOR (7*N,), konvensi _split_station_block (feat_dim blok
        # berurutan, tiap blok panjang N) -- lihat komentar di kelas policy.
        obs_flat = joint_obs.T.reshape(-1).astype(np.float32)
        mask = self._feasible_mask(feasible_ids)
        dummy_hist = np.zeros((1, 1), dtype=np.float32)   # tak dipakai forward(), hanya
                                                           # supaya kontrak act() terpenuhi

        # Modul preferensi (opsional): _use_pref/_build_pref_hist DIWARISI dari
        # RLRolloutAgent.__init__, murni berbasis indeks stasiun + user_id -- reuse
        # tanpa modifikasi (lihat docstring MasterStationPPOPrefPolicy).
        if self._use_pref:
            pref_hist = self._build_pref_hist(user)
            act = self.policy.act(obs_flat, mask, dummy_hist, k=self.k,
                                  critic_obs_np=obs_flat, epsilon=self.epsilon,
                                  threshold=self.threshold, pref_hist_np=pref_hist)
        else:
            pref_hist = None
            act = self.policy.act(obs_flat, mask, dummy_hist, k=self.k,
                                  critic_obs_np=obs_flat, epsilon=self.epsilon,
                                  threshold=self.threshold)

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
        tr.bids = act.get("bids")
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterStationPPOInferenceAgent:
    """Evaluasi bersih (tanpa akumulasi reward) -- pola SAMA dgn `InferenceAgent`
    (rollout.py), tetapi membungkus `MasterStationPPORolloutAgent` (bukan basis
    `RLRolloutAgent`) supaya observasi §3.1 murni ikut dipakai saat evaluasi juga."""

    def __init__(self, policy, sim, forecaster=None, k: int = 3, epsilon: float = 0.0,
                threshold: float = 0.20):
        self._roll = MasterStationPPORolloutAgent(
            policy, sim, RewardCalculatorStub(), forecaster, k=k)
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


class MasterStationPPOTrainer:
    """Trainer continuing-task PPO/GAE/clip -- pola SAMA persis
    `ppo.py::TorchContinuingTrainer.train()`, ditulis terpisah krn konstruktor
    `TorchPolicyTrainer` terikat erat pada `obs_dim`/`critic_obs_dim` konvensi
    request-sebagai-agen (`policy.py::STATION_FEAT_DIM`, 6 skalar) yang tak cocok
    utk kebijakan stasiun-desentralisasi §3.1 (0 skalar, 7 fitur/stasiun).

    Satu-satunya penyimpangan dari `TorchContinuingTrainer`: carry-forward trust lintas
    pass (lihat docstring modul) -- dipertahankan sengaja supaya sepadan dgn
    MasterDDPGTrainer, BUKAN dgn keluarga PPO lain.

    `policy_cls` (baku `MasterStationPPOPolicy`): ganti ke `MasterStationPPOPrefPolicy`
    utk menguji apakah modul preferensi membantu pada arsitektur stasiun-sebagai-agen
    §3.1 murni. `policy_kw` diteruskan apa adanya ke konstruktor kelas itu.
    """

    def __init__(self, dataset_path, rollout_steps: int = 96, seed: int = 0,
                verbose: bool = True, reward_calc=None, hidden: int = 64,
                critic_hidden: int = 128, k: int = 3, n_critics: int = 1,
                equity_calc=None, policy_cls=MasterStationPPOPolicy,
                policy_kw=None, **ppo_kw):
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
        """PERBAIKAN yg sama spt `master_ddpg_trainer.py` (2026-08-20): `trust` adalah
        @property HANYA-BACA -- salin via trust_alpha/trust_beta, bukan `u.trust = ...`."""
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
        agent = MasterStationPPORolloutAgent(self.policy, sim, self.rc, forecaster,
                                             k=self.k, equity_calc=self.equity_calc)
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
