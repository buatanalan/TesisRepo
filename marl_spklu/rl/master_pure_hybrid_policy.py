"""Master-Hybrid (2026-08-29) -- eksperimen baru user: modul P + keluaran aktor VEKTOR
(bukan skalar bid langsung) + penggabungan P SETELAH vektor terbentuk (opsi 'b', pola
LateCtx) + station attention antar-stasiun DI SISI AKTOR + kepala akhir per-algoritma:
    DDPG -> proyeksi ke bid SKALAR kontinu (mekanisme asli MASTER, argmax menang)
    PPO  -> proyeksi ke logit -> softmax (mekanisme Kandidat A/HPPOPolicy)
Keputusan disepakati 2026-08-29: DDPG TETAP bid kontinu (bukan dipaksa softmax via
Gumbel-Softmax) -- kepala akhir beda per algoritma sesuai kodratnya, tulang punggung
(P+attention+vektor) IDENTIK persis di kedua varian utk perbandingan adil.

Ukuran modul tambahan SENGAJA KECIL (vec_dim/pref_d_lstm/pref_d_attn/station_attn_dim
=8, hidden vektor=16) sesuai instruksi user.

CATATAN KESETIAAN: menambah modul P (kondisi pada RIWAYAT pemohon) MELANGGAR Pers. 11
MASTER (a^i_t=b^i(o^i_t), stasiun seharusnya buta thd pemohon) -- deviasi yg SAMA
kategorinya dgn `MasterBiddingPrefPolicy`/`h6b_utama` (sudah didokumentasikan &
diterima di kedua tempat itu dgn alasan serupa), BUKAN kealpaan baru. Observasi
stasiun `o^i_t` sendiri TETAP 7 fitur §3.1 murni -- P masuk lewat kanal TERPISAH
(late-merge), bukan mencemari `o^i_t` itu sendiri."""
import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER
from marl_spklu.rl.pdqn_policy import PreferenceAttention, hist_feat_dim, hist_feat_dim_feature


class StationVectorHead(nn.Module):
    """Tahap 1: fitur_stasiun(7) -> VEKTOR kecil per-stasiun (bukan skalar bid
    langsung spt `MasterPureActor` lama). Bobot dibagi semua stasiun."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, vec_dim: int = 8,
                hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(station_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, vec_dim),
        )

    def forward(self, station_obs):
        return self.net(station_obs)   # (B,N,vec_dim)


class LatePrefMerge(nn.Module):
    """Opsi 'b' (2026-08-29): vektor per-stasiun SUDAH terbentuk dari fitur mentahnya
    SENDIRI (`StationVectorHead`), P baru digabung SETELAHNYA lewat gerbang nol-awal
    -- pola SAMA `ctx_merge` (`MasterEVPPOPrefPolicySmallLateCtx`), terbukti
    mempersempit zona konflik gradien preferensi-vs-representasi-stasiun sesi ini."""

    def __init__(self, vec_dim: int, pref_dim: int):
        super().__init__()
        self.merge = nn.Linear(vec_dim + pref_dim, vec_dim)
        self.gate_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, vec, pref_ctx):
        n = vec.shape[1]
        pref_exp = pref_ctx.unsqueeze(1).expand(-1, n, -1)
        merged = torch.relu(self.merge(torch.cat([vec, pref_exp], dim=-1)))
        gate = torch.sigmoid(self.gate_raw)
        return vec + gate * (merged - vec)


class SmallStationAttention(nn.Module):
    """`StationSelfAttention` versi kecil (residual + gerbang nol-awal, pola sudah
    terverifikasi sesi ini) -- dim SENGAJA kecil (baku 8) sesuai instruksi user."""

    def __init__(self, vec_dim: int, d_attn: int = None):
        super().__init__()
        self.d_attn = int(d_attn or vec_dim)
        self.q = nn.Linear(vec_dim, self.d_attn)
        self.k = nn.Linear(vec_dim, self.d_attn)
        self.v = nn.Linear(vec_dim, self.d_attn)
        self.out = nn.Linear(self.d_attn, vec_dim)
        self.gate_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, vec, mask):
        q, k, v = self.q(vec), self.k(vec), self.v(vec)
        scores = torch.einsum("bnd,bmd->bnm", q, k) / (self.d_attn ** 0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        attended = self.out(torch.einsum("bnm,bmd->bnd", weights, v))
        gate = torch.sigmoid(self.gate_raw)
        return vec + gate * attended


class _PrefStationBackbone(nn.Module):
    """Tulang punggung BERSAMA (KOMPOSISI, bukan pewarisan) dipakai KEDUA varian
    (DDPG/PPO) -- station_feat -> vektor -> +P (late, opsi b) -> station attention
    -> vektor akhir (B,N,vec_dim), siap diproyeksikan kepala masing2 algoritma."""

    def __init__(self, n_spklu: int, station_feat_dim: int = STATION_FEAT_DIM_MASTER,
                vec_dim: int = 8, bid_hidden: int = 16, pref_d_lstm: int = 8,
                pref_d_attn: int = 8, station_attn_dim: int = 8,
                pref_feature_mode: bool = False):
        super().__init__()
        self.vec_head = StationVectorHead(station_feat_dim, vec_dim, bid_hidden)
        self.pref_feature_mode = bool(pref_feature_mode)
        self.pref_d_attn = int(pref_d_attn)
        pref_hist_feat_dim = (hist_feat_dim_feature() if self.pref_feature_mode
                              else hist_feat_dim(n_spklu))
        self.pref_lstm = nn.LSTM(pref_hist_feat_dim, pref_d_lstm, batch_first=True)
        self.pref_attn = PreferenceAttention(station_feat_dim, pref_d_lstm, pref_d_attn)
        self.pref_gate = nn.Parameter(torch.tensor(0.0))
        self.late_merge = LatePrefMerge(vec_dim, pref_d_attn)
        self.station_attn = SmallStationAttention(vec_dim, station_attn_dim)

    def _encode_pref(self, pref_hist):
        lengths = (pref_hist.abs().sum(dim=-1) > 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(pref_hist, lengths, batch_first=True,
                                                    enforce_sorted=False)
        _, (h_n, _) = self.pref_lstm(packed)
        return h_n[-1]

    def forward(self, station_obs, mask, pref_hist=None):
        vec = self.vec_head(station_obs)                      # (B,N,vec_dim), MURNI fitur stasiun
        if pref_hist is not None:
            c_pref = self._encode_pref(pref_hist)
            attended_pref, _ = self.pref_attn(station_obs, c_pref)
            attended_pref = self.pref_gate * attended_pref
        else:
            attended_pref = torch.zeros(station_obs.shape[0], self.pref_d_attn,
                                        device=station_obs.device)
        vec = self.late_merge(vec, attended_pref)               # opsi (b): P digabung SETELAH vec ada
        vec = self.station_attn(vec, mask)                       # attention antar-stasiun
        return vec


class MasterHybridDDPGActor(nn.Module):
    """Kepala DDPG: vektor akhir -> Linear(vec_dim,1) -> tanh*bid_scale (bid kontinu,
    mekanisme ASLI MASTER, argmax menang -- TIDAK dipaksa softmax, lih. docstring
    modul). Kompatibel LANGSUNG dgn `MasterPureCritic`/`MasterPureTrainer` yg SUDAH
    ada (`forward(obs)->bid (B,N)`, sama antarmuka `MasterPureActor` lama -- hanya
    isi internalnya beda, drop-in replacement)."""

    def __init__(self, n_spklu: int, station_feat_dim: int = STATION_FEAT_DIM_MASTER,
                vec_dim: int = 8, bid_hidden: int = 16, pref_d_lstm: int = 8,
                pref_d_attn: int = 8, station_attn_dim: int = 8,
                pref_feature_mode: bool = False, bid_scale: float = 10.0):
        super().__init__()
        self.backbone = _PrefStationBackbone(n_spklu, station_feat_dim, vec_dim, bid_hidden,
                                             pref_d_lstm, pref_d_attn, station_attn_dim,
                                             pref_feature_mode)
        self.pref_lstm = self.backbone.pref_lstm   # DIBACA RLRolloutAgent.__init__ (_use_pref)
        self.head = nn.Linear(vec_dim, 1)
        self.bid_scale = float(bid_scale)

    def forward(self, station_obs, mask=None, pref_hist=None):
        if mask is None:
            mask = torch.ones(station_obs.shape[:2], dtype=torch.bool, device=station_obs.device)
        vec = self.backbone(station_obs, mask, pref_hist)
        return torch.tanh(self.head(vec).squeeze(-1)) * self.bid_scale


class MasterHybridPPOActor(nn.Module):
    """Kepala PPO: vektor akhir -> Linear(vec_dim,1) -> logit -> softmax (mask
    stasiun tak-feasible -inf) -- mekanisme kategorikal Kandidat A/HPPOPolicy.
    Kompatibel LANGSUNG dgn `MasterPurePPOCritic` (V(s), tak berubah)."""

    def __init__(self, n_spklu: int, station_feat_dim: int = STATION_FEAT_DIM_MASTER,
                vec_dim: int = 8, bid_hidden: int = 16, pref_d_lstm: int = 8,
                pref_d_attn: int = 8, station_attn_dim: int = 8,
                pref_feature_mode: bool = False):
        super().__init__()
        self.backbone = _PrefStationBackbone(n_spklu, station_feat_dim, vec_dim, bid_hidden,
                                             pref_d_lstm, pref_d_attn, station_attn_dim,
                                             pref_feature_mode)
        self.pref_lstm = self.backbone.pref_lstm
        self.head = nn.Linear(vec_dim, 1)

    def forward(self, station_obs, mask=None, pref_hist=None):
        if mask is None:
            mask = torch.ones(station_obs.shape[:2], dtype=torch.bool, device=station_obs.device)
        vec = self.backbone(station_obs, mask, pref_hist)
        logits = self.head(vec).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))


class MasterHybridDDPGInferenceAgent:
    """Evaluasi bersih varian DDPG-Hybrid -- BEDA dari `MasterPureInferenceAgent` lama
    (yg TAK membangun `mask`/`pref_hist` sungguhan, hanya mengandalkan default kosong
    di `forward()`, salah utk aktor bermodul-P). Bid TERTINGGI menang (argmax)."""

    def __init__(self, actor, forecaster=None, k: int = 3, pref_feature_mode: bool = False):
        self.actor = actor
        self.actor.eval()
        from marl_spklu.rl.forecaster import FormulaForecaster
        self.forecaster = forecaster or FormulaForecaster()
        self.k = int(k)
        self.pref_feature_mode = bool(pref_feature_mode)
        self.sids = None; self.sid_to_idx = None; self.N = None

    def bind_to_sim(self, sim):
        self.sim = sim
        self.sids = list(sim.spklus.keys())
        self.sid_to_idx = {s: i for i, s in enumerate(self.sids)}
        self.N = len(self.sids)

    def get_recommendation(self, feasible_spklus: dict):
        from marl_spklu.rl.master_paper_obs import build_joint_obs_master
        assert self.sids is not None, "panggil bind_to_sim(sim) sebelum sim.run(agent=...)"
        time_now = self.sim.current_step * self.sim.dt_minutes
        joint_obs = build_joint_obs_master(self.sim, self.sids, time_now)
        mask = np.zeros(self.N, dtype=bool)
        for sid in feasible_spklus:
            if sid in self.sid_to_idx:
                mask[self.sid_to_idx[sid]] = True
        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        # `pref_hist` TIDAK dibangun sungguhan di sini (evaluasi via `_uji_konsolidasi.K.
        # satu_run`, TAK melewati mekanisme rollout/`_build_pref_hist` biasa) -- P efektif
        # netral (nol) saat evaluasi bersih, konsisten pola `MasterPurePPOInferenceAgent`
        # (jalur `None`) DAN `MasterEVPPOInferenceAgent` (yg jg tak isi pref_hist saat eval
        # metrik-agregat, hanya saat rollout latihan sungguhan).
        with torch.no_grad():
            bids = self.actor(obs_t, mask_t, None).squeeze(0).numpy()
        feasible_idx = np.nonzero(mask)[0]
        if feasible_idx.size == 0:
            return []
        k_eff = max(1, min(self.k, int(feasible_idx.size)))
        order = feasible_idx[np.argsort(-bids[feasible_idx], kind="stable")]
        return [self.sids[int(i)] for i in order[:k_eff]]

    def predict_waits(self, feasible_spklus: dict):
        time_now = self.sim.current_step * self.sim.dt_minutes
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        return self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
