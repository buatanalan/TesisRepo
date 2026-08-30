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
from marl_spklu.rl.pdqn_policy import (PreferenceAttention, hist_feat_dim,
                                       hist_feat_dim_feature, hist_feat_dim_feature_outcome)


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
                pref_feature_mode: bool = False, pref_pair_outcome: bool = False,
                pref_gate_init: float = 0.0, use_station_attn: bool = True,
                pref_hist_k: int = None):
        super().__init__()
        # `pref_hist_k` (2026-08-30, ABLASI): override panjang jendela riwayat P
        # (baku None -> ikut PDQN_HIST_K=10 global). Disimpan di sini (bukan tensor,
        # tak masuk state_dict) supaya rollout agent (train MAUPUN eval) bisa
        # MENURUNKANNYA dari aktor -- pola SAMA `station_feat_dim`, menjamin latih & uji
        # selalu memakai jendela yg sama tanpa perlu diteruskan terpisah di tiap panggilan.
        self.pref_hist_k = pref_hist_k
        # `use_station_attn=False` (2026-08-30, ABLASI): lewati `SmallStationAttention`
        # sepenuhnya di forward() -- utk mengisolasi kontribusi Modul P TANPA atensi
        # antar-stasiun (varian "P saja"), memisahkan efek station attention (base
        # hybrid) dari efek P murni pd perbandingan 3-arah: gabungan/attn-saja/P-saja.
        # Modul TETAP dibuat (checkpoint kompatibel antar-varian), hanya dilewati.
        self.use_station_attn = bool(use_station_attn)
        # Disimpan agar rollout agent & trainer bisa MENURUNKAN dimensi observasi dari
        # aktor (bukan diteruskan terpisah) -- menjamin obs yang dibangun selalu cocok.
        # 7  = STATION_FEAT_DIM_MASTER    (§3.1 murni, stasiun buta thd pemohon)
        # 10 = STATION_FEAT_DIM_MASTER_EV (+ jarak relatif, SoC, kapasitas baterai)
        self.station_feat_dim = int(station_feat_dim)
        self.vec_head = StationVectorHead(station_feat_dim, vec_dim, bid_hidden)
        self.pref_feature_mode = bool(pref_feature_mode)
        # Blok HASIL [complied, realized_gap_norm] ditempelkan di belakang pasangan fitur
        # (2026-08-29) -- lihat pdqn_policy.py::PREF_OUTCOME_DIM. WAJIB pref_feature_mode.
        # Dgn ini `pref_lstm` menduga preferensi DAN kepercayaan sekaligus, sehingga
        # `hist_lstm` terpisah tak lagi diperlukan (dan tak pernah ada di kelas ini).
        self.pref_pair_outcome = bool(pref_pair_outcome) and self.pref_feature_mode
        self.pref_d_attn = int(pref_d_attn)
        if self.pref_pair_outcome:
            pref_hist_feat_dim = hist_feat_dim_feature_outcome()
        elif self.pref_feature_mode:
            pref_hist_feat_dim = hist_feat_dim_feature()
        else:
            pref_hist_feat_dim = hist_feat_dim(n_spklu)
        self.pref_lstm = nn.LSTM(pref_hist_feat_dim, pref_d_lstm, batch_first=True)
        self.pref_attn = PreferenceAttention(station_feat_dim, pref_d_lstm, pref_d_attn)
        # `pref_gate_init` (2026-08-29): nilai AWAL gerbang preferensi. Baku 0.0 =
        # perilaku lama (GTrXL zero-init). MASALAH yg ditemukan: pada gerbang PERSIS 0,
        # gradien yg sampai ke `pref_lstm`/`pref_attn` adalah PERSIS NOL (terverifikasi:
        # gate=0.0 -> |grad pref_lstm|=0.000e+00; gate=0.05 -> 6.2e-03), sehingga modul
        # preferensi TAK BISA mulai belajar sampai gerbangnya bergeser sendiri -- padahal
        # gerbang itu sendiri bergerak lambat. Deadlock ayam-telur yg sama kelasnya dgn
        # `station_attn_gate` raw=-5.0 yg dulu terbukti MACET (turunan sigmoid 0,0067).
        # Nilai kecil bukan-nol (mis. 0.1) memutus deadlock TANPA meninggalkan semangat
        # "mulai hampir tak berkontribusi". CATATAN: gerbang ini BEBAS TANDA (bukan
        # sigmoid spt LatePrefMerge/SmallStationAttention) -- risiko konvergen ke tanda
        # berbeda antar-seed tetap ada, lih. diagnosis station_attn 2026-08-24.
        self.pref_gate = nn.Parameter(torch.tensor(float(pref_gate_init)))
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
        if self.use_station_attn:
            vec = self.station_attn(vec, mask)                   # attention antar-stasiun
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
                pref_feature_mode: bool = False, bid_scale: float = 10.0,
                pref_pair_outcome: bool = False, pref_gate_init: float = 0.0,
                use_station_attn: bool = True, pref_hist_k: int = None):
        super().__init__()
        self.backbone = _PrefStationBackbone(n_spklu, station_feat_dim, vec_dim, bid_hidden,
                                             pref_d_lstm, pref_d_attn, station_attn_dim,
                                             pref_feature_mode, pref_pair_outcome,
                                             pref_gate_init, use_station_attn, pref_hist_k)
        self.pref_lstm = self.backbone.pref_lstm
        self.station_feat_dim = self.backbone.station_feat_dim   # DIBACA RLRolloutAgent.__init__ (_use_pref)
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
                pref_feature_mode: bool = False, pref_pair_outcome: bool = False,
                pref_gate_init: float = 0.0, use_station_attn: bool = True,
                pref_hist_k: int = None):
        super().__init__()
        self.backbone = _PrefStationBackbone(n_spklu, station_feat_dim, vec_dim, bid_hidden,
                                             pref_d_lstm, pref_d_attn, station_attn_dim,
                                             pref_feature_mode, pref_pair_outcome,
                                             pref_gate_init, use_station_attn, pref_hist_k)
        self.pref_lstm = self.backbone.pref_lstm
        self.station_feat_dim = self.backbone.station_feat_dim
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
    di `forward()`, salah utk aktor bermodul-P). Bid TERTINGGI menang (argmax).

    DIUBAH 2026-08-29 -- versi sebelumnya memanggil `self.actor(obs, mask, None)`, yakni
    `pref_hist=None`, sehingga modul P **netral (nol) sepanjang evaluasi** dan metrik
    pembanding tak pernah mengukur kontribusinya. Komentar lama membenarkannya dgn
    menyebut `MasterEVPPOInferenceAgent` sbg preseden -- KLAIM ITU KELIRU: kelas tsb
    justru mendelegasikan ke rollout agent penuh. Kini kelas ini memakai pola delegasi
    yang sama (`MasterPureRolloutAgent` dgn `noise_std=0` = deterministik, argmax bid),
    supaya jalur `pref_hist` saat UJI identik dgn saat LATIH."""

    def __init__(self, actor, forecaster=None, k: int = 3, pref_feature_mode: bool = False):
        self.actor = actor
        self.actor.eval()
        from marl_spklu.rl.forecaster import FormulaForecaster
        self.forecaster = forecaster or FormulaForecaster()
        self.k = int(k)
        self.pref_feature_mode = bool(pref_feature_mode)
        self._roll = None

    def bind_to_sim(self, sim):
        # Impor lokal: menghindari lingkar impor policy <-> trainer.
        from marl_spklu.rl.master_pure_trainer import MasterPureRolloutAgent
        from marl_spklu.rl.rollout import RewardCalculatorStub
        _bb = getattr(self.actor, "backbone", None)
        self._roll = MasterPureRolloutAgent(
            self.actor, sim, RewardCalculatorStub(), self.forecaster,
            noise_std=0.0, k=self.k,                      # noise 0 -> deterministik
            pref_feature_mode=getattr(_bb, "pref_feature_mode", False),
            pref_pair_outcome=getattr(_bb, "pref_pair_outcome", False))
        self.sim = sim

    def get_recommendation(self, feasible_spklus: dict):
        assert self._roll is not None, "panggil bind_to_sim(sim) sebelum sim.run(agent=...)"
        recs = self._roll.get_recommendation(feasible_spklus)
        self._roll.transitions.clear()
        return recs

    def predict_waits(self, feasible_spklus: dict):
        return self._roll.predict_waits(feasible_spklus)

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        # WAJIB -- di sinilah `_record_pref` menambah pasangan (a_hat,a) ke riwayat.
        self._roll.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        self._roll.transitions.clear()

    def on_charge_complete(self, user):
        # WAJIB -- di sinilah blok HASIL (`realized_gap_norm`) di-backfill.
        self._roll.on_charge_complete(user)
        self._roll.transitions.clear()
