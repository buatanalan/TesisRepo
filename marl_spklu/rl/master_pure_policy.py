"""MASTER **benar-benar murni** (2026-08-28) -- diverifikasi LANGSUNG terhadap teks
paper (Zhang dkk., "Intelligent Electric Vehicle Charging Recommendation Based on
Multi-Agent Reinforcement Learning", WWW'21, `2102.07359v1.pdf`), BUKAN cuma komentar
kode `master_ddpg_policy.py` lama (yg ternyata sendiri menyimpang dari paper di
beberapa titik -- lih. `Eksekusi_RL/ARSITEKTUR_MASTER_REFERENSI.md` §8 utk daftar
lengkap). File ini TERPISAH dari `master_ddpg_policy.py` (tak mengubahnya) supaya
checkpoint/eksperimen lama yg sudah ada tetap kompatibel.

Deviasi `master_ddpg_policy.py` lama yg DIPERBAIKI di sini (rujukan §8 dokumen):
  1. Arah bid: PAPER bid TERTINGGI menang (argmax, §3.1) -- lama TERBALIK (terendah).
  2. Urutan pooling: PAPER jumlah-DULU-baru-proyeksi (Pers. 6) -- lama proyeksi-dulu.
  3. Skor atensi: PAPER v^T tanh(W_a·x) (Pers. 4) -- lama Linear polos tanpa tanh.
  5. Transformasi p^i_t: PAPER WAJIB ReLU(W_p·I) (Pers. 10) -- lama pakai future_avail
     MENTAH tanpa transformasi.
  7. global_scalar (Gini) di kepala kritik: TAK ADA di Pers. (4)-(14) manapun -- DIHAPUS.

Keputusan desain (disepakati bersama user, 2026-08-28):
  - Observasi: 7 fitur §3.1 MURNI (STATION_FEAT_DIM_MASTER, TANPA fitur pemohon +EV)
    -- Pers. 11 (a^i_t=b^i(o^i_t)) benar-benar ditegakkan, stasiun buta thd pemohon.
  - Objektif ke-2 (pengganti CP, yg TAK dimodelkan simulator ini -- selalu 0):
    Gini/pemerataan (STREAM_GLOBAL, `rollout.py`) -- DGR jadi bermakna nyata (dua
    sinyal yg genuinely bersaing: wait individual vs pemerataan populasi), didoku-
    mentasikan EKSPLISIT sbg substitusi terukur, BUKAN CWT/CP literal paper.
"""
import torch
import torch.nn as nn

from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER


class MasterPureActor(nn.Module):
    """b^i(o^i_t), Pers. 11. MLP 3-lapisan dimensi 64 (persis §4.1.2: "three linear
    network layers with dimension 64"), bobot DIBAGI semua stasiun (permutation-
    invariant, konsisten klaim §3.2.1 & konvensi StationEncoder/Gupta dkk. 2017).

    Keluaran BUKAN lagi "ETA yg ditawarkan" (interpretasi `master_ddpg_policy.py`
    lama, karenanya di-sigmoid ke [0,bid_max] dan bid TERENDAH menang) -- di sini
    "bid" murni SKOR KOMPETITIF ABSTRAK sesuai paper (argmax menang, §3.1), tak
    punya makna fisik langsung. Dibatasi via tanh (BUKAN dari paper -- paper tak
    menyebutkan aktivasi keluaran eksplisit; tanh dipilih krn DDPG deterministik
    BUTUH keluaran berbatas utk stabilitas gradien/target -- keputusan rekayasa yg
    diperlukan, didokumentasikan eksplisit sbg bukan bagian rumus paper)."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, hidden: int = 64,
                bid_scale: float = 10.0):
        super().__init__()
        self.bid_scale = float(bid_scale)
        self.net = nn.Sequential(
            nn.Linear(station_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, station_obs):
        """station_obs: (B,N,F). Return bid: (B,N) di [-bid_scale, bid_scale]."""
        raw = self.net(station_obs).squeeze(-1)
        return torch.tanh(raw) * self.bid_scale


class MasterPureAttentivePooling(nn.Module):
    """Persis Pers. (4)-(6) -- BUKAN reinterpretasi `AttentiveJointPooling` lama.

        e^i_t = v^T tanh(W_a (o^i_t ⊕ a^i_t ⊕ p^i_t))                  ...(4)
        alpha^i_t = softmax_i(e^i_t)  atas agen aktif                   ...(5)
        x_t = ReLU(W_c · Sum_i alpha^i_t (o^i_t ⊕ a^i_t ⊕ p^i_t))        ...(6)

    Beda kunci dari `AttentiveJointPooling` lama: bobot atensi dikalikan ke FITUR
    MENTAH (bukan embedding), PENJUMLAHAN dilakukan SEBELUM proyeksi, W_c+ReLU
    diterapkan SEKALI SAJA setelah penjumlahan (bukan per-stasiun sebelum jumlah)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.W_a = nn.Linear(in_dim, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.W_c = nn.Linear(in_dim, hidden)

    def forward(self, raw, mask):
        """raw: (B,N,in_dim) = concat(o,a,p) MENTAH per stasiun. mask: (B,N) bool."""
        e = self.v(torch.tanh(self.W_a(raw))).squeeze(-1)            # (B,N), Pers.4
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)                              # Pers.5
        alpha = torch.nan_to_num(alpha, nan=0.0)
        summed = torch.einsum("bn,bnd->bd", alpha, raw)                # Sum PADA fitur mentah
        x_t = torch.relu(self.W_c(summed))                             # Pers.6
        return x_t, alpha


class MasterPureCritic(nn.Module):
    """Q^k(x_t), Pers. (6)+(8)/(12) -- K kepala (Multi-Critics). TANPA global_scalar
    (tak ada di rumus paper manapun -- lih. docstring modul poin 7).

    `p^i_t = ReLU(W_p · I^i_t)` (Pers. 10) DIHITUNG DI SINI (bukan dianggap sudah
    jadi spt `future_avail` lama) -- `I^i_t` MENTAH (bisa negatif, lih.
    `master_pure_trainer.py::snapshot_slots_raw`) masuk method `forward` APA ADANYA,
    ditransformasi lewat `W_p`+ReLU SEBELUM digabung ke (o⊕a).

    Interpretasi ukuran jaringan (§4.1.2: "three linear layers dimension 64" utk
    kritik) -- DIDUGA merujuk W_a+W_c+head sbg "tiga lapisan", BUKAN dipastikan
    eksplisit dari teks paper (paper tak memberi diagram lapisan-demi-lapisan).
    Diberi catatan eksplisit sbg INFERENSI, bukan fakta pasti."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, hidden: int = 64,
                n_critics: int = 2, p_dim: int = 64):
        super().__init__()
        self.n_critics = int(n_critics)
        self.W_p = nn.Linear(1, p_dim)                     # Pers. 10
        in_dim = station_feat_dim + 1 + p_dim               # obs + aksi/bid + p^i_t
        self.pool = MasterPureAttentivePooling(in_dim, hidden)
        self.head = nn.Linear(hidden, self.n_critics)        # lih. catatan inferensi di atas

    def forward(self, joint_obs, joint_action, mask, I_raw):
        """joint_obs:(B,N,F) joint_action:(B,N) mask:(B,N) I_raw:(B,N) MENTAH (bisa
        negatif, BELUM ditransformasi) -> Q:(B,K)."""
        p = torch.relu(self.W_p(I_raw.unsqueeze(-1)))                     # (B,N,p_dim), Pers.10
        raw = torch.cat([joint_obs, joint_action.unsqueeze(-1), p], dim=-1)
        x_t, attn_weights = self.pool(raw, mask)
        return self.head(x_t), attn_weights
