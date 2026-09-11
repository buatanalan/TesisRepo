"""Master-PPO (2026-08-28) -- MASTER-murni (`master_pure_policy.py`) dgn tulang punggung
PPO standar menggantikan DDPG, sesuai permintaan eksplisit user. Kelas BARU, TIDAK
menumpuk pada `master_bidding_policy.py` (varian PPO+bidding LAMA yg SUDAH ADA, tapi
eksplisit BUKAN replikasi MASTER -- tanpa Delayed Access, tanpa DGR, kritik generik).

KONSEKUENSI STRUKTURAL dari memilih PPO standar (disepakati bersama user):
    Kritik `Q(o,a,p)` (Pers. 4, DDPG, MENERIMA aksi) -> `V(o,p)` (PPO standar, TIDAK
    menerima aksi -- baseline GAE harus independen-aksi). Mekanisme atensi (Pers. 4-6)
    DIPERTAHANKAN, HANYA suku `a^i_t` dihapus dari gabungan (o^i_t ⊕ p^i_t sisanya).

    Akibat LANGSUNG: gap-ratio DGR (Pers. 13, `Q*(x_t)|a=b*(o)` -- BERSYARAT PADA AKSI
    SPESIALIS) TAK LAGI TERDEFINISI utk V(s) (V scr definisi tak bisa dievaluasi "seolah
    aksi lain diambil"). Diganti gap-ratio berbasis RETURN teragregasi (pola SAMA
    `ppo.py::PPOTrainer._compute_beta`, dipakai lengan PPO lain repo ini) -- bukan jalan
    pintas, tapi satu-satunya adaptasi yg koheren utk V(s) standar.

Observasi 7 fitur §3.1 MURNI (sama `master_pure_policy.py`). Objektif ke-2 = gini
(pengganti CP, keputusan sama 2026-08-28)."""
import torch
import torch.nn as nn

from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER


class MasterPurePPOActor(nn.Module):
    """b^i(o^i_t), Pers. 11 -- MLP 3-lapisan dim 64 (§4.1.2), bobot dibagi. Keluaran
    rerata bid MENTAH (tanpa tanh -- distribusi Normal PPO butuh dukungan tak-terbatas
    utk log-prob yg benar; std ditangani terpisah lewat `bid_log_std`, pola SAMA
    `master_bidding_policy.py::_BiddingMixin`)."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, hidden: int = 64,
                bid_log_std_init: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(station_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Satu log-std DIBAGI seluruh stasiun (agen homogen, Gupta dkk. 2017) -- pola
        # sama _BiddingMixin, konsisten prinsip bobot-dibagi di seluruh arsitektur ini.
        self.bid_log_std = nn.Parameter(torch.full((1,), float(bid_log_std_init)))

    def forward(self, station_obs):
        """station_obs: (B,N,F) -> bid_mean: (B,N)."""
        return self.net(station_obs).squeeze(-1)

    def dist(self, bid_mean):
        std = torch.exp(self.bid_log_std).expand_as(bid_mean)
        return torch.distributions.Normal(bid_mean, std)


class MasterPurePPOActorV2(nn.Module):
    """Varian V2 (2026-09-12) -- memakai `StationVectorHead` (encoder kecil
    per-stasiun, SAMA dgn yg dipakai backbone hybrid) alih-alih MLP polos,
    TANPA atensi/Modul P -- dipakai KHUSUS pipeline PURE3 baru, TIDAK
    menggantikan `MasterPurePPOActor` asli (kompatibilitas checkpoint lama).

    Pola forward/dist IDENTIK `MasterPurePPOActor` (bid_mean MENTAH tanpa
    tanh, std lewat `bid_log_std` dibagi seluruh stasiun) -- HANYA `self.net`
    diganti jadi `vec_head` (encoder 7->16->8, `StationVectorHead`) + `head`
    (Linear(8,1)) proyeksi akhir ke bid_mean skalar per stasiun."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, vec_dim: int = 8,
                hidden: int = 16, bid_log_std_init: float = 0.0):
        super().__init__()
        from marl_spklu.rl.master_pure_hybrid_policy import StationVectorHead
        self.vec_head = StationVectorHead(station_feat_dim, vec_dim=vec_dim, hidden=hidden)
        self.head = nn.Linear(vec_dim, 1)
        self.bid_log_std = nn.Parameter(torch.full((1,), float(bid_log_std_init)))

    def forward(self, station_obs):
        """station_obs: (B,N,F) -> bid_mean: (B,N)."""
        vec = self.vec_head(station_obs)
        return self.head(vec).squeeze(-1)

    def dist(self, bid_mean):
        std = torch.exp(self.bid_log_std).expand_as(bid_mean)
        return torch.distributions.Normal(bid_mean, std)


class MasterPurePPOAttentivePooling(nn.Module):
    """Pers. (4)-(6), TANPA suku aksi `a^i_t` (V(s), bukan Q(o,a,p) -- lih. docstring
    modul). e^i_t = v^T tanh(W_a(o^i_t⊕p^i_t)); alpha=softmax; x_t=ReLU(W_c·Sum(alpha·raw))."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.W_a = nn.Linear(in_dim, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.W_c = nn.Linear(in_dim, hidden)

    def forward(self, raw, mask):
        e = self.v(torch.tanh(self.W_a(raw))).squeeze(-1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        summed = torch.einsum("bn,bnd->bd", alpha, raw)
        x_t = torch.relu(self.W_c(summed))
        return x_t, alpha


class MasterPurePPOCritic(nn.Module):
    """V^k(x_t) -- K kepala (Multi-Critics), TANPA aksi. `p^i_t=ReLU(W_p·I^i_t)`
    (Pers. 10) tetap dihitung persis sama versi DDPG -- Delayed Access Strategy TAK
    bergantung pada bentuk tulang-punggung latih (DDPG/PPO), murni properti observasi
    kritik."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, hidden: int = 64,
                n_critics: int = 2, p_dim: int = 64, n_priv: int = 1):
        super().__init__()
        self.n_critics = int(n_critics)
        # `n_priv` (2026-09-09): jumlah fitur ISTIMEWA per stasiun yang masuk W_p.
        #   n_priv=1 -> I^i_t saja (Pers. 10 MASTER, perilaku BAKU & satu-satunya yang
        #               kompatibel dgn seluruh checkpoint yang sudah ada)
        #   n_priv=2 -> mode `wait_trust`: [wait_actual_norm, trust_saat_keputusan]
        # Mengubah nilai ini mengubah bentuk W_p sehingga checkpoint lama TIDAK dapat
        # dimuat -- disengaja, supaya percampuran konfigurasi gagal keras, bukan diam2.
        self.n_priv = int(n_priv)
        self.W_p = nn.Linear(self.n_priv, p_dim)
        in_dim = station_feat_dim + p_dim   # obs + p^i_t (TANPA aksi)
        self.pool = MasterPurePPOAttentivePooling(in_dim, hidden)
        self.head = nn.Linear(hidden, self.n_critics)

    def forward(self, joint_obs, mask, I_raw):
        """joint_obs:(B,N,F) mask:(B,N) I_raw:(B,N) atau (B,N,n_priv) MENTAH -> V:(B,K)."""
        if I_raw.dim() == 2:
            I_raw = I_raw.unsqueeze(-1)
        p = torch.relu(self.W_p(I_raw))
        raw = torch.cat([joint_obs, p], dim=-1)
        x_t, attn_weights = self.pool(raw, mask)
        return self.head(x_t), attn_weights
