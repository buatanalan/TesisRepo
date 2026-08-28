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
                n_critics: int = 2, p_dim: int = 64):
        super().__init__()
        self.n_critics = int(n_critics)
        self.W_p = nn.Linear(1, p_dim)
        in_dim = station_feat_dim + p_dim   # obs + p^i_t (TANPA aksi)
        self.pool = MasterPurePPOAttentivePooling(in_dim, hidden)
        self.head = nn.Linear(hidden, self.n_critics)

    def forward(self, joint_obs, mask, I_raw):
        """joint_obs:(B,N,F) mask:(B,N) I_raw:(B,N) MENTAH -> V:(B,K)."""
        p = torch.relu(self.W_p(I_raw.unsqueeze(-1)))
        raw = torch.cat([joint_obs, p], dim=-1)
        x_t, attn_weights = self.pool(raw, mask)
        return self.head(x_t), attn_weights
