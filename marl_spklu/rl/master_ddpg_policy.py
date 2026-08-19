"""Jaringan MASTER "asli": aktor DESENTRALISASI per-stasiun (DDPG, bidding kontinu) +
Centralized Attentive Critic multi-kepala. Pelengkap `master_bidding_policy.py` (yang
eksplisit BUKAN replikasi -- substrat PPO, softmax pooling generik, tanpa akses
tertunda) dengan versi yang menahan KEEMPAT mekanisme MASTER (lihat modul saudara
`master_ddpg_trainer.py` utk Delayed Access Strategy & Dynamic Gradient Re-weighting).

ARSITEKTUR
----------
Aktor  b^i(o^i_t) -> bid IND. per stasiun. SATU jaringan dgn BOBOT DIBAGI lintas
       stasiun (agen homogen, Gupta dkk. 2017 -- sama spt konvensi `StationEncoder`
       di policy.py) -- dipanggil baris-demi-baris per stasiun, TIDAK melihat fitur
       stasiun lain (desentralisasi SESUNGGUHNYA, bukan broadcast+pool).
Kritik Q^k(o, a, p) -- SATU jaringan TERPUSAT (CTDE): melihat SELURUH observasi &
       bid stasiun sekaligus (o = joint obs (N,7), a = joint bids (N,)), meringkasnya
       lewat ATENSI (bobot dipelajari per stasiun, "mekanisme atensi yang secara
       otomatis menghitung bobot pengaruh koordinasi tiap stasiun" -- persis klaim
       user), plus `p` = fitur privileged tambahan (Delayed Access, lihat trainer).
       Multi-kepala K (default K=2, lihat master_ddpg_trainer.py) utk Dynamic
       Gradient Re-weighting.
"""
import torch
import torch.nn as nn

from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER


class MasterStationActor(nn.Module):
    """b^i(o^i_t): MLP kecil, BOBOT DIBAGI semua stasiun -- dipanggil per-baris.
    Deterministik (DDPG asli); eksplorasi via noise OU DITAMBAHKAN DI LUAR jaringan
    (sama pola dgn `pdqn_ddpg.py::PDQNDDPGRolloutAgent._sample_noise`)."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, hidden: int = 64,
                bid_max: float = 120.0):
        super().__init__()
        self.bid_max = float(bid_max)   # menit -- rentang bid (ETA yg diajukan)
        self.net = nn.Sequential(
            nn.Linear(station_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, station_obs):
        """station_obs: (B, N, F) SEMUA stasiun sekaligus (batched utk efisiensi --
        scr matematis identik memanggil per-baris krn bobot dibagi & tak ada
        percampuran lintas-stasiun di dalam `net`). Return bid: (B, N) di [0, bid_max]
        (ETA yg ditawarkan SELALU tak-negatif -- beda dgn `_BiddingMixin` lama yang
        bertumpu pd EstWait forecaster+delta bertanda)."""
        raw = self.net(station_obs).squeeze(-1)          # (B, N)
        return torch.sigmoid(raw) * self.bid_max


class AttentiveJointPooling(nn.Module):
    """Centralized Attentive Critic -- inti klaim MASTER poin 2: "mekanisme atensi yang
    secara otomatis menghitung bobot pengaruh koordinasi tiap stasiun". Beda dgn
    `policy.py::AttentionPooling` (yg meringkas EMBEDDING stasiun POST-encoder,
    dipakai kritik V(s) SEMUA lengan lain): kelas ini secara eksplisit dikondisikan
    pada (observasi, AKSI/bid) gabungan tiap stasiun -- bobot atensi jadi fungsi dari
    SEBERAPA KOMPETITIF bid stasiun itu, bukan cuma fitur statisnya."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)

    def forward(self, obs_action, mask):
        """obs_action: (B,N,in_dim). mask: (B,N) bool (stasiun tak-feasible dikeluarkan
        dari atensi, bukan cuma diberi bid sentinel -- mencegah kritik "memperhatikan"
        stasiun yang sebetulnya tak relevan bagi keputusan ini)."""
        emb = self.embed(obs_action)                      # (B,N,H)
        logits = self.score(emb).squeeze(-1)               # (B,N)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)             # (B,N), jumlah=1
        weights = torch.nan_to_num(weights, nan=0.0)         # baris tanpa stasiun feasible (jarang)
        pooled = torch.einsum("bn,bnh->bh", weights, emb)    # (B,H)
        return pooled, weights


class MasterAttentiveCritic(nn.Module):
    """Q^k(o, a, p) -- K kepala (Multi-Critics, lihat master_ddpg_trainer.py utk bobot
    Dynamic Gradient Re-weighting yg MENGGABUNGKAN kepala ini pada gradien aktor).

    `p` (privileged, Delayed Access Strategy): fitur GLOBAL tambahan, hanya milik
    kritik, TAK PERNAH terlihat aktor -- di sini `future_avail` (N,) = slot tersedia
    NYATA di t+d per stasiun (lihat master_ddpg_trainer.py::_SlotAvailLog), dan
    `global_scalar` (mis. Gini utilisasi saat ini) sbg konteks tambahan skalar."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER, hidden: int = 128,
                n_critics: int = 2, global_scalar_dim: int = 1):
        super().__init__()
        self.n_critics = int(n_critics)
        # in_dim per-stasiun: obs (F) + aksi/bid (1) + future_avail privileged (1).
        in_dim = station_feat_dim + 1 + 1
        self.pool = AttentiveJointPooling(in_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + global_scalar_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_critics),
        )

    def forward(self, joint_obs, joint_action, mask, future_avail, global_scalar):
        """joint_obs:(B,N,F) joint_action:(B,N) mask:(B,N) future_avail:(B,N)
        global_scalar:(B,global_scalar_dim) -> Q:(B,K)."""
        x = torch.cat([joint_obs, joint_action.unsqueeze(-1), future_avail.unsqueeze(-1)], dim=-1)
        pooled, attn_weights = self.pool(x, mask)
        out = self.head(torch.cat([pooled, global_scalar], dim=-1))
        return out, attn_weights
