"""Lengan MASTER murni: aktor TANPA encoder riwayat per-pengguna.

LATAR
-----
Ditemukan 2026-08-18 bahwa `HPPOPolicy` sudah memuat komponen turunan PDQN: encoder `hist`
(`rollout.py::_build_hist`) adalah LSTM atas riwayat interaksi per-pengguna -- pola
arsitektural yang sama dengan modul preferensi PDQN, hanya isinya diubah:

    hist      (K=5,  dim=4)  : (kepatuhan, janji EstWait, wait-default, kesenjangan terealisasi)
    pref_hist (K=10, dim=2N) : one-hot ganda (a_hat, a) -- IDENTITAS stasiun

Akibatnya perbandingan H-PPO vs P-PPO BUKAN "tanpa PDQN vs dengan PDQN" -- keduanya memuat
encoder riwayat turunan PDQN, hanya isinya berbeda. Tanpa baseline yang benar-benar tanpa
encoder riwayat, kontribusi adaptasi itu tak dapat diatribusikan.

Berkas ini menyediakan baseline tersebut. `policy.py` dan `p_ppo_policy.py` TIDAK diubah.

RANCANGAN 2x2
-------------
    lengan       hist   pref_hist   mewakili
    MASTER        -         -       baseline cabang pemerataan, murni
    MASTER+P      -         v       integrasi PDQN setia-paper
    H-PPO         v         -       MASTER + adaptasi riwayat      (sudah ada)
    P-PPO         v         v       MASTER + adaptasi + setia-paper (sudah ada)

DUA VARIAN MASTER, karena menjawab hal berbeda
----------------------------------------------
    MasterPolicy       LSTM riwayat DIHAPUS; konteks aktor & kritik menyusut sebesar
                       hist_hidden. Setia arsitektur MASTER (aktornya tidak mengondisikan
                       pada riwayat individual). Jumlah parameter BERBEDA dari H-PPO.
    MasterPolicyEqCap  LSTM tetap ada, keluarannya DIPAKSA NOL. Jumlah parameter SAMA
                       dengan H-PPO -> mengisolasi kontribusi INFORMASI, bukan kapasitas.

Keduanya perlu: yang pertama baseline yang benar terhadap rujukan, yang kedua ablasi yang
benar untuk atribusi. Melaporkan salah satu saja membuka bantahan dari sisi yang lain.

BATAS KLAIM
-----------
Ini BUKAN replikasi MASTER (Zhang dkk. 2021). MASTER memakai DDPG dengan bidding kontinu;
di sini PPO diskrit (top-K/threshold). Yang diklaim: "keluarga arsitektur MASTER tanpa
encoder riwayat per-pengguna". Mengklaim replikasi akan mudah dibantah.

CATATAN: menghapus `hist` juga menghapusnya dari KRITIK (`critic_context_dim` memuatnya).
Kritik tetap menerima trust asli sebagai variabel privileged, jadi tidak kehilangan
informasi trust sepenuhnya -- perbedaan ini harus disebut saat melaporkan.
"""
import torch

from marl_spklu.rl.policy import (HPPOPolicy, StationEncoder, STATION_FEAT_DIM,
                                  CRITIC_STATION_FEAT_DIM)
from marl_spklu.rl.p_ppo_policy import PPPOPolicy, PREF_D_LSTM, PREF_D_ATTN


def _buang_jalur_hist(pol, hidden, pref_d_attn=0):
    """Hapus LSTM riwayat dan bangun ulang encoder dgn konteks yang menyusut.

    Dipanggil SETELAH `super().__init__()` karena kelas induk membangun encoder memakai
    `hist_hidden`; di sini keduanya diganti dengan versi tanpa suku itu.

    `del self.hist_lstm` melepas submodul dari `_modules`, sehingga parameternya tak ikut
    `state_dict()` maupun optimizer -- inilah bedanya dgn MasterPolicyEqCap yang menahannya.
    """
    del pol.hist_lstm
    pol.hist_hidden = 0

    # Aktor: konteks = skalar (+ preferensi bila ada), tanpa c_t.
    actor_context_dim = pol.scalar_dim + pref_d_attn
    pol.station_encoder = StationEncoder(STATION_FEAT_DIM, actor_context_dim,
                                         pol.station_hidden)

    # Kritik: konteks = skalar gabungan (aktor + kritik), tanpa c_t.
    merged_station_feat_dim = STATION_FEAT_DIM + CRITIC_STATION_FEAT_DIM
    merged_scalar_dim = pol.scalar_dim + pol.critic_scalar_dim
    pol.critic_station_encoder = StationEncoder(merged_station_feat_dim, merged_scalar_dim,
                                                hidden)


class MasterPolicy(HPPOPolicy):
    """MASTER murni -- tanpa encoder riwayat per-pengguna sama sekali."""

    def __init__(self, obs_dim: int, critic_obs_dim: int, n_spklu: int, hidden: int = 128,
                 hist_hidden: int = 16, station_hidden: int = 64, n_critics: int = 1):
        super().__init__(obs_dim, critic_obs_dim, n_spklu, hidden=hidden,
                         hist_hidden=hist_hidden, station_hidden=station_hidden,
                         n_critics=n_critics)
        _buang_jalur_hist(self, hidden)

    def _encode_hist(self, hist):
        """Vektor konteks berlebar NOL -- `torch.cat` di `forward()` menjadi no-op,
        sehingga `forward()` induk dapat dipakai apa adanya tanpa disunting."""
        return torch.zeros(hist.shape[0], 0, device=hist.device, dtype=torch.float32)


class MasterPolicyEqCap(HPPOPolicy):
    """MASTER dgn KAPASITAS SETARA H-PPO -- LSTM riwayat tetap ada (jumlah parameter sama
    persis), tetapi keluarannya dipaksa nol sehingga tak ada INFORMASI riwayat yang mengalir.

    Selisih terhadap H-PPO karenanya hanya dapat berasal dari informasi riwayat, bukan dari
    kapasitas parameter -- prinsip yang sama dengan `PPPOPolicy(use_preference=False)`.
    """

    def _encode_hist(self, hist):
        return torch.zeros(hist.shape[0], self.hist_hidden, device=hist.device,
                           dtype=torch.float32)


class MasterPrefPolicy(PPPOPolicy):
    """MASTER + modul preferensi PDQN setia-paper, TANPA adaptasi riwayat.

    Menguji apakah modul preferensi setia-paper (identitas stasiun) cukup dengan sendirinya,
    tanpa jejak kepatuhan yang ditambahkan tesis ini.
    """

    def __init__(self, obs_dim: int, critic_obs_dim: int, n_spklu: int, hidden: int = 128,
                 hist_hidden: int = 16, station_hidden: int = 64,
                 pref_d_lstm: int = PREF_D_LSTM, pref_d_attn: int = PREF_D_ATTN,
                 pref_feature_mode: bool = False, use_preference: bool = True,
                 n_critics: int = 1):
        super().__init__(obs_dim, critic_obs_dim, n_spklu, hidden=hidden,
                         hist_hidden=hist_hidden, station_hidden=station_hidden,
                         pref_d_lstm=pref_d_lstm, pref_d_attn=pref_d_attn,
                         pref_feature_mode=pref_feature_mode,
                         use_preference=use_preference, n_critics=n_critics)
        # pref_d_attn DIPERTAHANKAN di konteks aktor -- yang dibuang hanya c_t.
        _buang_jalur_hist(self, hidden, pref_d_attn=self.pref_d_attn)

    def _encode_hist(self, hist):
        return torch.zeros(hist.shape[0], 0, device=hist.device, dtype=torch.float32)
