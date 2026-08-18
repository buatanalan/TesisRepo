"""Lengan T2 -- MASTER dengan STASIUN SEBAGAI AGEN dan aksi berupa *bidding*.

APA YANG DIUJI
--------------
Klaim arsitektural inti tesis adalah **transposisi peran agen**: dari stasiun (MASTER)
ke permintaan pengisian (P-PPO). Lengan ini menguji transposisi itu secara langsung,
dengan tulang punggung pembelajaran DITAHAN KONSTAN (PPO, tanpa replay).

    P-PPO           : satu agen-permintaan menilai SELURUH himpunan stasiun sekaligus.
                      Distribusi kebijakan MENGGABUNGKAN stasiun lewat softmax.
    T2 (berkas ini) : tiap stasiun adalah agen yang menawar SENDIRI. Distribusi kebijakan
                      adalah Gaussian INDEPENDEN per stasiun. Tak ada softmax.

Satu faktor berubah. Kalau P-PPO menang, transposisi peran agen terbukti membantu; kalau
kalah, klaim arsitektural tesis gugur dan itu jawaban yang jelas.

MENGAPA PERUBAHANNYA KECIL
--------------------------
`StationEncoder` (policy.py) SUDAH memproses tiap stasiun secara independen -- konteks
permintaan di-*broadcast*, tetapi tak ada stasiun yang melihat fitur stasiun lain.
Satu-satunya kopling lintas-stasiun di aktor P-PPO adalah **softmax** pada `disc_head`.

Karena itu lengan ini mewarisi seluruh encoder apa adanya dan hanya mengganti:

    keluaran kepala : logit (dinormalisasi lintas stasiun)  ->  rerata *bid* (independen)
    distribusi      : Categorical berurutan                 ->  Normal per stasiun
    seleksi         : threshold 0,20 + langit-langit k      ->  k *bid* TERENDAH

SEMANTIK BID
------------
*Bid* adalah **estimasi waktu tunggu yang diajukan stasiun**; `k` penawar TERENDAH
memenangkan rekomendasi. Nilainya dipakai HANYA untuk pemeringkatan.

⚠️ BATAS YANG WAJIB DILAPORKAN. EstWait yang DITAMPILKAN kepada pengguna tetap keluaran
jujur `forecaster` -- persis seperti seluruh lengan lain (Spesifikasi_Teknis_RL.md v2:
`a2`/`delta`/`alpha_honesty` dihapus, EstWait selalu jujur). Stasiun TIDAK dapat berbohong
kepada pengguna lewat *bid*-nya. Membuat *bid* menjadi janji yang ditampilkan akan
membuka kembali kanal ketidakjujuran yang sengaja ditutup di v2, dan mengubah lengan ini
dari "uji peran agen" menjadi "uji peran agen + kanal ketidakjujuran" -- dua faktor
sekaligus, tak dapat diatribusikan. Bila kanal itu hendak diuji, ia harus jadi lengan
TERPISAH.

EKSPLORASI
----------
Berasal dari pencuplikan Gaussian atas *bid*, bukan dari epsilon-greedy. Argumen `epsilon`
dan `threshold` tetap diterima demi keseragaman pemanggilan di `rollout.py`, tetapi
DIABAIKAN -- keduanya tak punya makna pada ruang aksi kontinu.

BATAS KLAIM
-----------
Ini BUKAN replikasi MASTER. Yang ditahan konstan dari P-PPO (dan karenanya BERBEDA dari
MASTER): PPO alih-alih DDPG, tanpa *replay*, kritik `V(s)` alih-alih `Q(o,a,p)` atas aksi
gabungan, tanpa *delayed access strategy*, tanpa *dynamic gradient re-weighting*. Yang
diklaim: **"peran agen MASTER (stasiun menawar) pada substrat pembelajaran P-PPO"**.
"""
import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.master_policy import MasterPolicy

# Bid stasiun tak-feasible tak pernah boleh menang. Dipakai sbg pengganti +inf supaya
# aman secara numerik saat di-argsort (inf ikut terurut benar, tetapi nan bila 0*inf).
BID_INF = 1e9


class MasterBiddingPolicy(MasterPolicy):
    """Stasiun-sebagai-agen; aksi = *bid* kontinu; k penawar terendah direkomendasikan.

    Mewarisi `MasterPolicy` -> tanpa encoder riwayat per-pengguna, setia pada kondisi
    informasi MASTER (aktor tidak mengondisikan pada riwayat individual).

    `disc_head` DIPAKAI ULANG sebagai kepala rerata *bid* -- bentuknya sudah tepat
    (Linear(station_hidden, 1), satu keluaran per stasiun) dan `forward()` induk karenanya
    dapat dipakai apa adanya. Yang berubah hanya TAFSIR keluarannya: bukan lagi logit yang
    akan di-softmax lintas stasiun, melainkan rerata Gaussian per stasiun yang berdiri
    sendiri.
    """

    def __init__(self, *args, bid_log_std_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Satu log-std DIBAGI seluruh stasiun -- konsisten dgn berbagi parameter antar
        # agen homogen (Gupta dkk. 2017). Std per-stasiun akan memberi tiap stasiun
        # tingkat eksplorasi sendiri, yang merusak homogenitas agen.
        self.bid_log_std = nn.Parameter(torch.full((1,), float(bid_log_std_init)))

    # ------------------------------------------------------------------ util
    def _bid_dist(self, bid_mean):
        std = torch.exp(self.bid_log_std).expand_as(bid_mean)
        return torch.distributions.Normal(bid_mean, std)

    @staticmethod
    def _pilih_terendah(bids_row, mask_row, k):
        """k *bid* TERENDAH di antara stasiun feasible, terurut menaik.

        Mengembalikan list indeks. Lantai 1 dipertahankan (sama spt aturan seleksi
        P-PPO): selama ada stasiun feasible, selalu ada minimal satu rekomendasi.
        """
        feasible_idx = np.nonzero(mask_row)[0]
        if feasible_idx.size == 0:
            return []
        k_eff = max(1, min(int(k), int(feasible_idx.size)))
        urut = feasible_idx[np.argsort(bids_row[feasible_idx], kind="stable")]
        return [int(i) for i in urut[:k_eff]]

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, k: int = 3, critic_obs_np=None,
            epsilon: float = 0.0, threshold: float = 0.20, **_abaikan):
        """`epsilon`/`threshold` DIABAIKAN -- lihat catatan EKSPLORASI di docstring modul.

        Return dict yg sama spt HPPOPolicy.act, DITAMBAH `bids` (N,) float32: aksi
        sesungguhnya yang harus disimpan di transisi, karena rasio PPO dihitung atas
        *bid*, bukan atas himpunan terpilih.
        """
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        critic_obs = (torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
                      if critic_obs_np is not None else None)

        bid_mean, value = self.forward(obs, hist, critic_obs)
        dist = self._bid_dist(bid_mean)
        bids = dist.sample()                                   # (1, N)

        mask_np = np.asarray(feasible_mask_np, dtype=bool)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool).unsqueeze(0)

        # log-prob HANYA atas stasiun feasible. Stasiun tak-feasible tak pernah bisa
        # menang, jadi *bid*-nya tak mempengaruhi hasil -- memasukkannya ke rasio PPO
        # akan menyuntikkan derau murni ke gradien.
        logp_all = dist.log_prob(bids)                         # (1, N)
        logp_total = float((logp_all * mask_t.float()).sum().item())

        bids_np = bids[0].cpu().numpy().astype("float32")
        bids_pilih = np.where(mask_np, bids_np, BID_INF)
        chosen_order = self._pilih_terendah(bids_pilih, mask_np, k)

        return {
            "chosen_indices": chosen_order,
            "n_rec": len(chosen_order),
            "logp": logp_total,
            "value": value[0].detach().cpu().numpy().astype("float64"),
            "bids": bids_np,
        }

    # ------------------------------------------------------------------ update PPO
    def evaluate(self, obs_b, mask_b, chosen_indices_b, n_rec_b, hist_b, critic_obs_b=None,
                 bids_b=None, **_abaikan):
        """`chosen_indices_b`/`n_rec_b` TIDAK dipakai -- dan itu memang benar.

        Seluruh stokastisitas kebijakan ada pada *bid*; seleksi k-terendah bersifat
        DETERMINISTIK begitu *bid* diketahui. Rasio PPO karenanya harus dihitung atas
        *bid*, bukan atas himpunan terpilih. Menghitungnya atas himpunan terpilih akan
        salah: himpunan yang sama dapat berasal dari *bid* yang sangat berbeda.
        """
        if bids_b is None:
            raise ValueError(
                "MasterBiddingPolicy.evaluate butuh `bids_b` (aksi sesungguhnya). "
                "Cek PPOTrainer.update -- jalur `use_bids` tidak aktif.")
        bid_mean, value = self.forward(obs_b, hist_b, critic_obs_b)
        dist = self._bid_dist(bid_mean)
        m = mask_b.float()
        logp = (dist.log_prob(bids_b) * m).sum(dim=-1)
        entropy = (dist.entropy() * m).sum(dim=-1)
        return logp, entropy, value
