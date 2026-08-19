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

from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.master_policy import MasterPolicy, MasterPrefPolicy

# Bid stasiun tak-feasible tak pernah boleh menang. Dipakai sbg pengganti +inf supaya
# aman secara numerik saat di-argsort (inf ikut terurut benar, tetapi nan bila 0*inf).
BID_INF = 1e9



class _BiddingMixin:
    """Bentuk aksi *bidding*: skor per stasiun ditafsirkan sebagai TAKSIRAN WAKTU TUNGGU,
    k taksiran TERENDAH direkomendasikan.

    Dipisah sebagai mixin karena bentuk aksi ini ORTOGONAL terhadap peran agen. Yang
    menentukan peran agen adalah kelas dasarnya:

        MasterPolicy  -> tanpa riwayat pengguna  -> agen = STASIUN
        HPPOPolicy    -> + `hist_lstm`           -> agen = PERMINTAAN
        PPPOPolicy    -> + modul preferensi      -> agen = PERMINTAAN

    ⚠️ CATATAN YANG WAJIB MASUK NASKAH. Dengan Gaussian INDEPENDEN per stasiun, "stasiun
    sebagai agen" dan "permintaan sebagai agen yang mengeluarkan N taksiran" menghasilkan
    fungsi yang IDENTIK -- logp, gradien, dan seleksi sama persis. Perbedaan peran agen
    baru menjadi nyata secara komputasi ketika agen mengondisikan pada RIWAYAT pengguna
    yang sedang meminta, karena hanya permintaan yang punya riwayat; stasiun tidak
    "mengingat" pengguna tertentu. Karena itu `hist_lstm` adalah penanda operasional
    peran agen di sini, bukan sekadar tambahan kapasitas.

    `disc_head` DIPAKAI ULANG sebagai kepala rerata bid -- bentuknya sudah tepat
    (Linear(station_hidden, 1)), sehingga `forward()` kelas dasar dipakai apa adanya.
    Yang berubah hanya TAFSIR keluarannya: bukan logit yang akan di-softmax lintas
    stasiun, melainkan rerata Gaussian per stasiun yang berdiri sendiri.
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

    def _fwd(self, obs, hist, critic_obs, pref_hist):
        """Panggil forward kelas dasar; teruskan pref_hist HANYA bila kelasnya punya
        modul preferensi (HPPOPolicy.forward tak menerima argumen itu)."""
        if pref_hist is not None and hasattr(self, "pref_lstm"):
            return self.forward(obs, hist, critic_obs, pref_hist=pref_hist)
        return self.forward(obs, hist, critic_obs)

    @staticmethod
    def _pilih_terendah(bids_row, mask_row, k):
        """k bid TERENDAH di antara stasiun feasible, terurut menaik. Lantai 1
        dipertahankan (sama spt aturan seleksi P-PPO)."""
        feasible_idx = np.nonzero(mask_row)[0]
        if feasible_idx.size == 0:
            return []
        k_eff = max(1, min(int(k), int(feasible_idx.size)))
        urut = feasible_idx[np.argsort(bids_row[feasible_idx], kind="stable")]
        return [int(i) for i in urut[:k_eff]]

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, k: int = 3, critic_obs_np=None,
            epsilon: float = 0.0, threshold: float = 0.20, pref_hist_np=None, **_abaikan):
        """`epsilon`/`threshold` DIABAIKAN -- eksplorasi berasal dari pencuplikan Gaussian
        atas bid, bukan epsilon-greedy. Keduanya tetap diterima demi keseragaman
        pemanggilan di rollout.py.

        Return dict spt HPPOPolicy.act DITAMBAH `bids` (N,): aksi sesungguhnya yang harus
        disimpan di transisi, karena rasio PPO dihitung atas bid.
        """
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        critic_obs = (torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
                      if critic_obs_np is not None else None)
        pref_hist = (torch.as_tensor(pref_hist_np, dtype=torch.float32).unsqueeze(0)
                     if pref_hist_np is not None else None)

        bid_mean, value = self._fwd(obs, hist, critic_obs, pref_hist)
        dist = self._bid_dist(bid_mean)
        bids = dist.sample()

        mask_np = np.asarray(feasible_mask_np, dtype=bool)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool).unsqueeze(0)
        # log-prob HANYA atas stasiun feasible: bid stasiun tak-feasible tak mempengaruhi
        # hasil, memasukkannya ke rasio PPO hanya menyuntikkan derau ke gradien.
        logp_total = float((dist.log_prob(bids) * mask_t.float()).sum().item())

        bids_np = bids[0].cpu().numpy().astype("float32")
        chosen_order = self._pilih_terendah(np.where(mask_np, bids_np, BID_INF), mask_np, k)
        return {
            "chosen_indices": chosen_order,
            "n_rec": len(chosen_order),
            "logp": logp_total,
            "value": value[0].detach().cpu().numpy().astype("float64"),
            "bids": bids_np,
        }

    # ------------------------------------------------------------------ update PPO
    def evaluate(self, obs_b, mask_b, chosen_indices_b, n_rec_b, hist_b, critic_obs_b=None,
                 bids_b=None, pref_hist_b=None, **_abaikan):
        """`chosen_indices_b`/`n_rec_b` TIDAK dipakai -- dan itu memang benar. Seluruh
        stokastisitas ada pada bid; seleksi k-terendah DETERMINISTIK begitu bid diketahui.
        Rasio PPO karenanya dihitung atas bid: himpunan terpilih yang sama dapat berasal
        dari bid yang sangat berbeda."""
        if bids_b is None:
            raise ValueError(
                "kebijakan bidding butuh `bids_b` (aksi sesungguhnya). "
                "Cek PPOTrainer.update -- jalur `use_bids` tidak aktif.")
        bid_mean, value = self._fwd(obs_b, hist_b, critic_obs_b, pref_hist_b)
        dist = self._bid_dist(bid_mean)
        m = mask_b.float()
        return (dist.log_prob(bids_b) * m).sum(dim=-1), (dist.entropy() * m).sum(dim=-1), value


class MasterBiddingPolicy(_BiddingMixin, MasterPolicy):
    """Agen = STASIUN. Tanpa riwayat per-pengguna.

    ⚠️ KOREKSI (audit 2026-08-19, temuan #15b). Versi sebelumnya menulis "setia pada
    kondisi informasi MASTER". Itu TERLALU LUAS: `_build_obs` tetap menyiarkan `soc`,
    koordinat pengguna, dan `battery_norm` ke tiap stasiun (`rollout.py`), padahal
    observasi stasiun MASTER (§3.1) hanya memuat indeks stasiun, waktu, slot tersedia,
    permintaan mendatang, daya, ETA, dan CP -- tanpa SoC, tanpa kapasitas baterai, tanpa
    koordinat mentah. Kesetiaan yang benar-benar berlaku hanya untuk RIWAYAT.

    Nama & parameter DIPERTAHANKAN persis (`bid_log_std`) supaya checkpoint yang sudah
    dilatih (2026-08-19) tetap dapat dimuat.
    """


class BiddingHistPolicy(_BiddingMixin, HPPOPolicy):
    """Agen = PERMINTAAN. Bentuk aksi bidding + `hist_lstm` (riwayat pengguna).

    Pembanding yang benar adalah `MasterBiddingPolicy`: bentuk aksi, reward, kritik, dan
    lingkungan identik -- yang berbeda HANYA apakah agen mengondisikan pada riwayat
    pengguna yang sedang meminta, yaitu apakah agennya permintaan atau stasiun.
    """


class MasterBiddingPrefPolicy(_BiddingMixin, MasterPrefPolicy):
    """**MASTER + preference** -- bidding tanpa riwayat, DITAMBAH modul preferensi PDQN.

    Pembanding yang benar adalah `MasterBiddingPolicy`: bentuk aksi, kritik, reward, dan
    lingkungan identik; satu-satunya pertambahan adalah `pref_lstm` + `PreferenceAttention`
    + `pref_gate`. Satu faktor, satu jawaban.

    HIPOTESIS YANG DIUJI
    --------------------
    Premis (seluruhnya lolos audit kesetiaan 2026-08-19):

      M-a  bid stasiun hanya bergantung observasi stasiun, `a^i_t = b^i(o^i_t)`
           (MASTER Pers. 11) -- identitas peminta masuk hanya lewat ETA
      M-b  penolak pergi ke stasiun ground-truth; MCWT/MCP/CFR dihitung HANYA atas yang
           menerima (MASTER Lampiran A) -> penolak di LUAR fungsi objektif
      P-a  preferensi dapat diekstraksi dari riwayat pasangan (a_hat, a) (PDQN §III-D)
      P-b  kinerja PDQN MENURUN seiring naiknya ketidakpatuhan `p_check` (PDQN §V-B)

    Penalaran: dari M-a, dua pengguna dengan keadaan stasiun sama menerima bid sama. Dari
    M-b, yang menolak tak memberi tekanan gradien apa pun. Di bawah kepatuhan EKSOGEN
    keduanya tak berbiaya -- probabilitas penerimaan tetap. Di bawah kepatuhan ENDOGEN
    keduanya jadi cacat: kebijakan tak dapat menjangkau populasi yang gagal dipersuasinya,
    dan populasi itu dibentuk riwayat kebijakan itu sendiri (P-b menunjukkan persoalan ini
    nyata di papernya sendiri). Dari P-a, riwayat (a_hat, a) menandai SIAPA yang meminta
    dalam istilah preferensi. Karena itu mengondisikan bid pada riwayat tersebut memulihkan
    kanal yang hilang.

    PREDIKSI
      (a) tingkat penerimaan NAIK dibanding `MasterBiddingPolicy`
      (b) keuntungannya LEBIH BESAR pada rezim ketidakpatuhan MENENGAH -- meniru bentuk
          PDQN §V-B Gbr. 10b (selisih terbesar pada p_check = 0,6). Prediksi (b) meramalkan
          BENTUK KURVA, bukan sekadar arah, sehingga sulit dipenuhi secara kebetulan.
      pemalsu: penerimaan tak naik, ATAU naik seragam tanpa bergantung tingkat kepatuhan.

    ⚠️ BATAS KLAIM. Klaimnya adalah "bid buta terhadap RIWAYAT PREFERENSI", BUKAN "buta
    terhadap pengguna" -- aktor tetap melihat SoC/koordinat/baterai (lihat koreksi di
    `MasterBiddingPolicy`). Langkah penalaran tetap sah karena SoC dan lokasi bukan
    preferensi. Penyimpangan lain dari paper: k=2 pemenang (MASTER k=1, §3.1 `argmax`),
    bid terendah menang (MASTER tertinggi -- tak substantif), tanpa penyaringan top-50,
    objektif pemerataan, horizon continuing, PPO bukan DDPG, tanpa replay, kritik `V(s)`.

    ⚠️ BUKTI AWAL CONDONG MELAWAN: pada lengan seleksi-himpunan, modul preferensi
    MENURUNKAN penerimaan (0,608->0,561 abs; 0,748->0,700 signed). Bila pola itu berulang
    di bidding, hipotesis ditolak -- dan itu hasil yang tetap layak dilaporkan.
    """


class BiddingPrefPolicy(_BiddingMixin, PPPOPolicy):
    """Agen = PERMINTAAN, + modul preferensi PDQN. Formula: MASTER-bid + preference."""
