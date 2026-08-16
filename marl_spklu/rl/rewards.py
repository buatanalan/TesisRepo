"""Reward EV Agent — sesuai Bab III.3 tesis (Reward Function): R_i = R_individual + R_global.

R_individual (komponen lokal per-agen):
    + alpha_wait  * relu(wait_default - wait_aktual) / wait_scale   Positif hanya jika
      rekomendasi yang DITERIMA memperpendek waktu tunggu dibanding stasiun default
      pengguna (dihitung lewat antrean virtual) -- mencegah agen sekadar
      merekomendasikan stasiun favorit pengguna yang padat.
    + beta_prox   * Prox(spklu_rekomendasi, spklu_terpilih)   Melatih agen membaca
      preferensi pengguna lewat kedekatan fitur fisik (lokasi, daya konektor, estimasi
      tunggu) antara SPKLU yang direkomendasikan dan yang benar-benar dipilih. Terdefinisi
      baik saat pengguna menerima maupun menolak rekomendasi.

    v2 (Spesifikasi_Teknis_RL.md): suku kejujuran (alpha_honesty) DIHAPUS bersama a2 --
    EstWait yang ditampilkan sekarang SELALU jujur (murni forecaster, tak lagi keluaran
    kebijakan), jadi honesty_gap tak lagi atribut ke aksi agen sama sekali.

R_global (dinilai oleh critic tersentralisasi, CTDE -- butuh keadaan global):
    - alpha_gini  * Gini(utilisasi antar-SPKLU)   Continuing/average-reward task: menghukum
      level Gini tiap langkah setara meminimalkan Gini rata-rata jangka panjang.
    - alpha_flock * flocking_penalty = (n_same-1)/n_window   Proporsi agen dalam jendela
      keputusan yang merekomendasikan stasiun sama dengan agen i.
"""
import numpy as np


def _gini(a) -> float:
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a)
    n = a.shape[0]
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


class EquityRewardCalculator:
    """Reward PEMERATAAN untuk PDQN baseline (Tahap 0). Menggantikan RewardCalculator yang
    terbukti TIDAK SELARAS dgn tujuan: pada uji keselarasan (jalankan kebijakan aturan-tetap,
    bandingkan reward vs Gini yang dihasilkan), RewardCalculator memberi reward TERTINGGI
    kepada `anti_greedy` -- kebijakan dgn Gini TERBURUK -- dan reward TERENDAH kepada
    `greedy_util` yang Gini-nya TERBAIK; korelasi(reward, Gini) = +0.55 di mu_hat=0.8,
    yakni terbalik dari yang seharusnya.

        R = alpha_rec   * (u_rata2_feasible - u(a_hat))        [suku REKOMENDASI]
          + alpha_shift * (u(default)       - u(dipilih))      [suku PERGESERAN]
          - alpha_flock * recent_recs(a_hat) / rec_activity_scale   [suku ANTI-OVERSHOOT]

    Suku REKOMENDASI ditentukan SEPENUHNYA oleh aksi agen -> sinyal belajar tajam, segera,
    bebas derau kepatuhan (bandingkan -Gini global lama yang disiarkan rata ke semua
    transisi sehingga kontribusi tiap keputusan tak terbedakan). CATATAN PENTING: suku ini
    scr matematis dimaksimalkan TEPAT oleh a_hat=argmin(utilisasi) -- identik dgn aturan
    GreedyAgent(mode="utilization"). Artinya batas atas suku ini SAMA dgn Greedy; PDQN
    hanya bisa MELAMPAUI Greedy lewat suku PERGESERAN atau suku ANTI-OVERSHOOT di bawah.

    Suku PERGESERAN mengukur EFEK NYATA interaksi: seberapa jauh pengguna berpindah dari
    stasiun favoritnya ke stasiun yang lebih sepi. Bernilai 0 bila pengguna menolak
    (dipilih == default), sehingga agen hanya dibayar kalau rekomendasinya benar-benar
    diikuti. (Ablasi §3.9 laporan: suku ini blm terbukti tereksploitasi scr terukur.)

    Suku ANTI-OVERSHOOT (BARU) menghukum rekomendasi ke SPKLU yang BARU SAJA ramai
    direkomendasikan dlm jendela 24 jam berjalan (`sim.recent_recs`, DIBACA SEBELUM
    keputusan ini menambah hitungannya -- lihat PDQNRolloutAgent.get_recommendation).
    Ini menyasar blind spot STRUKTURAL Greedy: Greedy stateless antar-rekomendasi, jadi
    bila beberapa permintaan muncul berdekatan waktu, Greedy mengirim SEMUA ke SPKLU yg
    tampak paling sepi SAAT snapshot diambil -> overshoot (contoh motivasi Fig.1 paper Lin
    et al. 2024: sistem semestinya memprediksi kedatangan pemohon berikutnya). PDQN yang
    melihat blok observasi `rec_activity` (pdqn_agent.py) BISA belajar menghindari ini;
    Greedy TIDAK MUNGKIN, apa pun anggaran/arsitekturnya -- inilah satu-satunya sumber
    keunggulan PDQN yg tak terikat batas atas suku REKOMENDASI di atas.

    Semua suku memakai kondisi TERNORMALISASI yang diukur pada saat keputusan diambil.
    Reward diketahui SEGERA -> transisi langsung resolved, tak ada reward tertunda."""

    def __init__(self, alpha_rec: float = 1.0, alpha_shift: float = 1.0,
                 alpha_flock: float = 0.0, rec_activity_scale: float = 10.0):
        self.alpha_rec = float(alpha_rec)
        self.alpha_shift = float(alpha_shift)
        self.alpha_flock = float(alpha_flock)
        self.rec_activity_scale = float(rec_activity_scale)

    def decision_reward_equity(self, utilizations, feasible_idx, a_hat_idx,
                               default_idx, chosen_idx, recent_rec_count: float = 0.0) -> float:
        u = np.asarray(utilizations, dtype=float)
        idx = list(feasible_idx) if len(feasible_idx) else list(range(len(u)))
        u_mean = float(u[idx].mean())
        r_rec = u_mean - float(u[a_hat_idx])
        r_shift = float(u[default_idx]) - float(u[chosen_idx])
        r_flock = -float(recent_rec_count) / self.rec_activity_scale
        return (self.alpha_rec * r_rec + self.alpha_shift * r_shift
                + self.alpha_flock * r_flock)


class RewardCalculator:
    """R_i = R_individual (wait-improvement + Prox - honesty) + R_global (-Gini - flocking).

    Kalibrasi bobot (Validasi_Generik/../Dokumen_Penting/Rumusan_Masalah_Teknis_RL.md §3.2b,
    Eksekusi_RL Tahap 0.1): std MENTAH (dibagi bobot lama) kelima suku sudah sebanding
    (prox 0,132; delta-gini 0,257; flock 0,132; improvement 0,160; honesty 0,093) -- yang
    timpang sebelumnya adalah MEAN, didominasi `gini` level absolut (mean mentah ~-0,62,
    hampir konstan tiap keputusan, bukan sinyal ttg keputusan tertentu). `use_delta_gini`
    mengubahnya jadi mean~0 (perubahan Gini, bukan level) sehingga BISA diseimbangkan
    berbasis std terhadap suku individual. `prox` TIDAK diikutkan penyeimbangan berbasis
    std (mean-mentahnya ~0,84, jauh lebih besar dari std-nya -> hampir konstan secara
    struktural; menaikkannya lewat penyeimbangan std akan membuat mean-nya meledak) --
    dipertahankan sbg suku minor tetap (`beta_prox` tak berubah lintas preset)."""

    def __init__(self, alpha_wait: float = 1.0, beta_prox: float = 0.1,
                 alpha_gini: float = 0.5,
                 alpha_flock: float = 0.3, prox_lambda: float = 0.1,
                 wait_scale: float = 60.0, use_delta_gini: bool = False):
        self.alpha_wait = float(alpha_wait)
        self.beta_prox = float(beta_prox)
        self.alpha_gini = float(alpha_gini)
        self.alpha_flock = float(alpha_flock)
        self.prox_lambda = float(prox_lambda)
        self.wait_scale = float(wait_scale)
        # Delta-gini (bukan level absolut) -- lihat catatan kelas. `_prev_gini` disimpan
        # per-instance (state internal), direset otomatis di keputusan pertama tiap episode
        # (None -> delta pertama = 0, tak ada sinyal palsu di awal).
        self.use_delta_gini = bool(use_delta_gini)
        self._prev_gini = None

    # ---- Tiga preset terkalibrasi (Tahap 0.1) — lihat rasio target di masing-masing ----
    # RE-KALIBRASI (setelah flocking dipindah ke jendela bergulir, `flock_reward_rolling`):
    # std MENTAH flock naik ~2x lipat (0,132 -> 0,280) drpd definisi per-langkah lama --
    # bobot alpha_flock/alpha_gini di bawah dihitung ulang dari statistik BARU, BUKAN
    # angka lama. std mentah acuan (kebijakan HPPOPolicy belum terlatih, seed 0):
    # gini(delta)=0,288, flock(rolling)=0,280, improvement=0,148, honesty=0,104.
    # v2 RE-KALIBRASI (setelah alpha_honesty dihapus & ruang aksi jadi top-K/threshold
    # deterministik + epsilon-greedy -- lihat Spesifikasi_Teknis_RL.md v2): std MENTAH
    # (alpha=1, kebijakan HPPOPolicy belum terlatih, seed=0, k=2, threshold=0.20,
    # epsilon=0.0) via Eksekusi_RL/common.py::reward_five_way BERUBAH DRASTIS drpd angka
    # v1 -- pemilihan deterministik pada logits nyaris seragam (untrained) sering jatuh
    # ke lantai-1 (softmax 1/6=16,7% < threshold 20%) -> sangat mirip Greedy stateless ->
    # herding jauh lebih besar drpd sampling stokastik v1. std mentah acuan BARU:
    # gini(delta)=0,2593, flock(rolling)=1,1985, improvement=1,6726 (prox tetap kecil,
    # 0,0148, TIDAK diikutkan penyeimbangan std spt v1 -- lihat catatan kelas).
    @classmethod
    def seimbang(cls, **override):
        """Suku individual (wait) & global (gini-delta, flock) ditarget std sama
        (~0,15) -- tak ada kanal yang mendominasi. Titik netral, tanpa preferensi arah."""
        kw = dict(alpha_wait=0.0897, alpha_gini=0.5785, alpha_flock=0.1252,
                  beta_prox=0.1, use_delta_gini=True)
        kw.update(override)
        return cls(**kw)

    @classmethod
    def individual(cls, **override):
        """Kanal individual (wait) ditarget std lebih tinggi (~0,20) drpd global
        (~0,08) -- RL diprioritaskan memperbaiki wait PER pengguna."""
        kw = dict(alpha_wait=0.1196, alpha_gini=0.3085, alpha_flock=0.0667,
                  beta_prox=0.1, use_delta_gini=True)
        kw.update(override)
        return cls(**kw)

    @classmethod
    def kumulatif(cls, **override):
        """Kanal global (gini-delta, flock) ditarget std lebih tinggi (~0,20) drpd individual
        (~0,08) -- RL diprioritaskan memperbaiki pemerataan sistem & anti-herding, bahkan
        bila itu berarti individu tertentu tak selalu mendapat wait terbaik."""
        kw = dict(alpha_wait=0.0478, alpha_gini=0.7713, alpha_flock=0.1669,
                  beta_prox=0.1, use_delta_gini=True)
        kw.update(override)
        return cls(**kw)

    # ------------------------------------------------------------------------------
    # REKALIBRASI REZIM 4x (2026-08-16) -- preset di atas dikalibrasi pada std mentah
    # gini(delta)=0,2593, flock=1,1985, improvement=1,6726. Pada rezim operasi 4x yang
    # DIBEKUKAN Tahap 1, std mentah terukur SANGAT BERBEDA:
    #     gini(delta)=0,0408   flock=5,1087   improvement=32,4005   prox=0,1579
    # (diukur `Eksekusi_RL/_kalibrasi_reward_4x.py`, HPPOPolicy belum terlatih, seed=0).
    #
    # Akibatnya bobot lama TIDAK LAGI menyeimbangkan apa pun di rezim 4x:
    #   * `seimbang()`  -> varians individual 95,2%; kontribusi gini dlm kanal global 0,1%
    #   * `gabungan`    -> varians individual  2,6%; kontribusi gini dlm kanal global 0,1%
    # Yakni: SUKU GINI -- objektif pemerataan itu sendiri -- praktis tak menghasilkan
    # gradien di kedua konfigurasi. Yang dipelajari agen sesungguhnya adalah anti-herding
    # (flock) & wait. Ini menjelaskan kenapa RL sulit mengungguli greedy pada Gini.
    #
    # CATATAN PENTING: preset lama TIDAK cacat rancangan -- pada rezim kalibrasinya ia
    # justru seimbang sempurna (0,5785x0,2593 = 0,1252x1,1985 = 0,150). Yang terjadi adalah
    # Tahap 1 membekukan rezim 4x TANPA mengulang kalibrasi Tahap 0. `seimbang4x()` di
    # bawah MEMULIHKAN maksud desain asli pada rezim yang benar, bukan mengubah filosofi.
    @classmethod
    def seimbang4x(cls, **override):
        """Versi `seimbang()` yang dikalibrasi ulang untuk REZIM OPERASI 4x (Tahap 1).

        Target sama dgn aslinya: std kanal individual ~= std kanal global (~0,15), DAN
        di dalam kanal global std gini ~= std flock -- sehingga objektif pemerataan
        benar-benar hadir di gradien, bukan dekoratif.

        Verifikasi pada rezim 4x: varians individual 55,9%, kontribusi gini dlm kanal
        global 39,7% (vs 0,1% pada preset lama).

        `beta_prox` tetap 0,1 (suku minor): prox struktural hampir konstan (rasio
        mean/std 6,53) -- menaikkannya hanya menambah offset, bukan sinyal. Ini pelajaran
        dari `gabungan` yang menaikkannya 4,5x dan mematikan kanal individualnya."""
        kw = dict(alpha_wait=0.0046, alpha_gini=2.6019, alpha_flock=0.0208,
                  beta_prox=0.1, use_delta_gini=True)
        kw.update(override)
        return cls(**kw)

    # ---- Prox: kedekatan fitur fisik SPKLU rekomendasi vs SPKLU benar-benar dipilih ----
    def prox(self, feat_rec, feat_chosen) -> float:
        d = float(np.linalg.norm(np.asarray(feat_rec, dtype=float) - np.asarray(feat_chosen, dtype=float)))
        return float(np.exp(-self.prox_lambda * d))

    # ---- Suku individual dievaluasi saat pengguna memutuskan (on_decision, segera) ----
    def decision_reward(self, prox_value: float) -> float:
        return self.beta_prox * prox_value

    # ---- Suku individual dievaluasi saat sesi pengisian selesai (delayed) ----
    # v2: `disp_estwait` dipertahankan di signature (dipanggil tr.disp_estwait dari
    # rollout.py) tapi TAK LAGI dipakai -- EstWait yang ditampilkan sekarang selalu jujur
    # (murni forecaster), jadi tak ada lagi honesty_gap yang bisa diatribusikan ke aksi agen.
    def wait_reward(self, wait_default: float, wait_actual: float, disp_estwait: float = 0.0) -> float:
        improvement = max(0.0, float(wait_default) - float(wait_actual)) / self.wait_scale
        return self.alpha_wait * improvement

    # ---- Penalti herding / flocking -- DEPRECATED (per-langkah/same-step). Aktif hanya
    # 10,1% transisi krn 75% langkah cuma punya 1 keputusan -- herding di sistem ini
    # bersifat TEMPORAL (permintaan berdekatan WAKTU, bukan simultan-per-langkah), jadi
    # jendela per-langkah salah tangkap fenomena. Dipertahankan utk kompatibilitas mundur
    # (mis. pemanggil lama/pengujian arsip), TAK LAGI dipakai jalur produksi
    # (RLRolloutAgent, lihat rollout.py::on_decision -> flock_reward_rolling di bawah).
    @staticmethod
    def flocking_penalty(n_same: int, n_window: int) -> float:
        if n_window <= 1:
            return 0.0
        return (n_same - 1) / n_window

    # ---- Penalti herding / flocking -- JENDELA BERGULIR (produksi, Tahap 0.1) ----
    # Dihitung per KEPUTUSAN (bukan per langkah) dari `sim.recent_recs[sid]` -- hitungan
    # berapa kali SPKLU ini direkomendasikan dlm jendela 24 jam BERJALAN (reset tiap 96
    # langkah, lihat Simulator.step_once), DIBACA SEBELUM keputusan ini menambah
    # hitungannya. Sama sumber data & semantik dgn suku anti-overshoot
    # `EquityRewardCalculator.decision_reward_equity` (rec_activity_scale).
    @staticmethod
    def flocking_penalty_rolling(recent_rec_count: float, scale: float = 10.0) -> float:
        return float(recent_rec_count) / float(scale)

    def flock_reward_rolling(self, recent_rec_count: float, scale: float = 10.0) -> float:
        return -self.alpha_flock * self.flocking_penalty_rolling(recent_rec_count, scale)

    # ---- Suku Gini (CTDE), per-langkah transisi dibuat: -Gini(-delta) SAJA ----
    # Dipisah dari flocking (dulu digabung di `global_reward`) krn flocking kini dihitung
    # di titik siklus-hidup BERBEDA (on_decision, bukan on_step_end) -- lihat rollout.py.
    def gini_reward(self, utilizations) -> float:
        gini_u = _gini(utilizations)
        if self.use_delta_gini:
            signal = 0.0 if self._prev_gini is None else (gini_u - self._prev_gini)
            self._prev_gini = gini_u
        else:
            signal = gini_u
        return -self.alpha_gini * signal

    def reset_episode_state(self) -> None:
        """Bersihkan state lintas-keputusan saat SIMULASI diganti (batas horizon).

        WAJIB dipanggil setiap `_fresh_sim()`. `_prev_gini` bersifat lintas-keputusan:
        tanpa reset, keputusan PERTAMA di simulasi BARU menghitung
        `gini(sim_baru, langkah 1) - gini(sim_lama, hari ke-30)` -- delta antara dua
        simulasi yang TAK BERHUBUNGAN.

        BUG TERUKUR (2026-08-16): sinyal palsu itu bermagnitudo ~0,3-0,5 sementara
        delta-Gini normal ber-std ~0,041; dikali `alpha_gini` ia jadi ~10x sinyal
        sebenarnya, DISIARKAN ke semua transisi langkah itu. Akibatnya kolaps entropi
        yang berulang: dari 23 onset kolaps terukur lintas seluruh run Tahap 2,
        **16 terjadi tepat pada iterasi pertama setelah batas horizon** (harapan acak
        2,3). Ini juga menjelaskan kenapa kalibrasi `ent_coef` tak menolong -- regularisasi
        entropi tak bisa melawan sinyal reward palsu."""
        self._prev_gini = None

    # ---- Komponen global (CTDE) LAMA -- DEPRECATED, gabungan gini+flocking per-langkah.
    # Dipertahankan utk kompatibilitas mundur; jalur produksi kini memanggil
    # `gini_reward` (on_step_end) & `flock_reward_rolling` (on_decision) terpisah.
    def global_reward(self, utilizations, n_same: int = 0, n_window: int = 0) -> float:
        gini_term = self.gini_reward(utilizations)
        penalty = self.flocking_penalty(n_same, n_window)
        return gini_term - self.alpha_flock * penalty

    def step_reward(self, utilizations, n_same: int = 0, n_window: int = 0) -> float:
        return self.global_reward(utilizations, n_same, n_window)
