# Metode yang Dibandingkan — Rincian Lengkap

Dokumen acuan untuk menulis bab metodologi. Isinya: apa **persis** yang sama dan apa yang
berbeda di antara ketiga lengan eksperimen, sampai ke tingkat komponen jaringan, suku
imbalan, dan hiperparameter.

Semua angka di sini dibaca langsung dari kode, bukan dari catatan. Sumbernya disebutkan
di tiap bagian supaya bisa diperiksa ulang.

---

## 1. Ringkasan: rancangan 2×2 atas dua teknik

| Lengan | Teknik preferensi | Pemisahan penilai | Mewakili |
|---|---|---|---|
| `h1a_pemerataan` | — | ✓ | Kemampuan koordinasi (turunan MASTER) |
| `h2a_selera` | ✓ | — | Kemampuan preferensi (turunan PDQN) |
| `h6b_utama` | ✓ | ✓ | Keduanya — **yang diuji** |
| `greedy_queue` | — | — | Heuristik antrean terpendek |
| `greedy_util` | — | — | Heuristik utilisasi terendah |

Sel keempat rancangan 2×2 (tanpa kedua teknik) **tidak dijalankan**; kedua `greedy`
mengisi peran lantai. Lihat §8.

---

## 2. Kenapa ablasi, bukan implementasi asli MASTER dan PDQN

Ketiga lengan RL **bukan tiga sistem berbeda**, melainkan satu sistem dengan komponen
dinyalakan dan dimatikan.

Menjalankan kode asli kedua paper akan mencampur dua hal yang tak bisa dipisahkan lagi:

| Paper | Asumsi domainnya | Kalau dijalankan apa adanya di sini |
|---|---|---|
| MASTER (Zhang dkk., 2021) | Armada tertutup, rekomendasi **pasti** dipatuhi | Kepatuhan sukarela membatalkan asumsi intinya |
| PDQN (Lin dkk., 2024) | Kepatuhan **stasioner** dalam episode | Kepercayaan yang bergerak membatalkan asumsinya |

Selisih yang teramati akan bercampur antara "beda arsitektur" dan "beda domain sekaligus
beda implementasi". Dengan ablasi pada basis kode yang sama, satu-satunya hal yang berbeda
adalah komponen yang sedang diteliti — dan itulah yang membuat atribusi mungkin.

---

## 3. Yang IDENTIK di ketiga lengan RL

### 3.1 Algoritma pembelajaran

| | Nilai | Berlaku di |
|---|---|---|
| Algoritma | PPO (*clipped surrogate*) | ketiganya |
| Kelas pelatih | `MasterEVPPOTrainer` | ketiganya |
| Unit agen | **Satu permintaan pengisian EV** (bukan stasiun) | ketiganya |
| Skema | CTDE — kritik terpusat, aktor terdesentralisasi | ketiganya |
| Panjang *rollout* | 96 langkah per pembaruan | ketiganya |
| Jumlah pembaruan | 300 | ketiganya |
| Penggabung aliran (`beta_mode`) | `fixed` — bobot seragam $1/K$, **DGR tidak aktif** | ketiganya |

Catatan penting: `beta_mode="fixed"` berarti **Dynamic Gradient Re-weighting milik MASTER
tidak dipakai**. Bobot antar-aliran tetap seragam sepanjang pelatihan. Ini harus disebut
di bab metodologi — mekanisme DGR ada di kode tetapi sengaja tidak diaktifkan.

Hiperparameter PPO (`PPOTrainer`, baku, identik ketiganya):

| | Nilai |
|---|---|
| Laju belajar | $3 \times 10^{-4}$ |
| Rasio kliping | 0,2 |
| Epoch per pembaruan | 10 |
| Ukuran *minibatch* | 64 |
| Koefisien entropi | 0,01 |
| Koefisien nilai | 0,5 |
| Batas norma gradien | 0,5 |
| Target KL | 0,03 |
| GAE $\lambda$ | 0,95 |
| **Formulasi** | **imbalan rerata** (`avg_reward=True`) |
| Batas jarak langkah GAE | 4 |

**Formulasi imbalan rerata**, bukan berdiskon. Karena masalahnya adalah tugas berkelanjutan
(populasi pengguna terus datang, tak ada batas episode alami), $\gamma = 0{,}99$ tidak
dipakai; sebagai gantinya $\delta_t = (r_t - \bar{r}) + V(s_{t+1}) - V(s_t)$.

`max_step_gap=4` memutus rantai *bootstrap* GAE bila dua transisi berurutan terpisah lebih
dari 4 langkah waktu. Ini perbaikan atas cacat yang pernah ditemukan: imbalan tertunda
(sampai 3 jam sesudah keputusan) membuat GAE mem-*bootstrap* melintasi transisi yang tak
berdekatan waktu.

*Sumber: `Eksekusi_Hipotesis/1_eksperimen.py` (BERSAMA), `marl_spklu/rl/ppo.py::PPOTrainer`,
`marl_spklu/rl/master_ev_ppo_policy.py`.*

### 3.2 Observasi aktor

Vektor fitur per stasiun kandidat, **10 dimensi** (`STATION_FEAT_DIM_MASTER_EV`):

- 7 dimensi §3.1 MASTER: indeks stasiun, waktu (sin/cos), slot tersedia, permintaan
  mendatang, daya, ETA
- 3 dimensi keadaan pemohon EV: jarak, SoC, kapasitas baterai

Aktor **tidak pernah** melihat `user.trust` mentah — konsisten dengan §3.1 MASTER
(desentralisasi, agen buta terhadap atribut internal pemohon).

*Sumber: `marl_spklu/rl/master_paper_obs.py`.*

### 3.3 Kritik

| | Nilai |
|---|---|
| Bentuk | $V(s)$ "buta-aksi" |
| Penggabung | `AttentionPooling` — invarian permutasi |
| Penyandi | `StationEncoder` berbobot-terbagi (pola *Deep Sets*) |
| Lebar tersembunyi | 128 |

Penyandi berbobot-terbagi ini yang menjawab pelanggaran invariansi permutasi PDQN
(`Linear(128,N)` datar). **Ketiga lengan memakainya** — jadi perbaikan itu bukan variabel
yang dibandingkan, melainkan dasar bersama.

### 3.4 Lingkungan dan pengukuran

| | Nilai |
|---|---|
| Dataset | `scenario_dataset_klaster12_4x.json` (30 hari) / `..._4x_90d.json` (90 hari) |
| Penaksir waktu tunggu | `VirtualWaitForecaster` (`sim.compute_virtual_wait`) |
| Jumlah rekomendasi ($k$) | 3 |
| Aturan kepercayaan | `TRUST_PENALTY_MODE = "signed"` — hanya keterlambatan menghukum |
| Model pilihan pengguna | $P_i(j) = T_i \cdot P_{rec}(j) + (1-T_i) \cdot P_{pref}(j)$ |
| Kepercayaan awal | 0,3 / 0,5 / 0,7 (disilangkan) |
| Seed pelatihan | 5 |
| Seed evaluasi | 10 |
| Model yang dievaluasi | seed ber-Gini **median** |

`VirtualWaitForecaster` memakai `remaining_time` sungguhan dari EV yang sedang mengisi —
informasi yang tak dimiliki sistem nyata. Ini **tidak membiaskan perbandingan antar-lengan**
(semua memakainya, termasuk kedua `greedy`), tetapi membatasi klaim validitas eksternal.
Sudah dicatat sebagai batas cakupan di `draft tesis/Hipotesis_Penelitian.md`.

### 3.5 Preset imbalan

Ketiganya memakai `RewardCalculator.seimbang4x()` — preset yang dikalibrasi untuk rezim
beban 4×:

| Parameter | Nilai | Arti |
|---|---|---|
| `alpha_wait` | 0,0046 | bobot perbaikan waktu tunggu |
| `alpha_gini` | 2,6019 | bobot pemerataan |
| `alpha_flock` | 0,0208 | bobot anti-penumpukan |
| `beta_prox` | 0,1 | bobot kecocokan fitur |
| `use_delta_gini` | `True` | memakai **perubahan** Gini, bukan level absolutnya |

Kalibrasi ini menyamakan simpangan baku kanal individual dengan kanal global (~0,15),
supaya objektif pemerataan benar-benar hadir di gradien. Verifikasi pada rezim 4×: varians
individual 55,9%, kontribusi Gini dalam kanal global 39,7% (preset lama: 0,1%).

*Sumber: `marl_spklu/rl/rewards.py::seimbang4x`.*

### 3.6 Penyandi riwayat interaksi — DIMATIKAN di ketiganya

`hist_lstm` (LSTM atas 4 fitur × 10 langkah terakhir: patuh, `estwait` ternormalisasi,
`wait_default` ternormalisasi, galat terealisasi) **dimatikan** lewat `--no-hist` di
**semua** lengan.

Modulnya tetap ada di jaringan; hanya kontribusinya yang dinolkan, sehingga bentuk jaringan
dan kompatibilitas model tak berubah.

> **Ini pernah salah dan sudah diperbaiki.** Bawaan kelas adalah `use_hist=True`. Pada
> versi awal, `--no-hist` tertinggal di `h1a_pemerataan`, sehingga lengan itu diam-diam
> memperoleh proxy kepercayaan yang tak dimiliki lengan lain — tanpa galat, tanpa
> peringatan. Sekarang dijaga `periksa_kesepadanan()` di `1_eksperimen.py`.

Konsekuensinya untuk tesis: **H1 (penaksiran kepercayaan dari riwayat) tidak diuji.**
Tujuan Fungsional §1.3 butir pertama perlu direvisi.

---

## 4. Yang BERBEDA — komponen jaringan

| Komponen | `h6b_utama` | `h1a_pemerataan` | `h2a_selera` |
|---|---|---|---|
| Kelas kebijakan | `MasterEVPPOPrefPolicy` | `MasterEVPPOPolicy` | `MasterEVPPOPrefPolicy` |
| `station_encoder` | ✓ | ✓ | ✓ |
| `disc_head` | ✓ | ✓ | ✓ |
| Kritik ber-atensi | ✓ | ✓ | ✓ |
| `hist_lstm` | dimatikan | dimatikan | dimatikan |
| **`pref_lstm`** | ✓ | **—** | ✓ |
| **`pref_attn`** | ✓ | **—** | ✓ |
| **`pref_gate`** | ✓ | **—** | ✓ |
| Kepala kritik | **3 keluaran** | **3 keluaran** | **1 keluaran** |

### 4.1 Modul preferensi (ada di `h6b_utama` dan `h2a_selera`)

Tiga bagian, mengikuti PDQN (Lin dkk., 2024):

| Bagian | Isi |
|---|---|
| `pref_lstm` | LSTM atas riwayat pasangan (rekomendasi, pilihan nyata), 10 langkah, dimensi tersembunyi 16 |
| `pref_attn` | `PreferenceAttention` — mencocokkan vektor preferensi ke tiap stasiun kandidat, dimensi 16 |
| `pref_gate` | Skalar tunggal, **diinisialisasi nol** |

**`pref_gate` berinisialisasi nol** (pola GTrXL, Parisotto dkk. 2019): kontribusi modul
preferensi masuk secara bertahap, bukan langsung penuh. Modul yang baru diinisialisasi acak
kalau disuntikkan penuh akan jadi derau besar bagi `station_encoder` yang sedianya belajar
lancar tanpanya.

**`--pref-feature-mode` aktif**: riwayat preferensi disandikan sebagai **vektor fitur
stasiun**, bukan *one-hot* identitas stasiun. Ini membuat modul bisa bergeneralisasi ke
stasiun yang belum pernah dilihat.

Akibat pada dimensi konteks `station_encoder`:

| | Dimensi konteks |
|---|---|
| Tanpa modul preferensi | 16 (`hist_hidden` saja) |
| Dengan modul preferensi | 16 + 16 = **32** |

Kritik **juga** menerima `attended_pref`, bukan hanya aktor. Alasannya: kalau kritik buta
terhadap sinyal yang mendorong aksi aktor, $V(s)$ tak bisa "menjelaskan" bagian nilai itu,
sehingga estimasi *advantage* jadi bias. `attended_pref` bukan informasi istimewa — aktor
sendiri sudah melihatnya — sehingga tetap konsisten dengan CTDE.

*Sumber: `marl_spklu/rl/master_ev_ppo_policy.py::MasterEVPPOPrefPolicy`.*

### 4.2 Pemisahan penilai (ada di `h6b_utama` dan `h1a_pemerataan`)

Kepala kritik mengeluarkan **3 nilai**, bukan 1. Tiap aliran diregresikan ke sasarannya
sendiri, dan keuntungan dinormalisasi **per-aliran** sebelum digabung.

| Aliran | Isi | Sifat |
|---|---|---|
| `STREAM_WAIT` (0) | perbaikan waktu tunggu | tertunda sampai pengisian selesai |
| `STREAM_PROX` (1) | kecocokan fitur stasiun | segera, tiap keputusan |
| `STREAM_GLOBAL3` (2) | Gini + anti-penumpukan + kepatuhan | campuran |

Dengan `--n-critics 1` (`h2a_selera`), ketiga aliran **dijumlahkan jadi satu angka** dan
diregresikan ke satu sasaran. Inilah bentuk yang sama dengan dekomposisi aditif PDQN, dan
inilah yang diteliti: apakah menjumlahkan tujuan berbeda skala terlalu dini membuat sinyal
yang jarang tenggelam.

**Total imbalannya identik.** Yang berbeda hanya apakah ia diregresikan terpisah atau
digabung. Jadi perbandingan `h6b` vs `h2a` benar-benar mengisolasi pemisahan penilai, bukan
besar imbalan.

*Sumber: `MasterEV3Transition.reward_vec`, `MasterEVPPORolloutAgent._split_prox`.*

---

## 5. Yang BERBEDA — suku imbalan

| Suku | Kapan diberikan | `h6b` | `h1a` | `h2a` |
|---|---|---|---|---|
| `wait_reward` | pengisian selesai (tertunda) | ✓ | ✓ | ✓ |
| `decision_reward` (Prox) | tiap keputusan (segera) | ✓ | ✓ | ✓ |
| `gini_reward` | tiap langkah simulasi | ✓ | ✓ | ✓ |
| `flock_reward_rolling` | tiap keputusan | ✓ | ✓ | ✓ |
| **`acceptance_reward`** | tiap keputusan (segera) | **✓** | **—** | **✓** |
| `trust_shaping_reward` | — | — | — | — |
| `local_equity_reward` | — | — | — | — |

`acceptance_reward` = $\alpha_{accept} \cdot (+1 \text{ patuh} / -1 \text{ tolak})$, dengan
$\alpha_{accept} = 1{,}0$. **Simetris** — menghukum penolakan, bukan hanya mengganjar
kepatuhan — dan **segera**, berbeda dari `wait_reward` yang tertunda.

### 5.1 Kenapa `acceptance_reward` bagian dari teknik preferensi

Keputusan 2026-08-23. Modul preferensi menyediakan **representasi** (siapa pengguna ini,
apa yang ia sukai); `acceptance_reward` menyediakan **objektifnya** (apakah pencocokan itu
benar-benar berbuah kepatuhan). Tanpa yang kedua, modul preferensi belajar mencocokkan
selera tanpa pernah memperoleh sinyal apakah pencocokan itu berguna.

Akibat metodologisnya menentukan — kedua perbandingan menjadi **satu faktor**:

| Perbandingan | Yang berbeda |
|---|---|
| `h6b` vs `h1a` (Gini) | teknik preferensi (modul + objektif) |
| `h6b` vs `h2a` (penerimaan) | pemisahan penilai |

Pada versi sebelumnya `acceptance_reward` hanya ada di `h6b`, sehingga `h6b` berbeda dari
tiap induk dalam **dua** hal dan komponen penyebab kemenangan tak dapat disimpulkan.

### 5.2 Ke aliran mana `acceptance_reward` masuk

| | Aliran |
|---|---|
| `h6b_utama` (3 penilai) | `STREAM_GLOBAL3` |
| `h2a_selera` (1 penilai) | `STREAM_INDIVIDUAL` — lalu dijumlahkan |

Terlihat berbeda, tetapi **total imbalannya sama**: dengan satu penilai semua aliran
dijumlahkan, jadi penempatannya tak berpengaruh.

Kenapa `STREAM_GLOBAL3` dan bukan `STREAM_WAIT`: diagnosis 21 Agustus 2026 menemukan suku
bermagnitudo $\pm 1$ yang datang **segera** setiap keputusan, kalau ditumpuk di aliran yang
sama dengan `wait_reward` yang kecil dan tertunda, membuat rerata `STREAM_WAIT` meledak
10–30× dibanding `STREAM_PROX`. Diduga penyebab dominan waktu tunggu dan Gini yang liar
pada percobaan 90 hari sebelumnya.

*Sumber: `master_ev_ppo_policy.py::on_decision` baris ~460, `rollout.py` baris ~440.*

### 5.3 Asumsi yang datanya akan terlihat sendiri

Kedudukan `acceptance_reward` sebagai bagian teknik preferensi sah **bila** erosi penerimaan
memang khas preferensi.

`h1a_pemerataan` sengaja tetap dijalankan tanpanya. Jadi periksa kolom **Terima** pada
`h1a` saat hasil keluar:

| Yang teramati | Artinya |
|---|---|
| Penerimaan `h1a` wajar | Framing terkonfirmasi — suku itu memang milik teknik preferensi |
| Penerimaan `h1a` tergerus parah | Suku itu perbaikan **umum**, bukan khas preferensi — catat sebagai keterbatasan |

Tak perlu eksperimen tambahan; datanya sudah ikut terkumpul.

---

## 6. Tabel pengaturan lengkap

| Pengaturan | `h6b_utama` | `h1a_pemerataan` | `h2a_selera` |
|---|---|---|---|
| `--pref` | ✓ | — | ✓ |
| `--pref-feature-mode` | ✓ | — | ✓ |
| `--alpha-accept` | **1.0** | 0.0 | **1.0** |
| `--n-critics` | **3** | **3** | **1** |
| `--no-hist` | ✓ | ✓ | ✓ |
| `--forecaster` | vwf | vwf | vwf |
| `--reward-preset` | seimbang4x | seimbang4x | seimbang4x |
| `--dataset` | 4x | 4x | 4x |
| `--k` | 3 | 3 | 3 |
| `--beta-mode` | fixed | fixed | fixed |
| `--rollout-steps` | 96 | 96 | 96 |
| `--n-updates` | 300 | 300 | 300 |
| `--n-train-seed` | 5 | 5 | 5 |
| `--n-eval-seed` | 10 | 10 | 10 |
| `--initial-trust` | 0,3 / 0,5 / 0,7 | 0,3 / 0,5 / 0,7 | 0,3 / 0,5 / 0,7 |

**Ringkasnya**: dari 14 pengaturan, hanya **3** yang berbeda — dan ketiganya adalah
komponen yang memang sedang diteliti.

---

## 7. Baseline greedy

Dua heuristik tanpa pembelajaran, dihitung ulang di **tiap** lengan pada dataset dan seed
evaluasi yang sama.

| | Aturan |
|---|---|
| `greedy_queue` | Pilih stasiun dengan antrean terpendek |
| `greedy_util` | Pilih stasiun dengan utilisasi terendah |

Keduanya memakai `VirtualWaitForecaster` yang sama dengan lengan RL — sehingga keunggulan
informasi penaksir itu **tidak** menjadi konfound yang membedakan lengan.

Greedy bukan lawan lemah di penelitian ini. Pada eksperimen sebelumnya greedy **menang** di
waktu tunggu dan kepercayaan; MARL hanya unggul di koordinasi.

---

## 8. Keterbatasan rancangan — untuk ditulis apa adanya

**8.1 Sel keempat 2×2 tidak dijalankan.** Versi tanpa teknik preferensi *dan* tanpa
pemisahan penilai. Sel itu akan menunjukkan apakah pemisahan penilai sendirian berbuah
apa-apa. Kedua `greedy` mengisi peran lantai, tetapi bukan sel yang setara. Ini bagian dari
H4 yang dikeluarkan dari cakupan.

**8.2 DGR tidak aktif.** `beta_mode="fixed"` — bobot antar-aliran tetap seragam. Mekanisme
*Dynamic Gradient Re-weighting* milik MASTER ada di kode tetapi sengaja tidak dipakai. Jadi
"pemisahan penilai" di sini berarti **kepala kritik terpisah dengan bobot tetap**, bukan
DGR penuh.

**8.3 H1 tidak diuji.** `hist_lstm` dimatikan di semua lengan, sehingga penaksiran
kepercayaan dari riwayat interaksi tak pernah diuji. Tujuan Fungsional §1.3 butir pertama
perlu direvisi.

**8.4 Penaksir waktu tunggu memakai informasi oracle.** `VirtualWaitForecaster` membaca
`remaining_time` sungguhan EV yang sedang mengisi. Tidak membiaskan perbandingan
antar-lengan, tetapi membatasi klaim validitas eksternal.

**8.5 Akurasi janji memburuk saat rekomendasi menumpuk.** Terdiagnosis (rho = 0,36; rasio
hingga 5,8×), tiga percobaan perbaikan gagal, dikeluarkan dari cakupan sebagai penelitian
lanjutan.

---

## 9. Rujukan silang

| Butir | Berkas |
|---|---|
| Definisi lengan & pemeriksa kesepadanan | `Eksekusi_Hipotesis/1_eksperimen.py` |
| Kelas kebijakan & aliran imbalan | `marl_spklu/rl/master_ev_ppo_policy.py` |
| Suku imbalan & preset | `marl_spklu/rl/rewards.py` |
| Observasi §3.1 | `marl_spklu/rl/master_paper_obs.py` |
| Aturan kepercayaan & model pilihan | `marl_spklu/env/user.py` |
| Hipotesis & syarat lulus | `draft tesis/Hipotesis_Penelitian.md` |
| Cara menjalankan | `Eksekusi_Hipotesis/README.md` |
