# Metode yang Dibandingkan — Rincian Lengkap

Dokumen acuan untuk menulis bab metodologi. Isinya: apa **persis** yang sama dan apa yang
berbeda di antara lengan-lengan eksperimen, sampai ke tingkat arsitektur jaringan, aliran
data, komponen, suku imbalan, dan hiperparameter.

**Jawaban ringkas untuk tiga pertanyaan yang sering muncul:**

| Pertanyaan | Jawaban |
|---|---|
| Apakah observasinya berbeda? | **Tidak.** Identik `(N, 10)` di seluruh lengan — lihat §3A.1 |
| Lalu apa yang berbeda? | Masukan **tambahan** (`pref_hist`), dimensi konteks, dan jumlah keluaran kritik — §3A.5 |
| Bagaimana alur datanya? | Diagram per lengan di §3A.3 dan §3A.4 |

Semua angka di sini dibaca langsung dari kode, bukan dari catatan. Sumbernya disebutkan
di tiap bagian supaya bisa diperiksa ulang.

---

## 1. Ringkasan: rancangan 2×2 atas dua teknik

| Lengan | Teknik preferensi | Pemisahan penilai | DGR | Mewakili |
|---|---|---|---|---|
| `h1a_pemerataan` | — | ✓ | — | Koordinasi (turunan MASTER, tanpa DGR) |
| `h2a_selera` | ✓ | — | — | Kemampuan preferensi (turunan PDQN) |
| `h6b_utama` | ✓ | ✓ | — | Keduanya — **yang diuji** |
| `h1a_pemerataan_dgr` | — | ✓ | **✓** | Koordinasi MASTER yang **setia** |
| `greedy_queue` | — | — | — | Heuristik antrean terpendek |
| `greedy_util` | — | — | — | Heuristik utilisasi terendah |

Tiga lengan pertama membentuk rancangan 2×2 pokok. Lengan keempat ditambahkan 2026-08-23
untuk menutup soal kesetiaan pada MASTER — lihat §4.3.

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
| Penggabung aliran (`beta_mode`) | `fixed` — bobot seragam $1/K$ | ketiga lengan pokok |

Catatan penting: `beta_mode="fixed"` pada ketiga lengan pokok berarti **Dynamic Gradient
Re-weighting milik MASTER tidak dipakai di sana**. Bobot antar-aliran tetap seragam
sepanjang pelatihan. Lengan `h1a_pemerataan_dgr` (§4.3) mengaktifkannya sebagai pembanding.

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

## 3A. Arsitektur & aliran data

### 3A.1 Observasi — IDENTIK di ketiga lengan

**Tidak ada perbedaan observasi sama sekali.** Ketiga lengan menerima matriks yang sama
persis dari `build_joint_obs_master_ev`:

```
station_feats : (N, 10)     N = jumlah stasiun kandidat layak
```

Tiap baris = satu stasiun kandidat, 10 kolom:

| Kolom | Isi | Asal |
|---|---|---|
| 1–7 | indeks stasiun, waktu sin, waktu cos, slot tersedia, permintaan mendatang, daya, ETA | §3.1 MASTER |
| 8–10 | jarak pemohon, SoC, kapasitas baterai | keadaan EV pemohon |

Perhatikan kolom 8–10: **keadaan pemohon disisipkan ke tiap baris kandidat**, bukan
diletakkan di blok skalar terpisah (`scalar_dim = 0`). Konsekuensinya, aktor tetap
mengetahui siapa yang sedang dilayani tanpa merusak invariansi permutasi atas stasiun.

Yang **tidak pernah** masuk observasi di lengan mana pun: `user.trust` mentah, dan
identitas pengguna.

### 3A.2 Yang berbeda bukan observasinya, melainkan MASUKAN TAMBAHAN

| Masukan | Bentuk | `h6b` | `h1a` | `h2a` |
|---|---|---|---|---|
| `station_feats` | (N, 10) | ✓ | ✓ | ✓ |
| `hist` | (10, 4) | ✓ tapi **dinolkan** | ✓ tapi **dinolkan** | ✓ tapi **dinolkan** |
| `pref_hist` | (10, 10) | ✓ | **—** | ✓ |

`hist` = riwayat interaksi 10 langkah × 4 fitur (patuh, `estwait` ternormalisasi,
`wait_default` ternormalisasi, galat terealisasi). Tetap dibangun dan diteruskan, tetapi
`--no-hist` membuat `_encode_hist` mengembalikan **vektor nol** tanpa menyentuh LSTM-nya.

`pref_hist` = riwayat 10 pasangan (rekomendasi, pilihan nyata), masing-masing disandikan
sebagai `feat(a_hat) ++ feat(a)` — dua vektor deskriptif stasiun 5-dimensi (jarak, wait,
antrean, konektor, utilisasi), bukan *one-hot* identitas. Karena itu modul preferensi
belajar dari **karakteristik** stasiun, bukan menghafal indeksnya. Padding nol diletakkan
**di belakang**, syarat `pack_padded_sequence` agar LSTM melewati langkah kosong
sepenuhnya.

### 3A.3 Aliran data — `h1a_pemerataan` (koordinasi saja)

```
station_feats (N,10) ─────────────────────────────┬──────────────────────┐
                                                  │                      │
hist (10,4) ──[ _encode_hist ]──► c_t (16)        │                      │
                 use_hist=False                   │                      │
                 → SELALU NOL                     │                      │
                        │                         │                      │
                        └── context (16, nol) ────┤                      │
                                                  ▼                      ▼
                                     StationEncoder(hidden=64)   StationEncoder(hidden=128)
                                     bobot TERBAGI antar-stasiun        [kritik]
                                                  │                      │
                                                  ▼                      ▼
                                            emb (N,64)             c_emb (N,128)
                                                  │                      │
                                          disc_head Linear(64,1)   AttentionPooling
                                                  │                      │
                                                  ▼                      ▼
                                            logits (N)             pooled (128)
                                                  │                      │
                                    mask → softmax → top-K/threshold      ▼
                                                  │              critic_head MLP
                                                  ▼                      │
                                          rekomendasi k=3                ▼
                                                                    V (3 nilai)
```

**Catatan penting**: karena `use_hist=False` **dan** tak ada modul preferensi, `context`
pada lengan ini adalah **16 dimensi bernilai nol seluruhnya**. Jaringannya secara efektif
menjadi MLP murni per-stasiun tanpa konteks per-pengguna dari riwayat — informasi pemohon
hanya masuk lewat kolom 8–10 tiap baris kandidat.

### 3A.4 Aliran data — `h6b_utama` dan `h2a_selera` (dengan teknik preferensi)

```
station_feats (N,10) ──────────────────────┬───────────────────┬──────────────┐
                                           │                   │              │
hist (10,4) ──[ _encode_hist ]──► c_t (16, SELALU NOL)         │              │
                                           │                   │              │
pref_hist (10,10)                          │                   │              │
     │                                     │                   │              │
     ▼ pack_padded_sequence                │                   │              │
  pref_lstm (LSTM 10→16)                   │                   │              │
     │                                     │                   │              │
     ▼ c_pref (16)                         │                   │              │
     └──────────► PreferenceAttention ◄────┘                   │              │
                   q = W_q·c_pref                              │              │
                   k,v = W_kv·station_feats                    │              │
                   w = softmax(q·kᵀ/√d)                        │              │
                   attended = Σ w·v          (16)              │              │
                          │                                    │              │
                          ▼                                    │              │
                  × pref_gate  (skalar, init 0)                │              │
                          │                                    │              │
        context = [ c_t(16, nol) ‖ attended_pref(16) ]  (32)    │              │
                          │                                    │              │
                          ├────────────────────────────────────┤              │
                          ▼                                    ▼              ▼
                  StationEncoder(64)                  StationEncoder(128)  [kritik]
                          │                                    │
                          ▼                                    ▼
                    logits (N)                          V (3 nilai / 1 nilai)
```

**`pref_gate` diinisialisasi nol**, sehingga pada awal pelatihan `attended_pref` = 0 dan
jaringannya berperilaku identik dengan lengan koordinasi. Kontribusi modul preferensi
masuk **bertahap** seiring gerbang belajar membuka.

**`PreferenceAttention` memakai preferensi sebagai *query***, fitur stasiun sebagai
*key/value*. Jadi keluarannya adalah ringkasan stasiun yang **dibobot menurut relevansinya
terhadap selera pengguna yang sedang dilayani** — bukan konkatenasi polos.

### 3A.5 Di mana ketiga lengan benar-benar berpisah

| Titik | `h1a` | `h2a` | `h6b` |
|---|---|---|---|
| Dimensi `context` | 16 (semua nol) | 32 | 32 |
| Cabang preferensi | tidak ada | ada | ada |
| Keluaran `critic_head` | **3** | **1** | **3** |

Selain tiga baris itu, **jalur datanya sama persis** — penyandi yang sama, penggabung
atensi yang sama, kepala diskrit yang sama, aturan pemilihan aksi yang sama.

### 3A.6 Aturan pemilihan aksi — identik ketiganya

1. `logits` dimasker: kandidat tak layak diberi $-\infty$
2. `softmax` atas kandidat layak
3. Ambil semua kandidat berpeluang $> 0{,}20$ (*threshold*); bila tak ada satu pun, ambil
   yang tertinggi
4. Potong pada $k = 3$
5. $\varepsilon$-*greedy*: dengan peluang $\varepsilon$, ambil subhimpunan acak
   (saat evaluasi $\varepsilon = 0$)

Log-peluangnya dihitung sebagai `Categorical` berurutan tanpa pengembalian.

### 3A.7 Invariansi permutasi — dasar bersama, bukan variabel

`StationEncoder` memproses **tiap baris stasiun lewat bobot yang sama** (pola *Deep Sets*),
dan `AttentionPooling` meringkas dengan bobot ber-*softmax* yang selalu berjumlah 1 berapa
pun $N$. Akibatnya menukar urutan stasiun menghasilkan keluaran yang tertukar identik.

Ini yang menjawab pelanggaran invariansi permutasi PDQN (`Linear(128,N)` datar). **Ketiga
lengan memakainya**, jadi perbaikan itu bukan variabel yang dibandingkan melainkan dasar
bersama — dan konsekuensinya tesis ini **tidak** mengukur seberapa besar perbaikan itu
sendiri berkontribusi.

*Sumber: `master_ev_ppo_policy.py::forward/act`, `policy.py::StationEncoder/AttentionPooling`,
`pdqn_policy.py::PreferenceAttention/hist_feat_dim_feature`, `master_paper_obs.py`.*

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

### 4.3 Dynamic Gradient Re-weighting — lengan kesetiaan

Ditambahkan 2026-08-23. `h1a_pemerataan_dgr` identik dengan `h1a_pemerataan` **kecuali**
`--beta-mode gap_ratio --beta-sigma 1.0`.

**Kenapa perlu.** Koordinasi MASTER = kritik terpusat ber-atensi **ditambah** DGR.
`h1a_pemerataan` hanya memuat yang pertama, karena `beta_mode` terbawa sebagai nilai bawaan
pipeline — **bukan keputusan yang diambil sengaja**. Akibatnya lengan yang seharusnya
mewakili kemampuan koordinasi MASTER hanya memuat separuhnya, dan pertanyaan "apakah MASTER
diimplementasikan dengan setia?" tak bisa dijawab ya.

**Cara kerja DGR** (`ppo.py::_compute_beta`): untuk tiap aliran $k$ dihitung selisih relatif
terhadap capaian terbaik yang pernah dilihat, $g^k = (R^{*k} - R^k)/|R^{*k}|$, lalu bobot
$\beta = \mathrm{softmax}(g^k/\sigma)$. Aliran yang paling tertinggal mendapat bobot
terbesar.

**DGR berfungsi, meski sempat diduga tidak.** Dugaan itu mengikuti temuan bahwa normalisasi
*advantage* per-aliran mematikan mekanisme berbasis penskalaan — yang memang membuat
formulasi CMDP tak berdaya. Pemeriksaan `ppo.py` baris 184–190 membantahnya:

```python
adv = (adv - adv.mean(axis=0)) / (adv.std(axis=0) + 1e-8)   # normalisasi per-aliran
beta = self._compute_beta(returns)
adv_combined = adv @ beta                                    # BARU digabung
```

Normalisasi terjadi **lebih dulu**, lalu $\beta$ menggabungkan aliran yang sudah setara —
sehingga $\beta$ benar-benar mengubah arah gradien. Berbeda dari pengali skalar pada
imbalan, yang memang tercuci normalisasi.

**`--beta-sigma 1.0`, bukan bawaan 0,1.** Ada catatan di `ppo.py` bahwa $\sigma = 0{,}1$
terlalu tajam: $g$ diklip pada $[0,10]$ tetapi $z = g/\sigma$ bisa mencapai 100, sehingga
$\beta$ kolaps nyaris *one-hot* antar-*chunk*. Diduga penyebab entropi kebijakan runtuh
permanen pada satu percobaan 90 hari sebelumnya.

**Bukti historis DGR kuat.** Setiap lengan koordinasi dengan `gap_ratio` mengalahkan
*greedy* pada Gini:

| Lengan historis | Gini | Cohen's $d$ vs `greedy_util` |
|---|---|---|
| eq1 K4 `gap_sig1`, 30 hari | 0,0664 | −3,23 |
| eq1 K4 `gap_sig1`, **90 hari** | **0,0330** | **−7,03** |
| eq1 accW1 K5 `gap_sig1` | 0,0744 | −2,59 |

Konfigurasinya tak sepadan dengan lengan baru, jadi ini indikasi, bukan bukti.

**Dua hal yang dijawab lengan ini**: (a) apakah DGR memang menolong — soal kesetiaan pada
paper; (b) apakah penolakan H6b bertahan setelah pembandingnya diperkuat.

Kesepadanan pasangannya dijaga `periksa_kesepadanan()`: keduanya harus identik kecuali
pengaturan `beta`, kalau tidak selisihnya tak dapat diatribusikan ke DGR.

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

| Pengaturan | `h6b_utama` | `h1a_pemerataan` | `h2a_selera` | `h1a_..._dgr` |
|---|---|---|---|---|
| `--pref` | ✓ | — | ✓ | — |
| `--pref-feature-mode` | ✓ | — | ✓ | — |
| `--alpha-accept` | **1.0** | 0.0 | **1.0** | 0.0 |
| `--n-critics` | **3** | **3** | **1** | **3** |
| `--no-hist` | ✓ | ✓ | ✓ | ✓ |
| `--forecaster` | vwf | vwf | vwf | vwf |
| `--reward-preset` | seimbang4x | seimbang4x | seimbang4x | seimbang4x |
| `--dataset` | 4x | 4x | 4x | 4x |
| `--k` | 3 | 3 | 3 | 3 |
| `--beta-mode` | fixed | fixed | fixed | **gap_ratio** |
| `--rollout-steps` | 96 | 96 | 96 | 96 |
| `--n-updates` | 300 | 300 | 300 | 300 |
| `--n-train-seed` | 5 | 5 | 5 | 5 |
| `--n-eval-seed` | 10 | 10 | 10 | 10 |
| `--initial-trust` | 0,3 / 0,5 / 0,7 | 0,3 / 0,5 / 0,7 | 0,3 / 0,5 / 0,7 | 0,3 / 0,5 / 0,7 |

**Ringkasnya**: di antara ketiga lengan pokok, dari 14 pengaturan hanya **3** yang berbeda —
dan ketiganya komponen yang memang sedang diteliti. Lengan DGR berbeda dari
`h1a_pemerataan` hanya pada `--beta-mode` dan `--beta-sigma`.

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

**8.2 DGR diuji terpisah, bukan di ketiga lengan pokok.** Ketiga lengan pokok memakai
`beta_mode="fixed"` — bobot antar-aliran tetap seragam. Nilai itu **terbawa sebagai bawaan
pipeline, bukan keputusan yang diambil sengaja**, dan akibatnya lengan yang mewakili
koordinasi MASTER sempat hanya memuat separuh kontribusi MASTER (kritik ber-atensi ada,
DGR tidak).

Ditutup 2026-08-23 dengan lengan `h1a_pemerataan_dgr` — identik `h1a_pemerataan` kecuali
`--beta-mode gap_ratio --beta-sigma 1.0`. Lihat §4.3.

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
