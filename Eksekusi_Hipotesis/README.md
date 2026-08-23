# Eksekusi Hipotesis — Rencana Eksperimen

Pengujian klaim utama tesis. Terpisah dari `Eksekusi_RL/` yang berisi eksplorasi dan
pencarian arsitektur.

**Bedanya**: `Eksekusi_RL/` tempat mencari, folder ini tempat **membuktikan**. Semua yang
dijalankan di sini harus memenuhi syarat lulus yang ditetapkan di muka.

---

## Yang diuji

Hanya **dua klaim**. Cakupan sengaja dipersempit (keputusan 2026-08-23) — rencana
sebelumnya memuat delapan eksperimen untuk enam hipotesis, dan itu tidak proporsional
untuk satu tesis.

| Klaim | Isi |
|---|---|
| **H6b** | Penyatuan (pemerataan + selera + penyeimbang) mengungguli **kedua** sistem asal yang dijalankan sendiri-sendiri, dan mengungguli *greedy* |
| **H2b** | Keunggulan itu **menyusut** seiring turunnya kepercayaan awal populasi |

H6b adalah klaim utama. H2b yang membuat tesis ini benar-benar tentang non-kepatuhan —
tanpanya, dinamika kepercayaan cuma jadi latar dan tak pernah diuji.

Keduanya dijawab oleh **kumpulan hasil yang sama**. H2b tidak menambah pelatihan apa pun,
hanya membaca hasil H6b dari sudut berbeda.

## Yang sengaja tidak diuji

| Dikeluarkan | Alasan | Jadi apa di tesis |
|---|---|---|
| Aturan kepercayaan asimetris | Sudah jadi bawaan kode (`TRUST_PENALTY_MODE = "signed"`) dan sudah tervalidasi 4/4 sel di lini lama | Dikutip, bukan diuji ulang |
| Penaksir kepercayaan dari riwayat | Butuh ablasi baru + 10 unit | Penelitian lanjutan; **Tujuan Fungsional §1.3 perlu direvisi** |
| Sel keempat 2×2 (tanpa keduanya) | Bagian dari H4 | Penelitian lanjutan; `greedy` mengisi peran lantai |
| Performativitas | Ada bukti dari lini lama (beban 1× vs 4×) | Dikutip sebagai temuan pendukung |

### Dua konsekuensi yang harus diterima

**1. Klaim "imbalan kepatuhan menyelamatkan penyatuan naif" gugur.** Setelah suku itu
dimasukkan ke dalam paket teknik preferensi, ia ada di *baseline* juga — sehingga tak bisa
lagi diklaim sebagai penstabil khusus penyatuan. Ini pertukaran yang menguntungkan: klaim
lama belum pernah terverifikasi, dan sebagai gantinya kedua perbandingan jadi bisa
diatribusikan.

**2. Tujuan Fungsional §1.3 memuat butir yang tidak lagi diuji** — *implicit trust
encoder* berbasis LSTM. Revisi §1.3 supaya tujuannya sesuai dengan yang benar-benar
dibuktikan, atau butir itu akan jadi lubang yang ditanyakan penguji.

---

## Aturan main

**Penamaan bertanggal.** Format `<tag>__it<kepercayaan>[_90d]_<tanggal>_eval.json`.
Kejadian 21–23 Agustus (model utama tertimpa karena nama file dipakai ulang) tidak boleh
terulang. Tanggal di nama membuat penimpaan diam-diam mustahil.

**Syarat lulus ditulis sebelum hasil dilihat.** Kalau perlu diubah, ubah sebelumnya dan
catat alasannya.

**Hasil negatif tetap disimpan dan dilaporkan.**

---

## Cara menjalankan

Empat langkah, berurutan.

```bash
cd Eksekusi_Hipotesis

python 0_asap.py                        # 1. periksa pemasangan   (~1 menit)
python 1_eksperimen.py --lihat          # 2. lihat rencananya
python 1_eksperimen.py                  # 3. jalankan             (~9 unit)
python 2_analisis.py                    # 4. vonis                (~detik)
```

Lalu uji ketahanan:

```bash
python 1_eksperimen.py --horizon 90d
python 2_analisis.py --horizon 90d
python 2_analisis.py --banding          # 30 vs 90 hari berdampingan
```

Di server, jalankan yang lama di latar belakang:

```bash
nohup python 1_eksperimen.py > logs/eksperimen_$(date +%Y%m%d).log 2>&1 &
```

**Tahan-putus.** Lengan yang berkasnya sudah ada akan dilewati. Kalau terputus, jalankan
ulang perintah yang sama.

**Memeriksa rangkaian dulu:** `--cepat` memakai 1 seed × 3 pembaruan, cukup untuk
memastikan kesembilan lengan berurutan tanpa galat. Hasilnya **tidak sah dilaporkan** dan
skripnya mencetak peringatan itu.

---

## Daftar skrip

| Skrip | Isi |
|---|---|
| `0_asap.py` | Memastikan pengaturan kepercayaan benar berpengaruh |
| `1_eksperimen.py` | Menjalankan 9 lengan (3 arsitektur × 3 tingkat kepercayaan) |
| `2_analisis.py` | Menerapkan syarat lulus, mencetak vonis |
| `_pipeline_hipotesis.py` | Mesin latih+evaluasi (dipanggil `1_eksperimen.py`) |
| `_kakas.py` | Perkakas bersama (metrik, format tabel, pemuat hasil) |

Berawalan `_` = pustaka, tidak dijalankan langsung.

---

## Langkah 1 — Periksa pemasangan

```bash
python 0_asap.py
```

**Sudah dijalankan 23 Agustus 2026: LULUS 15/15.**

| Yang diperiksa | Hasil |
|---|---|
| `initial_trust` 0,3 / 0,5 / 0,7 jadi nilai awal | tepat sampai 4 desimal |
| Dinamika tetap hidup di bawah `initial_trust` | pergeseran 0,33 |
| `constant_trust` benar membekukan | pergeseran 0,00 |
| `constant_trust_shadow` tetap bergerak | pergeseran 0,34 |
| Konteks dilepas bersih | kembali ke 0,500 |

Uji ini memeriksa **pengaruhnya**, bukan sekadar "jalan tanpa galat". Pipeline yang
menerima pengaturan lalu mengabaikannya akan lolos uji galat tetapi membuat seluruh
eksperimen tak berarti — kegagalan senyap yang paling mahal, karena baru ketahuan
setelah berhari-hari komputasi.

---

## Langkah 2 — Jalankan eksperimen

```bash
python 1_eksperimen.py
```

Sembilan lengan: tiga arsitektur × tiga tingkat kepercayaan awal.

### Rancangan 2×2 atas dua teknik

| | Teknik preferensi | Penilai terpisah | Mewakili |
|---|---|---|---|
| `h1a_pemerataan` | mati | ✓ | kemampuan koordinasi (MASTER) |
| `h2a_selera` | ✓ | mati | kemampuan preferensi (PDQN) |
| `h6b_utama` | ✓ | ✓ | keduanya — yang diuji |

**Teknik preferensi adalah satu paket**: modul selera (`--pref --pref-feature-mode`)
**ditambah** imbalan kepatuhan (`--alpha-accept 1.0`). Modul menyediakan *representasi*
(siapa pengguna ini, apa yang ia suka); imbalan kepatuhan menyediakan *objektifnya*
(apakah pencocokan itu benar-benar berbuah kepatuhan). Tanpa yang kedua, modul preferensi
belajar mencocokkan selera tanpa pernah diberi tahu apakah pencocokan itu berguna.

Karena paketnya utuh di kedua lengan yang memilikinya, **kedua perbandingan menjadi satu
faktor**:

| Perbandingan | Yang berbeda |
|---|---|
| `h6b` vs `h1a` — Gini | teknik preferensi |
| `h6b` vs `h2a` — penerimaan | pemisahan penilai |

Ini yang membuat selisihnya bisa diatribusikan. *(Keputusan 2026-08-23. Sebelumnya
`--alpha-accept` hanya dipasang di `h6b`, sehingga `h6b` berbeda dari tiap induk dalam dua
hal sekaligus dan penyebab kemenangan tak dapat disimpulkan.)*

**Pengaturan lengkap:**

| Lengan | Pengaturan khas |
|---|---|
| `h6b_utama` | `--pref --pref-feature-mode --alpha-accept 1.0 --no-hist --n-critics 3` |
| `h1a_pemerataan` | `--no-hist --n-critics 3` |
| `h2a_selera` | `--pref --pref-feature-mode --alpha-accept 1.0 --no-hist --n-critics 1` |

*Baseline* `greedy` (queue & util) dihitung otomatis di tiap lengan.

**Kenapa ketiga lengan harus dijalankan.** Klaimnya adalah penyatuan mengungguli **kedua**
sistem asal. Kalau hanya dibandingkan dengan satu, klaimnya bisa dijatuhkan dengan satu
kalimat: *"mungkin yang satunya juga menang, jadi penyatuannya tidak menambah apa-apa."*

Ketiganya juga bukan tiga sistem berbeda, melainkan **satu sistem dengan komponen
dinyalakan dan dimatikan** — penyandi, pelatih, penaksir, preset imbalan, dataset, dan
anggaran semuanya identik. Menjalankan kode asli MASTER dan PDQN justru akan mencampur
beda arsitektur dengan beda domain (MASTER mengasumsikan kepatuhan penuh, PDQN
mengasumsikan kepatuhan stasioner), sehingga selisihnya tak bisa diatribusikan.

**Pemeriksa kesepadanan.** `1_eksperimen.py` menjalankan `periksa_kesepadanan()` sebelum
apa pun dimulai, menjaga empat syarat: `--no-hist` ada di semua lengan, paket preferensi
tak pernah pecah, tepat satu lengan tanpa teknik preferensi, tepat satu lengan berpenilai
gabungan. Diuji dengan empat sabotase sengaja; semuanya tertangkap. Skripnya berhenti
sebelum membuang waktu komputasi, bukan setelah.

### Satu asumsi yang datanya akan terlihat sendiri

Framing "imbalan kepatuhan itu bagian dari teknik preferensi" benar **bila** erosi
penerimaan memang khas preferensi. `h1a_pemerataan` sengaja tetap dijalankan tanpa imbalan
kepatuhan — jadi kalau penerimaannya ternyata **ikut tergerus parah**, berarti suku itu
perbaikan umum, bukan milik teknik preferensi, dan framing ini harus dicatat sebagai
keterbatasan. Periksa kolom "Terima" pada `h1a` saat hasilnya keluar.

**Pengaturan bersama dikunci di kode**, bukan disalin ke tiap baris perintah:
`--forecaster vwf`, `--reward-preset seimbang4x`, `--n-eval-seed 10`, `--dataset 4x`.
Menyalin perintah dengan tangan adalah cara termudah membuat dua lengan tak sepadan tanpa
disadari — dan itu sudah pernah terjadi (`K3_gap` tak terpakai karena dua hal berubah
bersamaan).

**Keluaran:** satu berkas per lengan di `outputs/`, memuat empat metrik inti, sebaran
kepercayaan akhir, Wilcoxon dan Cohen's *d* terhadap kedua *greedy*.

---

## Langkah 3 — Vonis

```bash
python 2_analisis.py
```

Tidak melatih apa pun, hanya membaca `outputs/`. Aman diulang.

### Syarat lulus H6b

Ketiganya harus terpenuhi, minimal di kepercayaan **0,5 dan 0,7**:

| | Syarat |
|---|---|
| Gini | ≤ Gini `h1a_pemerataan` (toleransi 2%) |
| Penerimaan | ≥ 95% capaian `h2a_selera` |
| vs *greedy* | Wilcoxon *p* < 0,05 **dan** Gini lebih rendah |

### Syarat lulus H2b

Besar efek (Cohen's *d*) terhadap `greedy_util` mengecil berurutan dari kepercayaan
0,7 → 0,5 → 0,3. Yang diuji **arahnya**, bukan angka persisnya.

### Kalau gagal — sudah disiapkan jawabannya

**H6b gagal sebagian.** Kalau penyatuan tetap menang di penerimaan dan kepercayaan tapi
kalah di Gini, turunkan klaimnya jadi *"mempertahankan pemerataan sambil menaikkan
penerimaan"*. Klaim lebih lemah tapi sah. Jangan dipaksakan.

Ini bukan kemungkinan teoretis: lengan pemerataan-saja di lini lama (`eq1`) mencapai Gini
0,066 dengan sebaran antar-seed 0,026 — lebih rendah **dan** lebih stabil daripada
eksperimen utama (0,078; sebaran 0,049). Protokolnya berbeda sehingga belum sah
dibandingkan, tapi **ada kemungkinan nyata penyatuan kalah di Gini.**

**H2b gagal (kurvanya datar).** Ini temuan, bukan kegagalan teknis. Artinya masalah
non-kepatuhan tidak sepenting yang diklaim di latar belakang, dan pembingkaian tesis perlu
disesuaikan. Laporkan apa adanya.

---

## Langkah 4 — Ketahanan

```bash
python 1_eksperimen.py --horizon 90d
python 2_analisis.py --horizon 90d
python 2_analisis.py --banding
```

**Peringatan serius dari data yang sudah ada.** Satu-satunya hasil 90 hari yang tersedia
(`master_ev_ppo_pref_feat_nohist_acc1_vwf_K3_gap_sig1_90d`) **kalah telak**:

| | Gini |
|---|---|
| Penyatuan, 90 hari | **0,108** |
| greedy_queue | 0,089 |
| greedy_util | 0,098 |

Cohen's *d* = **+2,10** terhadap greedy_util — arah positif berarti lebih buruk. Sebaran
antar-seed juga liar: 0,192 / 0,083 / 0,108.

Konfigurasinya tidak persis sama (versi itu memakai DGR, yang utama tidak), jadi belum
vonis final. Tapi **antisipasi bahwa langkah ini bisa gagal.**

Kalau keunggulan hanya bertahan 30 hari dan hilang di 90 hari, itu batas berlaku yang
**wajib ditulis terang-terangan**. Melaporkan hasil 30 hari saja, sementara data 90 hari
ada dan menunjukkan sebaliknya, adalah persoalan integritas — dan penguji yang menemukannya
sendiri jauh lebih merugikan daripada melaporkannya lebih dulu.

---

## Menyimpan hasil ke git

`outputs/` sebagian besar diabaikan git. Yang ikut terlacak otomatis:

- `*_eval.json` — ringkasan hasil, puluhan KB
- `analisis_*.json` — vonis

Yang **tidak**: `*.pt` (model) dan `*_training_results.json` (memuat riwayat
per-pembaruan; versi lama sampai 31.000 baris). Model dibiarkan di server.

---

## Tabel lacak

| Langkah | Status | Hasil | Tanggal |
|---|---|---|---|
| 0 — periksa pemasangan | **SELESAI** | lulus 15/15 | 2026-08-23 |
| 1 — eksperimen 30 hari | **SELESAI** | 9/9 lengan, konfigurasi terverifikasi sepadan | 2026-08-23 |
| 2 — vonis 30 hari | **SELESAI** | **H6b DITOLAK**, H2b terdukung | 2026-08-23 |
| 3 — eksperimen 90 hari | belum | — | — |
| 4 — vonis ketahanan | belum | — | — |

### Hasil 30 hari — ringkas

**H6b ditolak.** Koordinasi-saja mengungguli penyatuan pada Gini di ketiga tingkat
kepercayaan (0,079 / 0,057 / 0,071 vs 0,139 / 0,103 / 0,080), seluruhnya signifikan.
Modul preferensi merusak pemerataan.

**Namun penyatuan unggul di segalanya yang lain**, seluruhnya signifikan: kepatuhan
+6 sampai +13 poin, waktu tunggu turun hingga **133 menit**, kepercayaan naik.

**H2b terdukung.** Ukuran efek bertanda terhadap *greedy* menyeberang nol seiring turunnya
kepercayaan awal: −1,90 (unggul) → +1,82 → +5,60 (kalah telak).

**Klaim revisi bersifat PASCA-HOC** — dirumuskan setelah data terlihat, jadi bukan
hipotesis yang lulus. Uji 90 hari adalah konfirmasi pertamanya.

Rincian lengkap: `draft tesis/Hipotesis_Penelitian.md` §HASIL, dan
`outputs/analisis_20260823.json`.
