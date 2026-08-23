# Eksekusi Hipotesis — Rencana Eksperimen

Folder ini khusus untuk pengujian hipotesis H1–H6 di `draft tesis/Hipotesis_Penelitian.md`.
Terpisah dari `Eksekusi_RL/` yang berisi eksplorasi dan pencarian arsitektur.

**Bedanya**: `Eksekusi_RL/` adalah tempat mencari; folder ini adalah tempat **membuktikan**.
Semua yang dijalankan di sini harus memenuhi syarat lulus yang sudah ditetapkan di muka.

---

## Status awal: apa yang sudah dan belum dijawab

| Dugaan | Status | Keterangan |
|---|---|---|
| H1 — kepercayaan ditaksir dari riwayat | **0%** | `--no-hist` menyala di eksperimen utama. Belum pernah diuji |
| H2 — penguncian & kurva menyusut | **0%** di lini ini | Ada bukti lama dari lini H-PPO, beda konfigurasi |
| H3 — rumus kepercayaan | **dipakai, belum diuji ulang** | Aturan asimetris sudah diadopsi, belum dikonfirmasi di lini ini |
| H4 — penilai terpisah vs gabungan | **0%** | Ada data 1-penilai, tapi tercampur beda `alpha_accept` |
| H5 — performativitas | **0%** di lini ini | Ada bukti lama dari perbandingan beban 1x/4x |
| H6a — penyatuan langsung gagal | **cukup** | Banyak varian gagal, sudah terdokumentasi |
| H6b — penyatuan berpenyeimbang berhasil | **sebagian** | 3 seed, 1 tingkat kepercayaan, tanpa uji statistik |

---

## Aturan main folder ini

**1. Penamaan wajib bertanda versi.** Format:

```
h<nomor>_<nama>__it<kepercayaan>_s<seed>_<tanggal>
contoh: h6b_utama__it05_s0_20260824
```

Kejadian 21–23 Agustus (model utama tertimpa karena nama file dipakai ulang) tidak boleh terulang.
Tanggal di akhir nama membuat penimpaan diam-diam mustahil.

**2. Tidak ada percobaan tanpa syarat lulus tertulis lebih dulu.** Tiap eksperimen di bawah sudah
punya syarat lulusnya. Kalau syaratnya perlu diubah, ubah **sebelum** hasilnya dilihat, dan catat alasannya.

**3. Hasil negatif tetap disimpan dan dilaporkan.** Tidak ada percobaan yang dihapus karena hasilnya
tidak diinginkan.

**4. Satu eksperimen = satu file hasil + satu log.** Simpan di `outputs/` dan `logs/`.

---

## E0 — Prasyarat: tambahkan pengaturan tingkat kepercayaan awal

> **Ini penghambat, bukan eksperimen.** Kerjakan lebih dulu atau E2, E4, dan sebagian besar syarat
> pembatalan hipotesis tidak bisa dijalankan sama sekali.

**Masalah.** `Eksekusi_RL/_run_master_ev_ppo_pipeline.py` **tidak punya** pengaturan `--initial-trust`.
Semua eksperimen selama ini berjalan di satu tingkat kepercayaan bawaan. Padahal kepercayaan awal
adalah faktor silang yang wajib di setiap tahap.

**Yang sudah tersedia** di `marl_spklu/experiments/ablations.py` — tidak perlu menulis mekanisme baru,
cukup menyambungkannya ke pipeline:

| Fungsi | Untuk |
|---|---|
| `initial_trust(value)` | H2 — tiga tingkat kepercayaan awal |
| `constant_trust(value)` | H5 — melatih dengan anggapan kepercayaan tetap |
| `no_history_encoder()` | H1 — versi tanpa membaca riwayat |

**Yang perlu dibuat baru:** satu *ablation* untuk H1b — mengacak **riwayat menerima/menolak** saja,
tanpa merusak sinyal selera. Ini belum ada.

**Selesai kalau:** pipeline menerima `--initial-trust 0.3|0.5|0.7` dan `--constant-trust`,
lalu uji asap 5 langkah berjalan tanpa galat di ketiga nilai.

**Biaya:** sangat kecil. Membungkus bagian latih dan evaluasi dengan *context manager* yang sudah ada.

**Skrip:**

| Skrip | Isi | Status |
|---|---|---|
| `_pipeline_hipotesis.py` | Pipeline latih+evaluasi, dengan `--initial-trust` / `--constant-trust` / `--constant-trust-shadow`, penamaan bertanggal, keluaran ke folder ini | **siap** |
| `_asap_e0.py` | Uji asap: memastikan pengaturan kepercayaan benar berpengaruh, bukan cuma diterima | **siap** |
| `_ablasi_acak_riwayat.py` | Untuk H1b di E5 — mengacak riwayat menerima/menolak saja | belum (baru perlu di E5) |

**Perintah — siap disalin:**

```bash
cd Eksekusi_Hipotesis
python _asap_e0.py
```

**Sudah dijalankan 23 Agustus 2026: LULUS 15/15.** Yang diverifikasi:

| Yang diperiksa | Hasil |
|---|---|
| `initial_trust` 0,3 / 0,5 / 0,7 benar jadi nilai awal | tepat sampai 4 desimal |
| Dinamika tetap hidup di bawah `initial_trust` | pergeseran 0,33 |
| `constant_trust` benar membekukan | pergeseran 0,00 |
| `constant_trust_shadow` tetap bergerak | pergeseran 0,34 |
| Konteks dilepas bersih setelah selesai | kembali ke 0,500 |

Uji ini sengaja memeriksa **pengaruhnya**, bukan sekadar "jalan tanpa galat". Pipeline yang
menerima pengaturan lalu mengabaikannya akan lolos uji "tidak galat" tapi membuat seluruh
E2/E4/E7 tak berarti — kegagalan senyap yang paling mahal, karena baru ketahuan setelah
berhari-hari komputasi.

**Uji ujung-ke-ujung juga sudah lulus:**

```bash
python _pipeline_hipotesis.py --tag zz_asap --pref --pref-feature-mode --no-hist \
  --alpha-accept 1.0 --n-critics 3 --forecaster vwf --initial-trust 0.5 \
  --n-train-seed 1 --n-eval-seed 2 --n-updates 2
```

Latih → evaluasi → tabel → simpan, semuanya jalan. Angkanya buruk (baru 2 pembaruan,
kebijakan praktis belum terlatih) dan berkasnya sudah dihapus. Yang diuji rangkaiannya, bukan hasilnya.

---

## Urutan pengerjaan

Satuan biaya: **1 unit = 1 seed pelatihan × 300 pembaruan × 30 hari.**
Dipakai untuk membandingkan besaran antar-eksperimen, bukan sebagai perkiraan jam.

| Urutan | Kode | Menguji | Biaya | Kenapa di urutan ini |
|---|---|---|---|---|
| 1 | **E1** | H3 | 0 unit | Menggerbangi semua tahap lain |
| 2 | **E2** | H6b | 15 unit | Klaim utama tesis, saat ini paling rapuh |
| 3 | **E3** | H6b | 5 unit | Syarat pembatalan #5, tanpa ini mekanismenya tak terbukti |
| 4 | **E4** | H2 | 0 unit tambahan | Menumpang hasil E2 |
| 5 | **E5** | H1 | 10 unit | Tujuan Fungsional #1, saat ini kosong |
| 6 | **E6** | H4 | 10 unit | Landasan penjelasan H6b |
| 7 | **E7** | H5 | 20 unit | Punya bukti lama, paling bisa ditunda |
| 8 | **E8** | ketahanan | 15 unit | Perlu pemenang dari E2 lebih dulu |

**Jalur minimum agar tesis bisa dipertahankan: E1 → E2 → E3 → E4.** Total sekitar 20 unit.
E5–E8 memperkuat, tapi E1–E4 adalah yang membuat klaim utama berdiri.

---

## E1 — Konfirmasi aturan kepercayaan (H3)

**Kenapa pertama.** Kalau rumus kepercayaannya keliru, setiap angka kepercayaan di eksperimen lain
tidak bisa ditafsirkan. Ini gerbang, bukan pelengkap.

**Cara.** Jalankan sistem yang **dibekukan identik** di bawah dua rumus: dua-arah (`|ΔW| > τ`) dan
satu-arah (hanya keterlambatan menghukum). Bandingkan lintasan kepercayaannya sepanjang 90 hari,
di ketiga tingkat kepercayaan awal.

Bagian "dibekukan identik" itu kuncinya — tanpa itu, tidak bisa dibedakan apakah yang berubah
rumusnya atau sistemnya.

**Lulus kalau:** kepercayaan turun terus dengan rumus dua-arah, **dan** berhenti turun dengan rumus
satu-arah, di seluruh kombinasi.

**Kalau gagal:** berarti penurunan kepercayaan itu nyata, bukan artefak. H3 dicoret, dan H2/H6
perlu ditinjau ulang karena keduanya berpijak pada kepercayaan yang bisa distabilkan.

**Catatan.** Aturan asimetris sudah pernah diuji di lini eksperimen lama dan lulus di 4 dari 4 sel.
E1 adalah **konfirmasi ulang di lini ini**, bukan penemuan baru — karena itu biayanya 0 unit
(tidak ada pelatihan, hanya evaluasi kebijakan yang sudah ada).

**Skrip:**

| Skrip | Asal | Perlu diubah? |
|---|---|---|
| `_e1_aturan_trust.py` | Salin dari `Eksekusi_RL/_uji_aturan_trust.py` | Ya — versi lama menargetkan lini H-PPO (`hppo_30d_abs`). Ganti pemuat *checkpoint* ke lini MASTER-EV-PPO |
| `_e1_aturan_trust_eval.py` | Salin dari `Eksekusi_RL/_uji_aturan_trust_eval.py` | Ya — sama, ganti pemuat *checkpoint* |

Logika faktorialnya (abs vs signed, tiap aturan dievaluasi dengan *baseline*-nya sendiri, plus sel
silang) sudah benar dan **jangan diubah** — di situlah kekuatan uji ini.

**Perintah:**

```bash
python _e1_aturan_trust.py       0,1,2,3,4 90d
python _e1_aturan_trust_eval.py  0,1,2,3,4 90d
```

**Peringatan daya uji.** Versi lama mencatat: dengan 3 seed per lengan, uji Mann-Whitney dua-sisi
mentok di $p=0{,}10$ — sesempurna apa pun pemisahannya, $p<0{,}05$ tidak tercapai. **Pakai minimal
5 seed** kalau signifikansi hendak dilaporkan.

**Keluaran:** `outputs/e1_aturan_trust_90d_<tanggal>.json`

---

## E2 — Eksperimen utama pada standar penuh (H6b)

**Kenapa kedua.** Inilah klaim utama tesis, dan saat ini ia berdiri di atas 3 seed, 1 tingkat
kepercayaan, dan tanpa uji statistik. Kalau E2 gagal, seluruh isi tesis berubah — jadi harus
diketahui sedini mungkin.

**Kesenjangan yang ditutup:**

| Syarat | Ditetapkan | Sekarang | Setelah E2 |
|---|---|---|---|
| Seed pelatihan | ≥5 | 3 | 5 |
| Tingkat kepercayaan | 3 | 1 | 3 |
| Seed evaluasi | ≥10 | 10 | 10 |
| Uji statistik | Wilcoxon | tidak ada | ada |

**Skrip:**

| Skrip | Isi |
|---|---|
| `_pipeline_hipotesis.py` | Dari E0. Melatih + evaluasi Gini dengan Wilcoxon |
| `_e2_jalankan.py` | Pembungkus: menjalankan 3 lengan × 3 tingkat kepercayaan berurutan, tahan-putus (lewati yang sudah ada) |
| `_e2_metrik.py` | Salin `Eksekusi_RL/_uji_master_ev_ppo_metrik_full.py`. Metrik kaya (penerimaan/tunggu/kepercayaan) yang tidak dihitung pipeline |
| `_e2_banding.py` | Menggabungkan ketiga lengan + greedy jadi satu tabel, lengkap dengan Wilcoxon dan Cohen's $d$ per tingkat kepercayaan |

**Perintah — siap disalin.** Satu perintah untuk semuanya:

```bash
cd Eksekusi_Hipotesis
python _e2_jalankan.py --lihat     # periksa rencananya dulu
python _e2_jalankan.py             # jalankan (9 lengan E2 + 1 lengan E3)
```

Di server, jalankan di latar belakang:

```bash
nohup python _e2_jalankan.py > logs/e2_$(date +%Y%m%d).log 2>&1 &
```

**Tahan-putus.** Lengan yang berkasnya sudah ada akan dilewati. Kalau terputus di tengah,
jalankan ulang perintah yang sama — yang sudah selesai tidak diulang.

**Pilihan lain:**

```bash
python _e2_jalankan.py --hanya e2          # 3 lengan pokok saja
python _e2_jalankan.py --hanya e3          # lengan atribusi saja
python _e2_jalankan.py --hanya h6b_utama   # satu lengan saja
python _e2_jalankan.py --cepat             # anggaran mini, cek rangkaian utuh
```

`--cepat` memakai 1 seed × 3 pembaruan. Berguna untuk memastikan kesepuluh lengan benar
berurutan tanpa galat sebelum menghabiskan waktu komputasi penuh. **Hasilnya tidak sah dilaporkan**
dan skripnya mencetak peringatan itu.

**Pengaturan bersama dikunci di kode**, bukan disalin ke tiap baris perintah:
`--forecaster vwf`, `--reward-preset seimbang4x`, `--n-eval-seed 10`, `--horizon 30d`,
`--dataset 4x`. Menyalin perintah dengan tangan adalah cara termudah membuat dua lengan
tidak sepadan tanpa disadari.

*Baseline* greedy (util & queue) dihitung otomatis di tiap lengan — tidak perlu dijalankan terpisah.

**Keluaran:** satu berkas per lengan, `outputs/h6b_utama__it03_<tanggal>_eval.json` dst.
Berisi empat metrik inti, sebaran kepercayaan akhir, Wilcoxon dan Cohen's $d$ terhadap kedua greedy.

**Lulus kalau:** Gini penyatuan tidak lebih buruk dari lengan pemerataan-saja, **dan** tingkat
penerimaannya mendekati lengan selera-saja — keduanya nyata lebih baik dari greedy
(Wilcoxon $p<0{,}05$), minimal di kepercayaan 0,5 dan 0,7.

**Peringatan yang sudah terlihat dari data lama.** Lengan pemerataan-saja (`eq1`) mencapai
Gini 0,066 dengan sebaran antar-seed 0,026 — lebih rendah **dan** lebih stabil daripada angka
eksperimen utama (0,078; sebaran 0,049). Keduanya diukur dengan protokol berbeda sehingga belum
sah dibandingkan, tapi **ada kemungkinan nyata E2 menunjukkan penyatuan kalah di Gini** dan hanya
menang di penerimaan serta kepercayaan.

Kalau itu terjadi, klaimnya diturunkan — bukan dipaksakan — menjadi: *penyatuan mempertahankan
pemerataan sambil menaikkan penerimaan dan kepercayaan.* Itu klaim yang lebih lemah tapi tetap sah,
dan sudah diantisipasi di butir interpretasi bersyarat.

---

## E3 — Membuktikan sumber keunggulan (H6b, syarat pembatalan #5)

**Kenapa ketiga.** Tesis mengklaim penyeimbangnya adalah imbalan kepatuhan (`alpha_accept`).
Klaim itu **belum pernah diuji**. Tanpa E3, bagian penjelasan mekanisme di H6b tidak boleh ditulis
sebagai terbukti — hanya angka hasilnya yang boleh dilaporkan.

**Cara.** Satu lengan tambahan, identik dengan E2 kecuali `--alpha-accept 0.0`.
Cukup di kepercayaan 0,5 dengan 5 seed.

**Lulus kalau:** versi tanpa `alpha_accept` **nyata lebih buruk**. Berarti penyeimbangnya terbukti berperan.

**Kalau gagal** (keduanya sama bagus): penjelasan mekanisme dicoret dari tesis. Yang tersisa hanya
"penyatuan ini berhasil", tanpa keterangan kenapa. Klaimnya jadi lebih lemah tapi jujur.

**Catatan.** Sudah ada dua kandidat pembanding di lini lama — `K3_gap` dan `K3_gap_sig1` — yang
sama-sama `pref=True` tanpa `alpha_accept`. Tapi keduanya juga memakai DGR, jadi **ada dua hal
yang berbeda sekaligus** dan hasilnya tidak bisa diatribusikan. E3 harus lengan baru yang bersih.

**Skrip:** tidak ada yang baru — lengan E3 sudah termasuk di `_e2_jalankan.py`.

**Perintah — siap disalin:**

```bash
cd Eksekusi_Hipotesis
python _e2_jalankan.py --hanya e3
```

Sudah ikut jalan kalau Anda menjalankan `_e2_jalankan.py` tanpa pilihan.

**Syarat kesepadanan dijaga otomatis.** `_e2_jalankan.py` punya pemeriksa `periksa_atribusi()`
yang berhenti dengan galat kalau `h6b_tanpa_accept` berbeda dari `h6b_utama` di lebih dari satu
pengaturan. Sudah diuji dengan dua sabotase sengaja (mengubah `--n-critics` diam-diam, dan
menghapus `--no-hist`) — keduanya tertangkap.

Ini bukan kehati-hatian berlebihan: persis kesalahan seperti itulah yang membuat `K3_gap` tidak
bisa dipakai sebagai pembanding, karena DGR ikut berubah bersamaan dengan `alpha_accept`.

**Keluaran:** `outputs/h6b_tanpa_accept__it05_<tanggal>_eval.json`

---

## E4 — Kurva kepercayaan (H2)

**Kenapa keempat, dan kenapa murah.** E2 sudah menjalankan tiga tingkat kepercayaan. E4 tidak
menjalankan apa pun yang baru — ia **menganalisis ulang** hasil E2 dari sudut berbeda.

**Cara.** Dari hasil E2, hitung ukuran efek (Cohen's $d$) keunggulan atas greedy di tiap tingkat
kepercayaan, lalu periksa arahnya. Catat juga sebaran akhir kepercayaan, bukan cuma rata-ratanya.

**Lulus H2a kalau:** sebaran akhir kepercayaan berbeda nyata antar kondisi awal, tanpa konvergensi.
**Lulus H2b kalau:** ukuran efek mengecil berurutan dari kepercayaan 0,7 → 0,5 → 0,3.

**Kalau gagal** (kurvanya datar): ini temuan penting, bukan kegagalan teknis. Artinya masalah
non-kepatuhan tidak sepenting yang diklaim di latar belakang, dan pembingkaian tesis perlu
disesuaikan. Harus dilaporkan apa adanya.

**Skrip yang perlu dibuat:**

| Skrip | Isi |
|---|---|
| `_e4_kurva_trust.py` | Membaca hasil E2, menghitung Cohen's $d$ per tingkat kepercayaan, memeriksa apakah urutannya menyusut. Juga menarik sebaran akhir kepercayaan (bukan rata-rata) untuk H2a |

**Perintah:**

```bash
python _e4_kurva_trust.py       # tanpa pelatihan — hanya membaca outputs/ dari E2
```

**Keluaran:** `outputs/e4_kurva_trust_<tanggal>.json` berisi tabel $d$ per tingkat kepercayaan
dan uji Kolmogorov-Smirnov antar sebaran kepercayaan akhir.

---

## E5 — Membaca riwayat interaksi (H1)

**Kenapa kelima meski penting.** H1 adalah **Tujuan Fungsional butir pertama** (§1.3 — *implicit
trust encoder*) dan saat ini **tidak punya bukti sama sekali**. Alasan ia tidak ditaruh lebih awal
murni praktis: E5 butuh kode ablasi baru, sementara E1–E4 bisa langsung jalan. Tapi jangan
tunda melewati titik ini — tujuan fungsional yang tak terbukti adalah lubang di bab hasil.

**Cara, dua bagian:**

- **H1a** — versi dengan riwayat (`hist_lstm` aktif) vs versi tanpa riwayat (`--no-hist`),
  sisanya identik dengan E2. Kepercayaan 0,5, 5 seed.
- **H1b** — pisahkan sumber keunggulannya lewat dua perusakan terpisah:
  - acak **riwayat menerima/menolak** → merusak sinyal kepercayaan, selera utuh
  - acak **sinyal selera** → merusak selera, riwayat kepatuhan utuh

**Lulus H1a kalau:** versi berriwayat nyata lebih baik.
**Lulus H1b kalau:** perusakan riwayat kepatuhan menghapus keunggulannya, perusakan selera tidak.

**Kalau H1b gagal** (kedua perusakan berakibat sama): yang dipelajari modul itu cuma selera —
sesuatu yang sudah dikerjakan PDQN — sehingga bukan kontribusi baru. Tujuan Fungsional #1
tidak tercapai dan harus dinyatakan begitu.

**Perlu dibuat:** ablasi pengacak riwayat kepatuhan. Belum ada di `ablations.py`.

**Skrip:**

| Skrip | Isi |
|---|---|
| `_ablasi_acak_riwayat.py` | Dari E0. Dua *context manager*: `acak_riwayat_kepatuhan()` dan `acak_sinyal_selera()` |
| `_e5_jalankan.py` | Menjalankan 4 lengan H1a/H1b berurutan |

**Perintah.** H1a dulu — dua lengan, bedanya hanya `--no-hist`:

```bash
# berriwayat
python _pipeline_hipotesis.py --tag h1_berriwayat \
  --pref --pref-feature-mode --alpha-accept 1.0 --n-critics 3 --forecaster vwf \
  --initial-trust 0.5 --n-train-seed 5 --n-eval-seed 10 \
  --n-updates 300 --dataset 4x --horizon 30d

# tanpa riwayat (sama dengan h6b_utama di E2 — pakai ulang, jangan latih lagi)
```

Lalu H1b — dua perusakan terpisah, **memakai ulang model `h1_berriwayat`**, tidak melatih ulang:

```bash
python _e5_ablasi.py --ckpt h1_berriwayat__it05 --rusak riwayat_kepatuhan --n-eval-seed 10
python _e5_ablasi.py --ckpt h1_berriwayat__it05 --rusak sinyal_selera     --n-eval-seed 10
python _e5_banding.py
```

**Kenapa memakai ulang model, bukan melatih ulang.** Perusakan dilakukan **saat evaluasi**, pada
model yang sama persis. Kalau tiap perusakan dilatih ulang, perbedaannya bisa berasal dari
pelatihan yang berbeda, bukan dari sinyal yang dirusak — dan uji H1b jadi tak berarti.

**Keluaran:** `outputs/e5_h1_<tanggal>.json`

---

## E6 — Penilai terpisah vs penilai gabungan (H4)

**Kenapa keenam.** H4 adalah **penjelasan** kenapa H6b berhasil. Kalau E2 dan E3 lulus, H4
memberi alasannya. Kalau E2 gagal, H4 jadi kurang relevan — karena itu ditaruh setelahnya.

**Cara, dua bagian:**
- **H4a** — sapu bobot pemerataan dengan **satu penilai gabungan** (`--n-critics 1`). Catat apakah
  menaikkan bobot benar-benar memperbaiki Gini, atau cuma menggeser skala angkanya.
- **H4b** — ulangi dengan `--n-critics 3`, bobot nominal sama.

**Lulus kalau:** perbaikan Gini nyata lebih besar pada penilai terpisah, pada bobot yang sama.

**Hal yang wajib dicatat, bukan diperlakukan sebagai bug.** Menyetarakan tiap tujuan sebelum
digabung akan membuat mekanisme apa pun yang bekerja lewat **memperbesar/memperkecil** satu tujuan
jadi tak berpengaruh — termasuk pengali Lagrangian pada CMDP. Kalau ini teramati, itu konsekuensi
wajar H4b dan harus ditulis sebagai batas berlakunya. Gejala ini **sudah pernah terlihat** di
eksperimen CMDP sebelumnya.

**Catatan.** Ada data lama `pref_feat_vwf` dengan `n_critics=1`, tapi `alpha_accept`-nya berbeda,
jadi tidak bisa dipakai sebagai pembanding bersih.

**Skrip yang perlu dibuat:**

| Skrip | Isi |
|---|---|
| `_e6_sapuan_bobot.py` | Menjalankan sapuan bobot pemerataan pada `--n-critics 1` lalu `--n-critics 3`, bobot identik |
| `_e6_banding.py` | Memplot Gini terhadap bobot untuk kedua versi. Yang dicari: apakah kurva penilai-terpisah lebih curam |

**Perintah:**

```bash
python _e6_sapuan_bobot.py --bobot 0.5,1.0,2.0,4.0 --n-train-seed 3 --initial-trust 0.5
python _e6_banding.py
```

3 seed cukup di sini — yang diuji adalah **bentuk kurva**, bukan menang/kalah terhadap greedy,
jadi tidak butuh daya statistik sebesar E2.

**Yang wajib dicatat di keluaran** (bukan diperlakukan sebagai kegagalan): apakah bobot masih
berpengaruh setelah penyetaraan per-tujuan. Kalau ternyata tidak, itu konsekuensi wajar H4b dan
harus ditulis sebagai batas berlakunya. `_e6_banding.py` harus melaporkan ini secara eksplisit,
bukan menyembunyikannya di balik "tidak ada efek".

**Keluaran:** `outputs/e6_sapuan_bobot_<tanggal>.json`

---

## E7 — Performativitas (H5)

**Kenapa ketujuh.** Sudah ada bukti dari perbandingan beban 1x/4x di lini lama, dan H5 tidak
menopang klaim utama secara langsung. Paling aman ditunda.

**Cara.** Latih dengan `constant_trust` (kepercayaan dianggap tetap), lalu **pakai** di lingkungan
berkepercayaan dinamis. Bandingkan dengan yang dilatih langsung di lingkungan dinamis.
Ulangi di dua tingkat keramaian: `--dataset 1x` (sepi) dan `--dataset 4x` (padat).

**Lulus kalau:** selisihnya nyata di beban padat.
**Selisih nol di beban sepi bukan kegagalan** — itu justru isi dugaan H5b, dan wajib dilaporkan
sebagai batas berlakunya.

**Batal kalau:** tidak ada selisih di kedua tingkat keramaian.

**Skrip:**

| Skrip | Asal | Perlu diubah? |
|---|---|---|
| `_e7_performativitas.py` | Salin logika dari `Eksekusi_RL/_banding_final.py` | Ya — versi lama membandingkan Tahap 2 vs Tahap 3 lini H-PPO. Ganti pemuat *checkpoint*, pertahankan logika `constant_trust_shadow` |

**Kenapa `constant_trust_shadow`, bukan `constant_trust` biasa.** Versi *shadow* tetap menjalankan
pembaruan kepercayaan tapi membekukan nilai yang **dipakai mengambil keputusan**. Jadi kita bisa
melihat apa yang *akan* terjadi pada kepercayaan seandainya lingkaran umpan baliknya tidak diputus.
Itu diagnostik yang persis dibutuhkan H5 — jangan diganti dengan `constant_trust` polos.

**Perintah:**

```bash
for DS in 1x 4x; do
  # dilatih dengan kepercayaan dianggap tetap
  python _pipeline_hipotesis.py --tag h5_statis_${DS} --constant-trust 0.5 \
    --pref --pref-feature-mode --no-hist --alpha-accept 1.0 --n-critics 3 \
    --forecaster vwf --n-train-seed 5 --n-eval-seed 10 --n-updates 300 \
    --dataset $DS --horizon 30d

  # dilatih langsung di kepercayaan bergerak
  python _pipeline_hipotesis.py --tag h5_dinamis_${DS} --initial-trust 0.5 \
    --pref --pref-feature-mode --no-hist --alpha-accept 1.0 --n-critics 3 \
    --forecaster vwf --n-train-seed 5 --n-eval-seed 10 --n-updates 300 \
    --dataset $DS --horizon 30d
done

python _e7_performativitas.py   # keduanya diuji di lingkungan DINAMIS
```

**Yang menentukan:** kedua model **dievaluasi di lingkungan dinamis**, apa pun cara melatihnya.
Kalau model statis juga diuji di lingkungan statis, tidak ada yang dibuktikan.

**Keluaran:** `outputs/e7_performativitas_<tanggal>.json`

---

## E8 — Ketahanan jangka panjang

**Kenapa terakhir.** Butuh pemenang dari E2 dulu. Tidak ada gunanya menguji ketahanan sistem yang
belum dipastikan menang.

**Cara.** Ulangi konfigurasi pemenang E2 pada 90 hari, ketiga tingkat kepercayaan.

**Peringatan serius dari data yang ada.** Satu-satunya hasil 90 hari yang tersedia
(`master_ev_ppo_pref_feat_nohist_acc1_vwf_K3_gap_sig1_90d`) **kalah telak**:

| | Gini |
|---|---|
| Penyatuan, 90 hari | **0,108** |
| greedy_queue | 0,089 |
| greedy_util | 0,098 |

Cohen's $d$ = **+2,10** terhadap greedy_util — arah positif berarti lebih buruk. Sebaran antar-seed
juga liar: 0,192 / 0,083 / 0,108.

Konfigurasinya tidak persis sama (versi itu memakai DGR, yang utama tidak), jadi belum vonis final.
Tapi **antisipasi bahwa E8 bisa gagal**, dan siapkan pelaporannya: kalau keunggulan hanya bertahan
30 hari dan hilang di 90 hari, itu batas berlaku yang harus ditulis terang-terangan, bukan
disembunyikan dengan hanya melaporkan hasil 30 hari.

**Skrip:** tidak ada yang baru. Pakai `_pipeline_hipotesis.py`, `_e2_metrik.py`, `_e2_banding.py`.

**Perintah.** Perhatikan `--dataset` dan `--horizon` **harus** sama-sama 90 hari:

```bash
for IT in 0.3 0.5 0.7; do
  python _pipeline_hipotesis.py --tag h8_ketahanan90d \
    --pref --pref-feature-mode --no-hist \
    --alpha-accept 1.0 --n-critics 3 --forecaster vwf \
    --initial-trust $IT --n-train-seed 5 --n-eval-seed 10 \
    --n-updates 300 \
    --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d
done

python _e2_banding.py --banding h8_ketahanan90d --lawan h6b_utama
```

**Jebakan yang pernah terjadi.** `--dataset` 90 hari tanpa `--horizon 90d` akan **menimpa
diam-diam** *checkpoint* 30 hari. Pipeline lama sudah memasang peringatan untuk ini; pastikan
`_pipeline_hipotesis.py` mewarisinya.

**Keluaran:** `outputs/e8_ketahanan_<tanggal>.json`

---

## Daftar induk skrip

Total **13 skrip**: 9 dibuat baru, 4 disalin-ubah dari `Eksekusi_RL/`.

| Skrip | Dipakai di | Asal | Status |
|---|---|---|---|
| `_pipeline_hipotesis.py` | E0, E2, E3, E5, E7, E8 | baru, berbasis `_run_master_ev_ppo_pipeline.py` | **siap & teruji** |
| `_asap_e0.py` | E0 | baru | **siap, lulus 15/15** |
| `_e2_jalankan.py` | E2, E3 | baru | **siap & teruji** |
| `_e2_banding.py` | E2, E3, E8 | baru | belum |
| `_e4_kurva_trust.py` | E4 | baru | belum |
| `_e1_aturan_trust.py` | E1 | salin `_uji_aturan_trust.py`, ganti pemuat model | belum |
| `_e1_aturan_trust_eval.py` | E1 | salin `_uji_aturan_trust_eval.py`, ganti pemuat model | belum |
| `_e2_metrik.py` | E2, E8 | salin `_uji_master_ev_ppo_metrik_full.py` | belum, opsional |
| `_ablasi_acak_riwayat.py` | E5 | baru | belum |
| `_e5_ablasi.py`, `_e5_banding.py` | E5 | baru | belum |
| `_e6_sapuan_bobot.py`, `_e6_banding.py` | E6 | baru | belum |
| `_e7_performativitas.py` | E7 | salin logika `_banding_final.py` | belum |

**Yang sudah bisa dijalankan sekarang: E0, E2, dan E3** — itu bagian termahal dari jalur minimum.
Sisa jalur minimum tinggal dua skrip analisis (`_e2_banding.py`, `_e4_kurva_trust.py`), dan keduanya
hanya membaca berkas hasil, tidak menjalankan pelatihan — jadi bisa dibuat sambil E2 berjalan.

`_e2_metrik.py` ditandai **opsional**: empat metrik inti (gini/penerimaan/tunggu/kepercayaan) plus
sebaran kepercayaan sudah dihitung langsung oleh pipeline. Skrip itu hanya perlu kalau metrik kaya
(*herding*, entropi rekomendasi, utilisasi per-stasiun) ingin dilaporkan juga.

### Urutan pembuatan sisanya

| Tahap | Skrip | Membuka |
|---|---|---|
| ~~1~~ | ~~`_pipeline_hipotesis.py`, `_asap_e0.py`, `_e2_jalankan.py`~~ | **selesai — E0, E2, E3 siap jalan** |
| 2 | `_e2_banding.py`, `_e4_kurva_trust.py` | E4 — **jalur minimum lengkap di sini** |
| 3 | `_e1_aturan_trust*.py` | E1 |
| 4 | `_ablasi_acak_riwayat.py`, `_e5_*.py` | E5 |
| 5 | `_e6_*.py`, `_e7_performativitas.py` | E6, E7 |

---

## Struktur folder

```
Eksekusi_Hipotesis/
├── README.md          <- dokumen ini
├── outputs/           <- hasil, satu file per eksperimen, nama bertanda tanggal
└── logs/              <- log jalannya, satu file per eksperimen
```

### Menyimpan hasil ke git

`.gitignore` mengecualikan isi `outputs/`, kecuali pola `uji_*.json`, `kalibrasi_*.json`,
dan `*_konsolidasi.json`. Hasil folder ini memakai awalan `e<nomor>_`, jadi **tidak otomatis
terlacak**. Setelah tiap eksperimen selesai:

```bash
git add -f Eksekusi_Hipotesis/outputs/e2_banding_20260824.json
```

Model (`*.pt`) **jangan** dimasukkan git — biarkan di server, catat namanya di tabel lacak.

---

## Tabel lacak

Diperbarui tiap satu eksperimen selesai.

| Kode | Menguji | Status | Hasil | Tanggal |
|---|---|---|---|---|
| E0 | prasyarat | **SELESAI** | uji asap lulus 15/15; pipeline teruji ujung-ke-ujung | 2026-08-23 |
| E1 | H3 | belum | — | — |
| E2 | H6b | belum | — | — |
| E3 | H6b | belum | — | — |
| E4 | H2 | belum | — | — |
| E5 | H1 | belum | — | — |
| E6 | H4 | belum | — | — |
| E7 | H5 | belum | — | — |
| E8 | ketahanan | belum | — | — |
