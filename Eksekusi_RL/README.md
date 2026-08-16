# Eksekusi_RL — Pelaksanaan Rencana Penelitian

Folder ini adalah **pelaksanaan** dari `../Dokumen_Penting/Rencana_Eksekusi_Penelitian.md`.
Setiap tahap (0–6) punya satu notebook. Notebook dibangun terprogram oleh
`build_notebooks.py` (agar sel markdown/kode konsisten dengan rencana), lalu **dieksekusi
dan diisi manual** seiring tahap berjalan — bukan sekali jalan dari awal sampai akhir.

## Struktur

```
Eksekusi_RL/
  common.py                          modul bersama: substrat, dekomposisi reward, uji
                                      alignment, sensitivitas performatif, runner baseline
  build_notebooks.py                 pembangun 7 notebook dari kerangka (jalankan ulang
                                      HANYA bila rencana induk direvisi struktural)
  00_Bekukan_Substrat.ipynb          Tahap 0 -- SUDAH DIEKSEKUSI (lihat status di bawah)
  01_Tetapkan_Rezim.ipynb            Tahap 1 -- SUDAH DIEKSEKUSI SKALA DEMO (lihat catatan)
  02_Replikasi_Baseline_PDQN.ipynb   Tahap 2 -- kerangka, training PDQN belum dijalankan
  03_Eksperimen_Pivot.ipynb          Tahap 3 -- kerangka, training belum dijalankan
  04_Solusi_RRM_Arsitektur.ipynb     Tahap 4 -- kerangka, training belum dijalankan
  05_Robustness_Stabilitas.ipynb     Tahap 5 -- kerangka
  06_Pelaporan.ipynb                 Tahap 6 -- kerangka konsolidasi akhir
  outputs/                           hasil JSON tiap tahap (dipakai ulang oleh notebook lain)
  figures/                           grafik PNG tiap tahap
```

## Status eksekusi saat ini

| Tahap | Status | Catatan |
|---|---|---|
| **0** Bekukan substrat | ✅ **GERBANG LULUS PENUH (re-verifikasi pasca ruang aksi v2)** | Ruang aksi disederhanakan total (`Spesifikasi_Teknis_RL.md` v2: top-K/threshold+ε-greedy, `alpha_honesty` dihapus, EstWait selalu jujur, `rec_activity` wajib di obs aktor) — notebook dijalankan ulang end-to-end. Atribusi varians re-kalibrasi (seimbang **25,2%** TEPAT DI BATAS turun dari 57,8% v1 krn honesty dulu ikut menyumbang; individual **74,3%**; kumulatif **6,5%** disengaja), anti-herding jendela bergulir **87,7%**, alignment per-suku flock-vs-herding ρ=**−0,91** tetap kuat. `pytest` 38/38 & F1 59/59 terkonfirmasi ulang. Horizon 60/90-hari tersedia sbg berkas permanen. |
| **1** Tetapkan rezim | ✅ **GERBANG LULUS PENUH (skala N_SEED=5)** | Rentang trust 4×/1× = **34,0×** (CI tak tumpang tindih), wait 4× = **22,60 menit** (dalam rentang 10-60), acceptance 4× turun monoton (−0,00132/hari) vs 1× datar (+0,00012/hari). Rezim operasi **4× dibekukan**. Baseline S0–S3 lengkap (10 seed) di rezim operasi, termasuk `greedy_util`. |
| **2** Replikasi baseline (titik A) | ✅ **GERBANG LULUS DENGAN CATATAN** | Titik A direvisi jadi **H-PPO v2** (bukan `PDQNContinuousTrainer` — modul itu dibangun di atas mekanisme a2 yg sudah dihapus v2). Sapuan trust statis 3 titik (0,4/0,65/0,9) x 3 train-seed x 10 eval-seed. **Bug struktural GAE ditemukan & diperbaiki** (`compute_gae` mem-bootstrap lintas transisi tak-berdekatan waktu krn reward tertunda — C1 re-sort + C4 time-distance gating, lihat `Rencana_Eksekusi_Penelitian.md` Tahap 2). Hasil akhir: menang signifikan vs greedy_queue **dan** greedy_util di 3/3 titik trust (p=0,0020), spread antar-seed membaik 10-30× (1/3 titik tepat target ≤0,005, 2/3 mendekati). Skrip: `_tahap2_train.py`/`_tahap2_eval.py`, hasil di `outputs/02_*.json`, kesimpulan penuh di `02_Replikasi_Baseline_PDQN.ipynb`. |
| **3** Eksperimen pivot | ⚠️ **POKOK TIDAK LULUS (hasil negatif sah) — PERLUASAN menemukan lock-in** | Lengan pokok (A=statis 0,65, B=dinamis, A′=statis 0,5144): B TIDAK signifikan lebih buruk dari A (p=0,16, d=+0,72) & TIDAK beda dari A′ (p=0,92) — performativitas tak terbukti merusak seiring waktu di rezim/horizon ini, trust B cuma bergerak +1,4% dari 0,5 (EstWait v2 selalu jujur). **Perluasan** (sapuan `initial_trust`=0,3/0,5/0,7 × horizon 90-hari): ditemukan **pola lock-in path-dependent** — trust awal RENDAH (0,3) → Gini ~2,5× lebih buruk drpd mulai 0,5/0,7, efek RAKSASA & presisi (d=+4,08 & +4,37, p=0,002; kontrol 0,5-vs-0,7 sama sekali tak beda, p=1,0). Klaim tesis direvisi: bukan "performativitas merusak seiring waktu", tapi **"sistem menunjukkan lock-in thd kondisi trust awal"**. Bug `initial_trust()` ditemukan & diperbaiki (sama pola dgn `constant_trust()`). Detail: `03_Eksperimen_Pivot.ipynb`. |
| **4–6** | ⏳ Kerangka siap, belum dieksekusi | Berisi kode kerja (bukan hanya deskripsi) untuk bagian yang murah/cepat; bagian training RRM sengaja **dikomentari** dan ditandai `[TODO]` karena butuh anggaran komputasi non-trivial di luar satu sesi interaktif. Struktur & kriteria sudah lengkap — tinggal mengaktifkan blok training, **WAJIB pakai konfigurasi Tahap 2 (`02_config_beku.json`, termasuk `max_step_gap` hasil perbaikan GAE) identik**, dgn kerangka klaim Tahap 3 yg DIREVISI (lock-in, bukan drift-seiring-waktu). |

## Catatan revisi: trust TIDAK di-carry-forward lintas horizon

Draf awal Tahap 0.2 menuntut carry-forward trust lintas batas horizon pelatihan. **Ini
dicabut** setelah didiskusikan ulang: kurva performativitas (Tahap 1) terbukti muncul dalam
**satu pass tunggal 30-hari** tanpa carry-forward sama sekali — beban (`load_multiplier`)
yang menentukan hidup-matinya loop, bukan reset. Reset per-horizon juga **konsisten** dengan
protokol evaluasi S0–S3 yang sudah mapan (tiap seed = simulasi 30-hari independen, trust
mulai 0,5). Kronologi lengkap: `Rumusan_Masalah_Teknis_RL.md` §4.1.

Yang tersisa hanyalah volume data pelatihan (kunjungan/pengguna rendah dalam 30 hari) —
diatasi dengan **memperpanjang horizon dataset itu sendiri** (60/90 hari, jendela kontinu),
bukan replay+carry-forward. `common.generate_horizon_datasets()` men-generate berkas
**permanen** di root repo dan mengembalikan config input + ringkasan output secara terpisah
(`common.measure_visit_density()` masih ada sbg utilitas pengukuran-saja/scratch, tidak
dipakai lagi oleh notebook 0.2).

**Dataset horizon-diperpanjang sudah tersedia** (di-generate sesi ini):
`scenario_dataset_klaster12_60d.json`, `scenario_dataset_klaster12_90d.json` — parameter
sama persis dengan 30-hari kanonik (`n_users=2636`, `load_multiplier=1.0`, `seed=42`,
`model_config.json` sama), hanya `horizon_days` yang berbeda.

## Cara melanjutkan

**Tahap 0 dan 1 sudah lulus gerbang penuh.** Rezim operasi (4× load_multiplier) sudah
dibekukan di `common.SUBSTRAT`. Langkah berikut:

1. **Tahap 2 (`02_Replikasi_Baseline_PDQN.ipynb`)**: aktifkan blok training PDQN yang
   dikomentari (titik A, trust statis) memakai substrat & rezim operasi yang sudah beku.
   Ikuti konfigurasi "dibekukan sebelum uji" di §2.0 notebook tersebut.
2. **Tahap 3 dan seterusnya**: aktifkan blok kode yang dikomentari begitu checkpoint
   Tahap 2 tersedia. Setiap notebook punya sel "konfigurasi dibekukan" yang **wajib**
   dipakai identik lintas tahap (lihat `Metodologi_Perbandingan_PDQN_RRM.md`).

## Menjalankan ulang / mem-build ulang notebook

```bash
# Membangun ULANG seluruh notebook dari kerangka (menghapus hasil eksekusi manual!
# hanya jalankan bila rencana induk berubah struktural — bukan untuk update rutin)
python Eksekusi_RL/build_notebooks.py

# Mengeksekusi satu notebook end-to-end via nbclient (dari root repo)
python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('Eksekusi_RL/00_Bekukan_Substrat.ipynb', as_version=4)
NotebookClient(nb, timeout=600, kernel_name='python3',
               resources={'metadata': {'path': 'Eksekusi_RL'}}).execute()
nbformat.write(nb, 'Eksekusi_RL/00_Bekukan_Substrat.ipynb')
"
```

**Peringatan**: `build_notebooks.py` menimpa seluruh isi notebook (termasuk output & catatan
kesimpulan yang sudah diisi manual). Setelah tahap mulai diisi manual, jangan jalankan
`build_notebooks.py` ulang untuk notebook itu kecuali memang ingin mengembalikannya ke
kerangka kosong.

## Rujukan

- `../Dokumen_Penting/Rencana_Eksekusi_Penelitian.md` — rencana lengkap (tahapan, tujuan,
  ukuran ketercapaian, gerbang)
- `../Dokumen_Penting/Rumusan_Masalah_Teknis_RL.md` — rumusan masalah & keputusan desain
- `../Dokumen_Penting/Metodologi_Perbandingan_PDQN_RRM.md` — aturan atribusi 3-lapisan,
  tangga ablasi A–E
- `../Validasi_Generik/LAPORAN_VALIDASI.md` — status validasi simulator (substrat)
