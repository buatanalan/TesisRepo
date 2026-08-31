# Arsitektur: MASTER Murni vs Usulan

Gambaran tingkat tinggi. Ukuran lapisan dan dimensi ada di tabel terpisah di bagian akhir.

---

## 1. MASTER Murni

Unit agennya **stasiun**. Setiap stasiun mengajukan tawaran untuk tiap permintaan yang
masuk, dan tawaran tertinggi memenangkan permintaan itu.

```
                          ┌─────────────────────────────┐
                          │   Observasi per stasiun      │
                          │   (§3.1 paper MASTER)        │
                          └─────────────┬───────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 │                                             │
                 ▼                                             ▼
        ╔════════════════╗                          ╔════════════════════╗
        ║     AKTOR      ║                          ║       KRITIK       ║
        ║ (bobot dibagi  ║                          ║   (terpusat)       ║
        ║  antar stasiun)║                          ╚════════╤═══════════╝
        ╚════════╤═══════╝                                   │
                 │                          Delayed Access ──┤
                 ▼                          Strategy         │
          ┌─────────────┐                        │           │
          │  MLP tawar  │                        ▼           ▼
          └──────┬──────┘                   ┌─────────────────────┐
                 │                          │  gabung obs seluruh │
                 ▼                          │  stasiun + penanda  │
        ┌─────────────────┐                 │  keterlambatan      │
        │ tawaran stasiun │                 └──────────┬──────────┘
        │  (satu skalar   │                            │
        │   per stasiun)  │                            ▼
        └────────┬────────┘                 ┌─────────────────────┐
                 │                          │  Penggabung         │
                 ▼                          │  ber-ATENSI         │
        ┌─────────────────┐                 │  (invarian urutan)  │
        │  GAME LELANG    │                 └──────────┬──────────┘
        │  tawaran        │                            │
        │  tertinggi      │                            ▼
        │  menang         │                 ┌─────────────────────┐
        └────────┬────────┘                 │  Kepala nilai       │
                 │                          │  GANDA (multi-      │
                 ▼                          │  critic)            │
        ┌─────────────────┐                 └──────────┬──────────┘
        │  penugasan EV   │                            │
        │  ke stasiun     │                            ▼
        │  pemenang       │                 ┌─────────────────────┐
        └─────────────────┘                 │  DGR — bobot antar  │
                                            │  kepala menyesuaikan│
                                            │  diri              │
                                            └─────────────────────┘
```

**Ciri utamanya:**

| | |
|---|---|
| Unit agen | Stasiun |
| Aksi | Tawaran kontinu, satu skalar per stasiun |
| Pemilihan | Lelang — tawaran tertinggi menang |
| Kritik | Terpusat, melihat seluruh stasiun sekaligus |
| Atensi | Di **kritik**, untuk meringkas keadaan bersama |
| Info pemohon | **Tidak ada** — stasiun buta terhadap siapa yang dilayani |

---

## 2. Usulan

Unit agennya **permintaan pengisian EV**. Satu permintaan melihat seluruh stasiun
kandidat sekaligus, lalu memilih beberapa teratas.

```
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ Observasi tiap   │   │ Riwayat          │   │ Riwayat          │
   │ stasiun kandidat │   │ interaksi        │   │ preferensi       │
   │ + keadaan EV     │   │ (kepatuhan)      │   │ (rekomendasi vs  │
   │   pemohon        │   │                  │   │  pilihan nyata)  │
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            │                      │                      │
            │                      ▼                      ▼
            │             ┌────────────────┐    ┌──────────────────┐
            │             │ Penyandi       │    │ Penyandi         │
            │             │ rekuren        │    │ rekuren          │
            │             │ riwayat        │    │ preferensi       │
            │             └────────┬───────┘    └────────┬─────────┘
            │                      │                     │
            ├──────────────────────┼─────────────────────┤
            │                      │                     ▼
            │                      │           ┌──────────────────┐
            │                      │           │ ATENSI           │
            ├──────────────────────┼──────────►│ PREFERENSI       │
            │  (stasiun sbg        │           │ preferensi = ku- │
            │   kunci & nilai)     │           │ nci pencarian    │
            │                      │           └────────┬─────────┘
            │                      │                    │
            │                      │                    ▼
            │                      │           ┌──────────────────┐
            │                      │           │ GERBANG          │
            │                      │           │ (mulai dari nol) │
            │                      │           └────────┬─────────┘
            │                      │                    │
            │                      └─────────┬──────────┘
            │                                ▼
            │                       ┌──────────────────┐
            │                       │ KONTEKS gabungan │
            │                       └────────┬─────────┘
            │                                │
            ├────────────────┬───────────────┤
            │                │               │
            ▼                ▼               ▼
   ╔════════════════╗    (konteks disiarkan ke tiap stasiun)
   ║     AKTOR      ║                        │
   ╚════════╤═══════╝                        │
            │                                │
            ▼                                ▼
   ┌──────────────────┐            ╔══════════════════╗
   │ Penyandi stasiun │            ║      KRITIK      ║
   │ (bobot dibagi)   │            ╚═════════╤════════╝
   └────────┬─────────┘                      │
            │                                ▼
            ▼                       ┌──────────────────┐
   ┌──────────────────┐             │ Penyandi stasiun │
   │ Kepala pemilihan │             │ (bobot dibagi)   │
   └────────┬─────────┘             └────────┬─────────┘
            │                                │
            ▼                                ▼
   ┌──────────────────┐             ┌──────────────────┐
   │ skor per stasiun │             │ Penggabung       │
   └────────┬─────────┘             │ ber-ATENSI       │
            │                       └────────┬─────────┘
            ▼                                │
   ┌──────────────────┐                      ▼
   │ tapis kelayakan  │             ┌──────────────────┐
   │ → ambang         │             │ Kepala nilai     │
   │ → ambil top-k    │             │ TERPISAH         │
   └────────┬─────────┘             │ per aliran tujuan│
            │                       └──────────────────┘
            ▼
   ┌──────────────────┐
   │ k rekomendasi    │
   └──────────────────┘
```

**Ciri utamanya:**

| | |
|---|---|
| Unit agen | Permintaan pengisian EV |
| Aksi | Pemilihan diskrit — beberapa stasiun teratas |
| Pemilihan | Ambang lalu potong di $k$ |
| Kritik | Terpusat, melihat seluruh stasiun kandidat |
| Atensi | Di **dua tempat**: penggabung kritik, dan pencocokan preferensi |
| Info pemohon | **Ada** — disisipkan ke tiap baris kandidat |

---

## 3. Perbedaan pokoknya

```
                  MASTER MURNI                    USULAN
                  ────────────                    ──────

  agen            stasiun                         permintaan EV
                       │                                │
  aksi            tawaran kontinu                  pilih beberapa stasiun
                       │                                │
  penentuan       lelang antar-stasiun            ambang + top-k
                       │                                │
  atensi          kritik saja                      kritik + preferensi
                       │                                │
  preferensi      tidak ada                        modul terpisah + gerbang
                       │                                │
  riwayat         tidak ada                        penyandi rekuren
                       │                                │
  keadaan EV      tidak terlihat                   masuk tiap baris kandidat
                       │                                │
  kepala nilai    ganda + DGR                      terpisah per aliran tujuan
```

**Tiga hal yang berpindah tempat:**

1. **Siapa yang memutuskan.** Dari stasiun berebut permintaan, menjadi permintaan memilih
   stasiun. Ini yang membuat riwayat pengguna jadi bermakna — pada MASTER murni tidak ada
   "pengguna" yang bisa diingat, karena agennya stasiun.

2. **Atensi mendapat pekerjaan kedua.** Pada MASTER murni, atensi hanya meringkas keadaan
   bersama untuk kritik. Pada usulan, ada atensi kedua yang mencocokkan preferensi dengan
   ciri stasiun.

3. **Kepala nilai ganda dipakai berbeda.** MASTER murni memakainya untuk beberapa objektif
   dengan bobot yang menyesuaikan diri (DGR). Usulan memakainya untuk memisahkan aliran
   tujuan yang berbeda watak — tertunda vs segera, jarang vs sering.

---

## 4. Ukuran

### 4.1 MASTER murni

| Bagian | Ukuran |
|---|---|
| Fitur per stasiun | 7 |
| Aktor — lapis tersembunyi | 64 |
| Aktor — keluaran | 1 per stasiun |
| Sebaran tawaran | 1 parameter, dibagi seluruh stasiun |
| Penanda keterlambatan → sandi | 1 → 64 |
| Masukan kritik per stasiun | 7 + 64 |
| Atensi kritik — lapis tersembunyi | 64 |
| Kepala nilai | 2 |

### 4.2 Usulan

| Bagian | Ukuran |
|---|---|
| Fitur per stasiun kandidat | 10 |
| — dari §3.1 MASTER | 7 |
| — keadaan EV pemohon | 3 |
| Riwayat interaksi | 10 langkah × 4 fitur |
| Penyandi riwayat — keluaran | 16 |
| Riwayat preferensi | 10 langkah × 10 fitur |
| Penyandi preferensi — keluaran | 16 |
| Atensi preferensi — keluaran | 16 |
| Gerbang preferensi | 1 parameter |
| Konteks gabungan | 16 + 16 = 32 |
| Penyandi stasiun aktor — tersembunyi | 64 |
| Penyandi stasiun kritik — tersembunyi | 128 |
| Kepala pemilihan — keluaran | 1 per stasiun |
| Kepala nilai | 3 |
| Rekomendasi per permintaan | 3 |

### 4.3 Fitur riwayat

| Riwayat | Isi tiap langkah |
|---|---|
| Interaksi | patuh, janji tunggu, tunggu bawaan, galat terealisasi |
| Preferensi | ciri stasiun yang direkomendasikan (5) + ciri stasiun yang dipilih (5) |

Ciri stasiun untuk riwayat preferensi: jarak, tunggu, antrean, konektor, utilisasi.

---

## 5. Catatan

**Penyandi riwayat interaksi dimatikan** pada seluruh lengan yang diuji. Modulnya tetap
ada di jaringan dan ukurannya tak berubah, tetapi kontribusinya dinolkan. Akibatnya pada
lengan tanpa modul preferensi, konteksnya kosong seluruhnya.

**Gerbang preferensi dimulai dari nol**, sehingga di awal pelatihan jaringan berperilaku
seolah tak punya modul preferensi. Kontribusinya masuk bertahap.

**Bobot dibagi antar stasiun** di kedua arsitektur — tiap stasiun diproses lewat jaringan
yang sama. Menukar urutan stasiun menghasilkan keluaran yang tertukar identik, bukan
keluaran berbeda.

*Sumber: `marl_spklu/rl/master_pure_ppo_policy.py`, `marl_spklu/rl/master_ev_ppo_policy.py`,
`marl_spklu/rl/policy.py`, `marl_spklu/rl/pdqn_policy.py`, `marl_spklu/rl/master_paper_obs.py`.*
