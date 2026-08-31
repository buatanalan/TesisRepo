# Arsitektur: MASTER-Pure-PPO vs Hybrid

Gambaran tingkat tinggi pada diagram; ukuran dan susunan lapisan tiap komponen ada di
tabel §4.

Keduanya memakai **tulang punggung latih yang sama** (PPO) dan **observasi stasiun yang
sama** (7 fitur §3.1 murni). Yang ditambahkan Hybrid: modul preferensi dan atensi
antar-stasiun di sisi aktor.

---

## 1. MASTER-Pure-PPO

Aktor mengubah fitur stasiun langsung menjadi tawaran. Tidak ada memori, tidak ada
informasi pemohon.

```
   ┌───────────────────────┐
   │  Observasi stasiun    │
   │  (§3.1 murni)         │
   └───────────┬───────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
 ╔═══════════╗    ╔═══════════════╗
 ║   AKTOR   ║    ║    KRITIK     ║
 ╚═════╤═════╝    ╚═══════╤═══════╝
       │                  │
       ▼                  │   Delayed Access
 ┌───────────┐            │   Strategy
 │    MLP    │            │        │
 │  (bobot   │            ▼        ▼
 │  dibagi)  │       ┌──────────────────┐
 └─────┬─────┘       │  gabung obs +    │
       │             │  penanda tunda   │
       ▼             └────────┬─────────┘
 ┌───────────┐                │
 │  tawaran  │                ▼
 │ per       │       ┌──────────────────┐
 │ stasiun   │       │  Penggabung      │
 └─────┬─────┘       │  ber-ATENSI      │
       │             └────────┬─────────┘
       ▼                      │
 ┌───────────┐                ▼
 │  tapis    │       ┌──────────────────┐
 │  kelayakan│       │  Kepala nilai    │
 └─────┬─────┘       │  GANDA           │
       │             └────────┬─────────┘
       ▼                      │
 ┌───────────┐                ▼
 │  softmax  │       ┌──────────────────┐
 │  → pilih  │       │  DGR — bobot     │
 └─────┬─────┘       │  antar kepala    │
       │             │  menyesuaikan    │
       ▼             └──────────────────┘
 ┌───────────┐
 │  stasiun  │
 │ terpilih  │
 └───────────┘
```

---

## 2. Hybrid — atensi antar-stasiun + modul P

Aktor tidak lagi langsung menghasilkan tawaran. Ia membentuk **vektor** per stasiun,
menyisipkan preferensi setelah vektor terbentuk, lalu membiarkan stasiun saling
memandang sebelum diskor.

```
   ┌───────────────────────┐        ┌───────────────────────┐
   │  Observasi stasiun    │        │  Riwayat preferensi   │
   │  (§3.1 murni — SAMA)  │        │  (rekomendasi vs      │
   └───────────┬───────────┘        │   pilihan nyata)      │
               │                    └───────────┬───────────┘
               │                                │
        ┌──────┴──────┐                         ▼
        │             │               ┌───────────────────┐
        │             │               │  Penyandi rekuren │
        │             │               │  preferensi       │
        │             │               └─────────┬─────────┘
        │             │                         │
        │             └────────────┐            │
        │                          ▼            ▼
        │                 ┌────────────────────────────┐
        │                 │   ATENSI PREFERENSI        │
        │                 │   preferensi = kunci cari  │
        │                 │   stasiun  = yang dicari   │
        │                 └────────────┬───────────────┘
        │                              │
        ▼                              ▼
 ┌──────────────┐              ┌───────────────┐
 │ Kepala       │              │  GERBANG      │
 │ vektor       │              │  preferensi   │
 │ stasiun      │              └───────┬───────┘
 │ (bobot       │                      │
 │  dibagi)     │                      │
 └──────┬───────┘                      │
        │                              │
        │  vektor per stasiun,         │
        │  MURNI fitur sendiri         │
        │                              │
        └──────────────┬───────────────┘
                       ▼
             ┌───────────────────┐
             │  PENGGABUNG AKHIR │   ← P masuk SETELAH vektor
             │  (gerbang)        │      terbentuk (late-merge)
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │  ATENSI ANTAR-    │   ← stasiun saling
             │  STASIUN          │      memandang
             │  (gerbang,        │
             │   residual)       │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │  vektor akhir     │
             │  per stasiun      │
             └─────────┬─────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐           ┌─────────────┐
   │ kepala PPO  │           │ kepala DDPG │
   │ → skor      │           │ → tawaran   │
   │ → softmax   │           │   kontinu   │
   └─────────────┘           └─────────────┘
```

Sisi kritiknya sama dengan MASTER-Pure-PPO. Ada varian kritik yang **juga** melihat
riwayat preferensi, karena kritik buta-P tidak bisa menjelaskan hasil yang didorong
preferensi — sumber variansi tambahan pada pendugaan keuntungan.

---

## 3. Perbedaan pokoknya

```
                  MASTER-PURE-PPO              HYBRID
                  ───────────────              ──────

  observasi       7 fitur §3.1                 7 fitur §3.1   ← SAMA
                       │                            │
  keluaran        skalar langsung              vektor per stasiun
  aktor tahap 1        │                            │
                       │                            ▼
  preferensi      tidak ada                    disisipkan SETELAH
                       │                       vektor terbentuk
                       │                            │
                       │                            ▼
  antar-stasiun   tidak ada di aktor           atensi antar-stasiun
                       │                            │
                       ▼                            ▼
  kepala akhir    tawaran / skor               tawaran / skor
                       │                            │
  kritik          ber-atensi + DGR             sama (+ varian ber-P)
```

**Tiga tambahan Hybrid, dan alasan urutannya:**

1. **Keluaran aktor jadi vektor, bukan skalar.** Skalar tak menyisakan ruang untuk
   disisipi apa pun. Vektor memberi tempat bagi preferensi dan atensi antar-stasiun
   bekerja sebelum keputusan dipadatkan jadi satu angka.

2. **Preferensi disisipkan belakangan.** Vektor stasiun dibentuk lebih dulu dari fitur
   stasiun **sendiri**, baru preferensi digabungkan. Kalau preferensi ikut sejak awal,
   representasi stasiun dan preferensi berebut kapasitas yang sama.

3. **Atensi antar-stasiun di sisi aktor.** MASTER murni hanya punya atensi di kritik —
   untuk menilai, bukan memutuskan. Di Hybrid, stasiun saling memandang sebelum diskor,
   sehingga keputusan tiap stasiun memperhitungkan tetangganya.

**Semua tambahan memakai gerbang.** Di awal pelatihan, ketiganya nyaris tak berkontribusi
dan jaringan berperilaku mendekati MASTER murni.

## Kesetiaan terhadap paper

Menambahkan modul P **melanggar Pers. 11 MASTER** — stasiun seharusnya buta terhadap
pemohon. Deviasi ini disengaja dan terdokumentasi.

Yang **dijaga**: observasi stasiun tetap 7 fitur §3.1 murni. Preferensi masuk lewat kanal
**terpisah**, tidak mencemari observasi stasiun itu sendiri. Jadi pelanggarannya terbatas
pada satu titik yang bisa ditunjuk, bukan menyebar ke seluruh arsitektur.

---

## 4. Ukuran dan susunan lapisan

Notasi: `LIN(masuk→keluar)`. Penanda `−b` berarti tanpa bias.

### 4.1 MASTER-Pure-PPO

| Komponen | Susunan lapisan | Keluaran |
|---|---|---|
| Aktor (bobot dibagi) | `LIN(7→64) → ReLU → LIN(64→64) → ReLU → LIN(64→1)` | 1 per stasiun, mentah |
| Sebaran tawaran (DDPG) | `skala = exp(log_std)` → `Normal(rerata, skala)` | 1 parameter, dibagi |
| Penyandi keterlambatan | `LIN(1→64) → ReLU` | 64 |
| Gabungan masukan kritik | `concat[ obs(7), p(64) ]` | 71 |
| Atensi kritik — skor | `LIN(71→64, −b) → tanh → LIN(64→1, −b)` | 1 per stasiun |
| Atensi kritik — bobot | `mask −∞ → softmax` | N, jumlah 1 |
| Atensi kritik — keluaran | `bobot · raw → LIN(71→64) → ReLU` | 64 |
| Kepala nilai | `LIN(64→2)` | 2 |

### 4.2 Hybrid

Modul tambahannya **sengaja kecil** supaya tidak mendominasi tulang punggung.

| Komponen | Susunan lapisan | Keluaran |
|---|---|---|
| Kepala vektor stasiun | `LIN(7→16) → ReLU → LIN(16→8)` | 8 per stasiun |
| Penyandi preferensi | `pack_padded → LSTM(10 atau 12 → 8)` → keadaan terakhir | 8 |
| Atensi preferensi — kunci/nilai | `LIN(7→8)` | 8 per stasiun |
| Atensi preferensi — pencari | `LIN(8→8)` | 8 |
| Atensi preferensi — bobot | `kv·q/√8 → softmax` | N, jumlah 1 |
| Atensi preferensi — keluaran | `bobot · kv` lalu `× pref_gate` | 8 |
| Penggabung akhir | `concat[vec(8), pref(8)] → LIN(16→8) → ReLU` | 8 |
| Penggabung akhir — bentuk | `vec + sigmoid(g) × (merged − vec)` | interpolasi |
| Atensi antar-stasiun — q/k/v | `LIN(8→8)` × 3 | 8 masing-masing |
| Atensi antar-stasiun — bobot | `q·kᵀ/√8 → mask −∞ → softmax` | N×N |
| Atensi antar-stasiun — keluaran | `bobot · v → LIN(8→8)` | 8 |
| Atensi antar-stasiun — bentuk | `vec + sigmoid(g) × attended` | residual |
| Kepala akhir PPO | `LIN(8→1) → mask −∞ → softmax` | skor per stasiun |
| Kepala akhir DDPG | `LIN(8→1) → tanh → × 10` | tawaran kontinu |
| Kritik | sama persis dengan §4.1 | 2 |

Dua catatan yang mudah terlewat:

**Kepala vektor berakhir tanpa aktivasi.** Vektornya sengaja dibiarkan linear supaya
penggabungan berikutnya tidak dibatasi tanda.

**Kedua gerbang berbentuk berbeda.** Penggabung akhir memakai *interpolasi* — pada gerbang
0 keluarannya persis vektor semula. Atensi antar-stasiun memakai *residual* — pada gerbang
0 tambahannya nol. Serupa akibatnya di awal, tetapi tidak sama setelah gerbang membuka.

### 4.3 Ringkasan aktivasi

| Komponen | Aktivasi | Catatan |
|---|---|---|
| Aktor MASTER murni | ReLU × 2 | keluaran mentah, tanpa `tanh` |
| Kepala vektor Hybrid | ReLU × 1 | keluaran linear, disengaja |
| Penyandi preferensi | `tanh`/`sigmoid` internal LSTM | baku PyTorch |
| Atensi preferensi | — | hanya `softmax` |
| Penggabung akhir | ReLU + gerbang `sigmoid` | interpolasi |
| Atensi antar-stasiun | gerbang `sigmoid` | residual |
| Atensi kritik | `tanh` di skor, ReLU di keluaran | dua tempat berbeda |
| Kepala PPO | `softmax` | atas kandidat layak |
| Kepala DDPG | `tanh` | diskalakan × 10 |

### 4.4 Gerbang

| Gerbang | Nilai awal | Bentuk | Rentang |
|---|---|---|---|
| Preferensi | 0 (atau kecil bukan-nol) | bebas tanda | tak terbatas |
| Penggabung akhir | `sigmoid(−2)` ≈ 0,12 | `sigmoid` | (0, 1) |
| Atensi antar-stasiun | `sigmoid(−2)` ≈ 0,12 | `sigmoid` | (0, 1) |

Gerbang preferensi **bebas tanda**, bukan `sigmoid` seperti dua lainnya. Konsekuensinya ia
bisa konvergen ke tanda berbeda antar-seed.

### 4.5 Fitur riwayat preferensi

| Mode | Isi tiap langkah | Ukuran |
|---|---|---|
| Identitas | one-hot rekomendasi + one-hot pilihan | 2 × jumlah stasiun |
| Ciri | ciri stasiun direkomendasikan + ciri stasiun dipilih | 10 |
| Ciri + hasil | ditambah patuh dan galat terealisasi | 12 |

Ciri stasiun: jarak, tunggu, antrean, konektor, utilisasi. Panjang jendela 10 langkah.

Mode ciri-plus-hasil membuat penyandi preferensi menduga **preferensi dan kepercayaan
sekaligus**, sehingga penyandi riwayat terpisah tak lagi diperlukan.

---

## 5. Catatan

**Gerbang tepat nol menciptakan kebuntuan.** Pada gerbang preferensi persis 0, gradien
yang sampai ke modul preferensi adalah persis nol — modul tak bisa mulai belajar sampai
gerbangnya bergeser, padahal gerbang itu sendiri bergerak lambat. Nilai awal kecil
bukan-nol memutus kebuntuan tanpa meninggalkan semangat "mulai hampir tak berkontribusi".

**Atensi antar-stasiun bisa dimatikan** lewat ablasi, untuk memisahkan kontribusinya dari
modul preferensi. Modulnya tetap dibangun agar model tetap dapat dimuat lintas varian,
hanya dilewati saat maju.

**Kepala akhir berbeda per algoritma.** DDPG mempertahankan tawaran kontinu dengan argmax
— mekanisme asli MASTER. PPO memakai skor dan softmax. Tulang punggungnya identik supaya
perbandingannya adil.

*Sumber: `marl_spklu/rl/master_pure_ppo_policy.py`,
`marl_spklu/rl/master_pure_hybrid_policy.py`, `marl_spklu/rl/pdqn_policy.py`,
`marl_spklu/rl/master_paper_obs.py`.*
