# Arsitektur: MASTER-Pure-PPO vs Hybrid

Gambaran tingkat tinggi. Ukuran ada di tabel terpisah di bagian akhir.

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

## 4. Ukuran

### 4.1 MASTER-Pure-PPO

| Bagian | Ukuran |
|---|---|
| Fitur per stasiun | 7 |
| Aktor — lapis tersembunyi | 64 |
| Aktor — keluaran | 1 per stasiun |
| Sebaran tawaran (varian DDPG) | 1 parameter, dibagi |
| Penanda keterlambatan → sandi | 1 → 64 |
| Masukan kritik per stasiun | 7 + 64 |
| Atensi kritik — tersembunyi | 64 |
| Kepala nilai | 2 |

### 4.2 Hybrid

Modul tambahannya **sengaja dibuat kecil** supaya tidak mendominasi tulang punggung.

| Bagian | Ukuran |
|---|---|
| Fitur per stasiun | 7 (sama) |
| Kepala vektor — tersembunyi | 16 |
| Vektor per stasiun | 8 |
| Riwayat preferensi | 10 langkah |
| Penyandi preferensi — keluaran | 8 |
| Atensi preferensi — keluaran | 8 |
| Atensi antar-stasiun | 8 |
| Kepala akhir | 1 per stasiun |
| Kritik | sama dengan MASTER-Pure-PPO |

Bandingkan: aktor MASTER murni memakai lapis tersembunyi **64**, sedangkan seluruh jalur
tambahan Hybrid bekerja pada dimensi **8**. Tambahannya jauh lebih ramping daripada yang
ditambahi.

### 4.3 Gerbang

| Gerbang | Nilai awal | Bentuk |
|---|---|---|
| Preferensi | 0 (atau kecil bukan-nol) | bebas tanda |
| Penggabung akhir | sigmoid(−2) ≈ 0,12 | sigmoid |
| Atensi antar-stasiun | sigmoid(−2) ≈ 0,12 | sigmoid |

Gerbang preferensi **bebas tanda**, bukan sigmoid seperti dua lainnya. Konsekuensinya ia
bisa konvergen ke tanda berbeda antar-seed.

### 4.4 Fitur riwayat preferensi

| Mode | Isi tiap langkah | Ukuran |
|---|---|---|
| Identitas | one-hot rekomendasi + one-hot pilihan | 2 × jumlah stasiun |
| Ciri | ciri stasiun direkomendasikan + ciri stasiun dipilih | 10 |
| Ciri + hasil | ditambah patuh dan galat terealisasi | 12 |

Mode ciri memakai jarak, tunggu, antrean, konektor, utilisasi. Mode ciri-plus-hasil
membuat penyandi preferensi menduga **preferensi dan kepercayaan sekaligus**, sehingga
penyandi riwayat terpisah tak lagi diperlukan.

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
