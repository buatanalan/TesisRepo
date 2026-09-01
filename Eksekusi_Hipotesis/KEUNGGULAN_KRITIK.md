# Keunggulan Kritik atas Aktor — Implementasi di Kode

Enam klaim umum tentang peran kritik dalam *actor-critic*, diperiksa satu per satu
terhadap implementasi Hybrid-PPO (`master_pure_hybrid_trainer.py`,
`master_pure_hybrid_policy.py`, `ppo.py`). Dua **tidak berlaku seperti klaim umumnya** —
ditandai eksplisit, bukan disamarkan.

---

## 1. Generalisasi nilai masa depan — BERLAKU

**Klaim**: kritik mengestimasi nilai jangka panjang lewat persamaan Bellman, bukan
sekadar imbalan instan.

**Implementasi**: `compute_gae` (`ppo.py`) memakai bentuk *differential return*
(Sutton & Barto Bab 10), karena tugasnya berkelanjutan tanpa titik reset alami:

```python
delta = (r - r_bar) + v_next * nonterminal - v
last  = delta + lam * nonterminal * last
```

Bukan $\gamma$-diskon biasa. Suku $v_{next} - v$ **adalah** persamaan Bellman dalam
bentuk selisih: nilai keadaan berikutnya menyusup ke taksiran keadaan sekarang. Tanpa
suku ini, sinyal belajar hanya berisi imbalan yang benar-benar teramati di episode
berjalan — buta terhadap apa yang menanti sesudahnya.

`r_bar` adalah taksiran laju imbalan rata-rata, dikelola trainer lintas *chunk*
pelatihan (bukan konstanta tetap).

*Sumber: `ppo.py::compute_gae`, baris 27–56.*

---

## 2. Reduksi variansi gradien — BERLAKU

**Klaim**: kritik menghasilkan sinyal evaluasi lebih stabil dibanding akumulasi imbalan
mentah satu episode.

**Implementasi**: keuntungan (*advantage*) yang dipakai memperbarui kebijakan bukan
imbalan mentah, melainkan **selisih** terhadap taksiran kritik — dan selisih itu
dinormalisasi lagi sebelum dipakai:

```python
returns, adv = compute_gae(transitions, self.gamma, self.lam, max_step_gap=self.max_step_gap)
adv = (adv - adv.mean(axis=0, keepdims=True)) / (adv.std(axis=0, keepdims=True) + 1e-8)
```

Normalisasi ini dilakukan **per-aliran** (K aliran imbalan, satu kolom per aliran) —
bukan atas gabungan semuanya. Konsekuensinya: ketimpangan skala antar-aliran (imbalan
individual yang sering & kecil vs imbalan global yang jarang & besar) hilang secara
struktural sebelum digabung lewat DGR.

Ini bekerja **karena** ada kritik yang menghasilkan $V$ untuk dikurangkan. Tanpa
kritik, yang tersisa hanyalah *return* Monte Carlo — variansinya jauh lebih besar,
terutama pada horizon panjang dan imbalan yang tertunda (lihat §6).

*Sumber: `master_pure_hybrid_trainer.py`, baris 461.*

---

## 3. Akses fitur & informasi lebih luas — ADA, TAPI DIMATIKAN SAAT DIPAKAI

**Klaim**: kritik menerima informasi tambahan yang tak tersedia bagi aktor.

**Implementasinya nyata.** Observasi aktor (§3.1 MASTER) memakai slot tersedia yang
dipotong di nol:

```python
slots_avail = max(0, cap_total - charging_total)
```

Sementara kanal *Delayed Access* milik kritik ($I^i_t$, Pers. 10 paper) memakai
besaran **tak dipotong**, bisa negatif:

```python
out[sid] = float((cap_total - charging_total) - queue_total)
```

Begitu stasiun penuh, aktor selalu melihat 0 — tak bisa membedakan "penuh tapi lengang"
dari "penuh dan menumpuk". Kritik bisa, lewat nilai negatif $I^i_t$ yang secara harfiah
berarti *"sekian EV mengantre"*.

Ini dijahit ke kritik lewat kanal terpisah:

```python
p = torch.relu(self.W_p(I_raw.unsqueeze(-1)))            # LIN(1->64), Pers. 10
raw = torch.cat([joint_obs, p, pref_exp], dim=-1)          # 7 + 64 + 8 = 79
```

**Tapi kanal ini dimatikan tepat di titik ia seharusnya berguna.** Saat pengumpulan
data — nilai $V$ yang dipakai membentuk *advantage* di §2 — kritik dipanggil dengan
$I$ dinolkan:

```python
zero_I = torch.zeros_like(mask_t, dtype=torch.float32)
value, _ = self.critic(obs_t, mask_t, zero_I, pref_hist_t)
```

Sementara saat pembaruan bobot (rugi nilai, MSE terhadap *return*), $I$ nyata dipakai:

```python
value, _ = self.critic(obs_b[mb], mask_b[mb], I_b[mb], ...)
v_loss = nn.functional.mse_loss(value, ret_b[mb])
```

Karena kanal ini menyumbang 64 dari 79 dimensi masukan kritik, $V(s, I{=}0)$ dan
$V(s, I_{\text{nyata}})$ berpotensi jadi dua fungsi yang cukup berbeda. Keuntungan
gradien kebijakan dihitung dari yang pertama; kritiknya dilatih untuk menjadi yang
kedua. **Manfaat pengurangan variansi dari §2 tidak sepenuhnya sampai**, karena
kanal informasi istimewa ini absen justru saat menghitung keuntungan.

*Sumber: `master_pure_trainer.py::snapshot_slots_raw` (definisi $I$),
`master_pure_hybrid_trainer.py` baris 132–136 (rollout, $I{=}0$) vs baris 492
(pembaruan, $I$ nyata).*

---

## 4. Efisiensi pemanfaatan data lama — TIDAK BERLAKU untuk lengan ini

**Klaim**: kritik mendukung pembelajaran *off-policy* dan *experience replay*.

**Tidak berlaku.** PPO adalah algoritma *on-policy*. Tidak ada `ReplayBuffer` di jalur
Hybrid-PPO — dicek langsung: kelas itu hanya muncul di jalur DDPG
(`master_pure_trainer.py`, `master_ddpg_trainer.py`, `pdqn_ddpg.py`), tidak pernah
diimpor oleh `master_pure_hybrid_trainer.py`.

Data yang dikumpulkan satu putaran *rollout* dipakai untuk `epochs=10` kali pembaruan
dalam putaran itu saja, dikoreksi rasio kliping PPO:

```python
ratio = torch.exp(logp - old_logp[mb])
s1 = ratio * adv_b[mb]
s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b[mb]
```

Ini **bukan** *replay* — hanya pemakaian ulang terbatas dalam satu putaran yang sama,
dengan koreksi untuk mencegah pembaruan menyimpang terlalu jauh dari kebijakan yang
menghasilkan data itu. Setelah putaran selesai, seluruh transisi dibuang.

Klaim ini berlaku untuk **varian DDPG** (`MasterHybridDDPGActor` +
`master_pure_trainer.py`), yang memang memakai *replay buffer*. Tidak berlaku untuk
varian PPO yang jadi fokus perbandingan arsitektur di dokumen ini.

*Sumber: pencarian `ReplayBuffer`/`replay_buffer` di seluruh `marl_spklu/rl/` — tidak
ditemukan di berkas Hybrid-PPO.*

---

## 5. Stabilitas penanganan masalah regresi — BERLAKU

**Klaim**: kritik menyelesaikan regresi nilai yang lebih terstruktur dan konvergen
lebih terukur dibanding ruang optimasi kebijakan yang non-konveks.

**Implementasi**: rugi kritik dan rugi kebijakan memang dua rezim optimasi yang
berbeda, dijumlahkan dengan bobot berbeda dalam satu rugi total:

```python
pi_loss  = -torch.min(s1, s2).mean()                       # non-konveks, permukaan kliping
v_loss   = nn.functional.mse_loss(value, ret_b[mb])          # regresi kuadrat, konveks lokal
ent_loss = -ent.mean()
loss = pi_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss   # vf_coef = 0,5
```

`v_loss` adalah regresi MSE murni — permukaan rugi konveks terhadap keluaran kritik.
`pi_loss` melibatkan rasio kliping dan distribusi kategorikal — permukaannya jauh
lebih tidak beraturan. Memisahkan keduanya (bukan satu rugi tunggal tercampur) adalah
apa yang membuat masing-masing bisa dianalisis dan disetel independen — `vf_coef`
mengatur seberapa besar regresi nilai ikut menekan pembaruan gabungan.

**Catatan yang perlu diperiksa**: kedua rugi ini diturunkan bersama lewat **satu**
optimizer, dan norma gradiennya dipotong **gabungan**:

```python
self.opt = torch.optim.Adam(
    list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
...
gn = nn.utils.clip_grad_norm_(
    list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
```

Karena kritik memiliki 11.275 parameter berbanding 1.476 milik aktor, kritik
mendominasi norma gabungan itu. Bila gradien kritik besar, pemotongan turut mengecilkan
gradien aktor meski gradien aktor sendiri wajar — sesuatu yang tak akan terjadi bila
keduanya punya optimizer dan pemotongan terpisah.

*Sumber: `master_pure_hybrid_trainer.py` baris 344–345 (optimizer bersama), baris
499–511 (rugi gabungan & pemotongan gradien).*

---

## 6. Pemberian umpan balik bertahap — BERLAKU, DENGAN SYARAT

**Klaim**: galat perbedaan temporal memberi koreksi tiap langkah, aktor tak perlu
menunggu akhir episode.

**Implementasi**: rekursi GAE memang bergerak mundur langkah demi langkah:

```python
for t in reversed(range(T)):
    ...
    last = delta + lam * nonterminal * last
    adv[t] = last
```

**Syaratnya**: imbalan di lingkungan ini **tertunda** — baru diemit saat
`on_charge_complete`, bisa berjam-jam setelah keputusan direkam. Transisi ke-$t{+}1$
dalam daftar terselesaikan sering *bukan* keadaan sesaat setelah transisi ke-$t$ secara
waktu-sungguhan (rata-rata ~2 langkah, maksimum 12, 45% pasangan berjarak >1 langkah,
diverifikasi empiris).

*Bootstrap* $V(s_{t+1})$ melintasi jarak sebesar itu melanggar asumsi Markov TD dan
mencemari *advantage* dengan derau struktural — pernah menyebabkan *explained variance*
anjlok ke $-300.000$ dan entropi aktor tak pernah menajam meski dilatih berlebihan pada
satu *batch* tetap (bukti diagnostik yang tercatat).

Perbaikannya, `max_step_gap=4`:

```python
if nonterminal and max_step_gap is not None and (t + 1) < T:
    gap = transitions[t + 1].step - transitions[t].step
    if gap > max_step_gap:
        nonterminal = 0.0
```

Rantai *bootstrap* **diputus** bila jaraknya melampaui 4 langkah — diperlakukan seperti
batas episode lokal, bukan dipaksa menyambung ke keadaan yang tak relevan.

Konsekuensinya: umpan baliknya bertahap **hanya dalam jendela 4 langkah**. Pada
transisi yang jaraknya lebih jauh dari itu (dan itu memang terjadi, sampai 45% dari
pasangan berjarak >1 langkah), rantainya terputus dan koreksinya kembali ke pola
episodik lokal, bukan sungguh-sungguh bertahap sepanjang lintasan.

*Sumber: `ppo.py::compute_gae`, baris 33–42 (dokumentasi diagnosis) dan baris 68–71
(implementasi pemutus rantai).*

---

## Ringkasan

| # | Klaim | Status | Kunci implementasi |
|---|---|---|---|
| 1 | Generalisasi nilai masa depan | berlaku | `delta = r - r_bar + v_next - v` |
| 2 | Reduksi variansi gradien | berlaku | normalisasi *advantage* per-aliran |
| 3 | Akses informasi lebih luas | ada, **dimatikan saat rollout** | `zero_I` vs `I_b[mb]` |
| 4 | Pemanfaatan data lama | **tidak berlaku** (PPO) | tak ada `ReplayBuffer` di jalur ini |
| 5 | Stabilitas regresi | berlaku | rugi terpisah, **optimizer bersama** |
| 6 | Umpan balik bertahap | berlaku, **berjendela 4 langkah** | `max_step_gap` |

Dua catatan lintas-butir yang layak ditulis eksplisit di bab hasil atau keterbatasan:

- **Butir 3** — ketidakcocokan $I{=}0$ (rollout) vs $I$ nyata (pembaruan) memerlukan
  keputusan: apakah ini disengaja (informasi keterlambatan memang belum tersedia saat
  keputusan diambil) atau cacat yang perlu diperbaiki. Cara memeriksa murah:
  jalankan sekali dengan `I_b` juga dinolkan pada baris pembaruan, bandingkan `v_loss`
  dan `grad_norm` terhadap versi asli.
- **Butir 5** — optimizer dan pemotongan gradien bersama berarti kapasitas kritik yang
  jauh lebih besar (§ *ARSITEKTUR.md* §4) bisa mendominasi dinamika pembaruan aktor
  secara tak langsung, meski keduanya dilatih dari rugi yang dijumlah eksplisit.

*Sumber lengkap: `marl_spklu/rl/ppo.py`, `marl_spklu/rl/master_pure_hybrid_trainer.py`,
`marl_spklu/rl/master_pure_hybrid_policy.py`, `marl_spklu/rl/master_pure_trainer.py`.*
