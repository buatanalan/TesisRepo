"""Skrip pembangun 7 notebook eksekusi (00-06) + README index. Dijalankan sekali dari root
repo: `python Eksekusi_RL/build_notebooks.py`. Notebook dibangun terprogram (bukan ditulis
manual) agar sel markdown & kode konsisten lintas notebook dan mudah diperbarui bila
Rencana_Eksekusi_Penelitian.md direvisi.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


NB_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}


def make_notebook(cells):
    return {"cells": cells, "metadata": NB_META, "nbformat": 4, "nbformat_minor": 5}


HEADER_SETUP = code("""import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
import common
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
pd.set_option("display.precision", 4)
ROOT = common.ROOT
print("ROOT:", ROOT)
print("Substrat saat ini:", common.SUBSTRAT)""")


def nav(prev_name=None, next_name=None, current="?"):
    parts = ["**Navigasi Eksekusi_RL**: "]
    links = []
    names = ["00_Bekukan_Substrat", "01_Tetapkan_Rezim", "02_Replikasi_Baseline_PDQN",
             "03_Eksperimen_Pivot", "04_Solusi_RRM_Arsitektur", "05_Robustness_Stabilitas",
             "06_Pelaporan"]
    labels = ["Tahap 0", "Tahap 1", "Tahap 2", "Tahap 3", "Tahap 4", "Tahap 5", "Tahap 6"]
    row = []
    for n, l in zip(names, labels):
        if n == current:
            row.append(f"**[{l}]**")
        else:
            row.append(f"[{l}]({n}.ipynb)")
    return md(parts[0] + " → ".join(row) +
             "\n\nDokumen rujukan: `../Dokumen_Penting/Rencana_Eksekusi_Penelitian.md` "
             "(rencana lengkap), `../Dokumen_Penting/Rumusan_Masalah_Teknis_RL.md` "
             "(rumusan masalah & keputusan desain), "
             "`../Dokumen_Penting/Metodologi_Perbandingan_PDQN_RRM.md` (aturan atribusi).")


def checklist_md(title, items):
    lines = [f"### {title}\n\n"]
    for it in items:
        lines.append(f"- [ ] {it}\n")
    return md("".join(lines))


def gate_md(text):
    return md(f"### 🚧 Gerbang\n\n{text}")


def kesimpulan_template(tahap_label):
    return md(f"""## Kesimpulan {tahap_label} (isi setelah eksekusi)

**Tanggal eksekusi**: _(isi)_
**Seed / konfigurasi**: _(isi)_

**Ringkasan hasil**:
_(isi — angka kunci dari sel di atas)_

**Status gerbang**: ✅ Lulus / ⚠ Lulus dengan catatan / ❌ Tidak lulus

**Keputusan / langkah berikut**:
_(isi)_
""")


# =============================================================================
# TAHAP 0 -- Bekukan Substrat
# =============================================================================
def build_00():
    cells = [
        md("""# Tahap 0 — Bekukan Substrat

**Tujuan**: menghilangkan cacat formulasi reward & lingkungan yang dapat mengaburkan atau
membatalkan perbandingan metode, lalu **mengunci** konfigurasi bersama. Setelah tahap ini
lulus gerbang, **tidak boleh ada perubahan substrat** — bila terjadi, seluruh tahap
berikutnya yang sudah dijalankan harus diulang.

**Sifat**: tanpa training. Seluruhnya verifikasi kode & pengukuran murah (hitungan menit)."""),
        nav(current="00_Bekukan_Substrat"),
        HEADER_SETUP,

        md("""## 0.1 Dekomposisi reward — ukur ketimpangan skala antar-suku

Baseline (sebelum perbaikan apa pun): suku global \\|mean\\| **0,3195** vs objektif sejati
(`wait`) \\|mean\\| **0,0233** → rasio **13,7×**. Target: rasio ≤ **3×**, fraksi anti-herding
aktif ≥ **50%**, fraksi reward ter-atribusi ke aksi sendiri ≥ **25%**.

Dijalankan dengan `HPPOPolicy` BELUM TERLATIH (disengaja) — mengukur sifat struktural
reward, bukan hasil optimasi."""),
        code("""summary = common.reward_decomposition(seed=0, k=2)
for k, v in summary.items():
    print(f"{k:12s} {v}")

rasio_skala = abs(summary["global"]["mean"]) / max(abs(summary["wait"]["mean"]), 1e-9)
print(f"\\nRasio |mean| suku global vs objektif sejati (wait): {rasio_skala:.1f}x  (target <= 3x)")
print(f"Fraksi anti-herding aktif: {summary['flocking']['frac_active']*100:.1f}%  (target >= 50%)")
print(f"Fraksi objektif sejati (wait) nonzero: {summary['wait']['frac_nonzero']*100:.1f}%")
print(f"Fraksi transisi UNRESOLVED (trainer episodik, tanpa filter): "
     f"{100*summary['resolusi']['unresolved']/summary['resolusi']['total']:.1f}%")"""),

        md("""**Catatan implementasi** (lihat `marl_spklu/rl/rewards.py` &
`Dokumen_Penting/Rumusan_Masalah_Teknis_RL.md` §3.2b untuk detail akar penyebab):
- Akar ketimpangan skala: `wait_scale=60` membagi objektif sejati dengan 60, sementara suku
  Gini TIDAK diskalakan sama sekali. **Perbaikan yang disarankan**: samakan orde besaran
  (mis. naikkan `alpha_wait` atau turunkan `alpha_gini`/tambah faktor skala Gini eksplisit),
  lalu ukur ulang sel di atas sampai rasio ≤3×.
- Ganti Gini **sesaat** → Gini **rolling 24 jam** pada `global_reward` (menurunkan derau
  eksogen, std sesaat terukur 0,287).
- Ganti anti-herding jendela **per-langkah** → **bergulir** (mis. `sim.recent_recs`, 24 jam).

Implementasikan perbaikan ini di `marl_spklu/rl/rewards.py` (atau turunan
`RewardCalculator` baru), lalu jalankan ulang sel di atas sampai kriteria terpenuhi."""),

        code("""# TODO setelah perbaikan rewards.py: jalankan ulang dekomposisi & catat perbandingan
# before/after di sel Kesimpulan di bagian bawah notebook ini.
# summary_after = common.reward_decomposition(seed=0, k=2)
# (bandingkan dengan `summary` di atas)"""),

        md("""## 0.2 Horizon pelatihan — DIREVISI (bukan lagi carry-forward)

**Status sebelumnya (dicabut)**: draf awal menuntut carry-forward trust lintas batas horizon
dengan alasan "reset menghapus variabel performatif". **Klaim ini ditarik** setelah
didiskusikan ulang -- lihat `Rumusan_Masalah_Teknis_RL.md` §4.1 untuk kronologi lengkap.
Ringkas: (a) kurva performativitas (Tahap 1) terbukti muncul dalam **satu pass tunggal**,
tanpa carry-forward -- beban yang menentukan, bukan reset; (b) horizon 30-hari independen
justru **konsisten** dengan protokol evaluasi S0-S3 (tiap seed = simulasi independen, trust
mulai 0,5).

Yang tersisa: horizon pendek membatasi **volume kunjungan/pengguna** (median 0 kunjungan
dalam 30 hari). Diukur di sini pada beban kanonik untuk memutuskan apakah horizon perlu
diperpanjang (60/90 hari, dataset kontinu -- BUKAN replay+carry-forward)."""),
        code("""audit = common.check_carry_forward_available()
print("Audit (catatan, bukan lagi kriteria gerbang):", audit)
print("Trainer dasar memang tidak carry-forward -- INI BUKAN CACAT, lihat 0.2 di atas.")
print()

# Generate dataset 60/90-hari sbg ARTEFAK PERMANEN di root repo (bukan scratch) --
# config INPUT dan ringkasan OUTPUT dikembalikan terpisah secara eksplisit.
config_input, output_rows = common.generate_horizon_datasets(horizon_days_list=(60, 90))

print("=== CONFIG INPUT (sama utk semua horizon, kecuali horizon_days itu sendiri) ===")
for k, v in config_input.items():
    print(f"  {k:20s} = {v}")

print()
print("=== OUTPUT per horizon (diukur, bukan ditentukan) ===")
df_visit = pd.DataFrame(output_rows)
common.save_json({"config_input": config_input, "output": output_rows}, "00_horizon_datasets.json")
display(df_visit)

print()
print(f"Horizon 30 hari (referensi): median kunjungan/user = "
     f"{df_visit.loc[df_visit.horizon_days==30,'median_kunjungan'].iloc[0]:.0f}, "
     f"frac>=5 kunjungan = {df_visit.loc[df_visit.horizon_days==30,'frac_ge5'].iloc[0]*100:.1f}%")
print(f"Horizon 90 hari: median kunjungan/user = "
     f"{df_visit.loc[df_visit.horizon_days==90,'median_kunjungan'].iloc[0]:.0f}, "
     f"frac>=5 kunjungan = {df_visit.loc[df_visit.horizon_days==90,'frac_ge5'].iloc[0]*100:.1f}%")
print()
print("Berkas dataset permanen (root repo):")
for r in output_rows:
    print(f"  {r['out_path']}  ({r['file_size_kb']} KB, {r['n_events']} event)")"""),

        md("""**Keputusan** (isi setelah melihat hasil di atas & konteks Tahap 2/3):
- Bila median kunjungan/user 30-hari dinilai cukup untuk trust bergerak pada rezim operasi
  (beban 4x sudah terbukti cukup untuk PERFORMATIVITAS di Tahap 1 -- lihat catatan di atas),
  **pertahankan horizon 30-hari** sebagai default pelatihan & evaluasi.
- Bila Tahap 2/3 menunjukkan volume data pelatihan jadi hambatan nyata (kurva belajar belum
  mendatar, terlalu sedikit update/pass), **tambahkan dataset 60/90-hari** sebagai jendela
  pelatihan tambahan -- 30-hari tetap referensi tervalidasi.

**Ukuran ketercapaian 0.2**:
- [ ] Keputusan horizon tertulis (30-hari default, atau 60/90-hari + alasan)
- [ ] Bila 60/90-hari dipilih: median kunjungan/user naik terukur dari baseline (lihat tabel)
- [ ] 30-hari tetap dilaporkan sbg referensi tervalidasi thd data riil 2024"""),

        md("""## 0.3 Jalur trainer & filter `resolved`

Konfirmasi fraksi transisi unresolved yang ikut masuk update (target: **0%** pada jalur
yang dipakai)."""),
        code("""print(f"Fraksi UNRESOLVED pada dekomposisi 0.1 (rollout_steps=penuh): "
     f"{100*summary['resolusi']['unresolved']/summary['resolusi']['total']:.2f}%")
print()
print("Keputusan: gunakan TorchContinuingTrainer (marl_spklu/rl/ppo.py) sbg jalur UTAMA --")
print("trainer ini SUDAH memisahkan resolved/pending secara eksplisit (lihat kode `train()`,")
print("baris `resolved = [t for t in agent.transitions if t.resolved]`).")
print()
print("Verifikasi willingness_ratio konsisten di seluruh jalur:")
from marl_spklu.experiments import harness
from marl_spklu.rl import training as rl_training
print("harness.DEFAULT_WILLINGNESS_RATIO =", harness.DEFAULT_WILLINGNESS_RATIO)
assert harness.DEFAULT_WILLINGNESS_RATIO is None
print("OK -- konsisten dgn substrat (None)")"""),

        md("""## 0.4 Uji alignment reward

`RewardCalculator` (default jalur H-PPO) pernah **terbukti tidak selaras** pada desain lama
(korelasi(reward, Gini) = **+0,55**, seharusnya negatif). Diuji ulang di sini pada desain
reward SAAT INI, substrat kanonik.

**Kriteria**: korelasi(reward, Gini) **negatif**, \\|ρ\\| ≥ 0,7. `greedy_util` (Gini terbaik)
harus mendapat reward tertinggi; `anti_greedy` (Gini terburuk) reward terendah."""),
        code("""rows = common.reward_alignment_test(seed=42)
df = pd.DataFrame(rows).sort_values("gini_mean")
display(df[["policy", "gini_mean", "reward_mean_local", "reward_mean_wait",
           "reward_mean_global", "reward_mean_total"]])

corr = common.reward_alignment_correlation(rows, "reward_mean_total")
print(f"\\nKorelasi(reward_total, gini_mean) = {corr:+.4f}  (target: negatif, |rho|>=0.7)")

best_gini = df.iloc[0]
worst_gini = df.iloc[-1]
print(f"\\nGini TERBAIK  ({best_gini['policy']}): reward = {best_gini['reward_mean_total']:+.4f}")
print(f"Gini TERBURUK ({worst_gini['policy']}): reward = {worst_gini['reward_mean_total']:+.4f}")
print("Urutan BENAR jika reward Gini-terbaik > reward Gini-terburuk:",
     best_gini['reward_mean_total'] > worst_gini['reward_mean_total'])"""),

        code("""fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(df["gini_mean"], df["reward_mean_total"], s=60)
for _, r in df.iterrows():
    ax.annotate(r["policy"], (r["gini_mean"], r["reward_mean_total"]),
               textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("Gini (mean, sesi)"); ax.set_ylabel("Reward rata-rata / keputusan")
ax.set_title(f"Uji alignment reward -- korelasi = {corr:+.3f} (target: negatif)")
ax.axhline(0, color="gray", lw=0.5); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(common.FIGDIR, "00_alignment_reward.png"))
plt.show()"""),

        md("""**Bila korelasi tetap positif (misalign) setelah 0.1 diperbaiki:**
diagnosis lebih lanjut disarankan (`EquityRewardCalculator` di `rewards.py` sudah
terverifikasi selaras di jalur PDQN diskrit — pertimbangkan mengadaptasi suku
REKOMENDASI-nya, $\\bar u_{feasible} - u(\\hat a)$, sebagai pengganti/pelengkap suku
global `RewardCalculator` biasa). Uji ulang sel di atas sampai kriteria terpenuhi — jangan
lanjut ke Tahap 1 dengan reward yang belum selaras."""),

        code("""common.save_json(rows, "00_alignment_test.json")
common.save_json(summary, "00_reward_decomposition.json")
print("Tersimpan ke Eksekusi_RL/outputs/")"""),

        checklist_md("Ukuran ketercapaian Tahap 0 (ringkasan)", [
            "Rasio |mean| antar-suku reward utama ≤ 3× (0.1)",
            "Fraksi transisi anti-herding aktif ≥ 50% (0.1)",
            "Varians reward ter-atribusi ke aksi sendiri naik dari 5,5% → ≥25% (0.1)",
            "Keputusan horizon pelatihan tertulis: 30-hari default atau 60/90-hari + alasan (0.2)",
            "Fraksi transisi unresolved yang ikut update = 0% pada jalur terpilih (0.3)",
            "`willingness_ratio=None` terkonfirmasi seluruh jalur (0.3)",
            "Korelasi(reward, Gini) negatif, |ρ|≥0,7 (0.4)",
            "F1 Verification tetap 59/59 & suite repo 38/38 (jalankan `pytest` terpisah)",
        ]),
        gate_md("**Tidak melanjutkan ke Tahap 1** sebelum seluruh kotak di atas tercentang. "
               "Bila 0.4 gagal setelah 0.1 diperbaiki: berhenti dan diagnosis reward -- "
               "melatih di atas reward yang salah arah menjamin hasil yang tidak dapat "
               "ditafsirkan."),
        kesimpulan_template("Tahap 0"),
    ]
    return make_notebook(cells)


# =============================================================================
# TAHAP 1 -- Tetapkan Rezim
# =============================================================================
def build_01():
    cells = [
        md("""# Tahap 1 — Tetapkan Rezim Eksperimen

**Tujuan ganda**: (a) menentukan beban operasi eksperimen secara TERUKUR, bukan intuitif;
(b) menghasilkan **kurva kekuatan loop performatif vs beban** — hasil tesis tersendiri.

**Prasyarat**: Tahap 0 sudah lulus gerbang (substrat dibekukan)."""),
        nav(current="01_Tetapkan_Rezim"),
        HEADER_SETUP,

        md("""## 1.1 Sapuan performativitas multi-seed

Sensitivitas diukur dengan membungkus `GreedyAgent`: EstWait yang **ditampilkan** dikalikan
bias $b$, sementara rekomendasi (pilihan SPKLU) tetap identik — mengisolasi jalur
*kualitas janji → trust* dari jalur alokasi.

Hipotesis awal (1 seed, sesi analisis sebelumnya):

| Beban | wait (mnt) | rentang trust (b: 0,25→2) | % trust bergerak |
|---|---|---|---|
| 1× | 2,50 | 0,0020 | 23,4% |
| 4× | 23,71 | 0,0217 | 52,1% |
| 8× | 818,67 (kolaps) | 0,0322 | 66,3% |

**PERINGATAN BIAYA**: sel di bawah menjalankan `n_beban × n_bias × n_seed` episode penuh
30-hari. Dengan default (5 beban × 3 bias × 5 seed = 75 run), perkirakan waktu total
sebelum menjalankan (uji 1 run dulu di sel berikutnya)."""),

        code("""import time
t0 = time.time()
_ = common.performativity_sensitivity(common.DATASET_KANONIK, 1.0, seed=42)
dt = time.time() - t0
print(f"1 run (beban 1x): {dt:.1f} detik")
print(f"Estimasi sapuan penuh (5 beban x 3 bias x 5 seed = 75 run, beban tinggi lebih lambat "
     f"krn lebih banyak sesi): kasar ~{dt*75*1.5/60:.1f} menit -- SESUAIKAN N_SEED bila perlu.")"""),

        code("""N_SEED = 5   # turunkan ke 2-3 dulu utk uji cepat, naikkan ke >=5 utk hasil final
LOADS = [1.0, 2.0, 4.0, 6.0, 8.0]
BIASES = [0.25, 1.0, 2.0]

rows = common.performativity_sweep(LOADS, biases=BIASES, seeds=range(N_SEED))
df = pd.DataFrame(rows)
common.save_json(rows, "01_performativity_sweep.json")
df.head()"""),

        code("""# Agregasi: rentang trust (max-min lintas bias) per beban, per seed -> lalu CI 95% antar-seed
def rentang_per_seed(g):
    return g.groupby("bias")["trust_mean"].mean().max() - g.groupby("bias")["trust_mean"].mean().min()

agg = []
for lm in LOADS:
    sub = df[df["load_multiplier"] == lm]
    rentangs = [rentang_per_seed(sub[sub["seed"] == s]) for s in sub["seed"].unique()]
    rentangs = np.array(rentangs)
    ci = 1.96 * rentangs.std(ddof=1) / np.sqrt(len(rentangs)) if len(rentangs) > 1 else 0.0
    wait_mean = sub[sub["bias"] == 1.0]["wait_mean"].mean()
    agg.append(dict(load_multiplier=lm, rentang_trust_mean=rentangs.mean(),
                    rentang_trust_ci95=ci, wait_mean_b1=wait_mean,
                    pct_trust_bergerak=sub[sub["bias"]==1.0]["trust_moved_frac"].mean()*100))
agg_df = pd.DataFrame(agg)
display(agg_df)

rentang_1x = agg_df.loc[agg_df.load_multiplier == 1.0, "rentang_trust_mean"].iloc[0]
rentang_4x = agg_df.loc[agg_df.load_multiplier == 4.0, "rentang_trust_mean"].iloc[0] \\
    if 4.0 in LOADS else None
if rentang_4x is not None:
    print(f"\\nRasio rentang 4x/1x = {rentang_4x/max(rentang_1x,1e-9):.1f}x  (target >= 5x)")"""),

        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].errorbar(agg_df["load_multiplier"], agg_df["rentang_trust_mean"],
                 yerr=agg_df["rentang_trust_ci95"], marker="o", capsize=4)
axes[0].set_xlabel("load_multiplier"); axes[0].set_ylabel("Rentang trust (bias 0,25->2)")
axes[0].set_title("Kekuatan loop performatif vs beban"); axes[0].grid(alpha=0.3)

axes[1].plot(agg_df["load_multiplier"], agg_df["wait_mean_b1"], marker="s", color="tab:red")
axes[1].set_xlabel("load_multiplier"); axes[1].set_ylabel("wait mean (menit), bias=1x")
axes[1].set_title("Kongesti vs beban"); axes[1].grid(alpha=0.3); axes[1].set_yscale("log")
fig.tight_layout()
fig.savefig(os.path.join(common.FIGDIR, "01_kurva_performativitas_vs_beban.png"))
plt.show()"""),

        md("""**Ukuran ketercapaian 1.1**:
- [ ] Rentang trust pada beban terpilih ≥5× lipat rentang pada 1×, CI 95% tidak tumpang tindih
- [ ] wait mean pada beban terpilih berada di 10–60 menit (bukan <5 = tak ada kongesti,
      bukan >120 = kolaps)
- [ ] Setiap sel ≥5 seed"""),

        md("""## 1.1b Bukti non-stasionaritas sejati: drift temporal

Trajektori `GreedyAgent` jujur pada beban 1× (rezim referensi) dan rezim operasi kandidat.
Kriteria: acceptance **datar** pada 1×, **turun monoton** pada rezim operasi."""),
        code("""REZIM_KANDIDAT = 4.0   # sesuaikan berdasar hasil 1.1
ds_kandidat = common.DATASET_KANONIK if REZIM_KANDIDAT == 1.0 else \\
    common.generate_load_dataset(REZIM_KANDIDAT)

traj_1x = common.performativity_trajectory(common.DATASET_KANONIK, seed=42)
traj_kandidat = common.performativity_trajectory(ds_kandidat, seed=42)

df1 = pd.DataFrame(traj_1x); df1["rezim"] = "1x"
dfk = pd.DataFrame(traj_kandidat); dfk["rezim"] = f"{REZIM_KANDIDAT}x"
traj_all = pd.concat([df1, dfk], ignore_index=True)
common.save_json(traj_all.to_dict("records"), "01_trajektori_drift.json")
display(traj_all)"""),

        code("""fig, ax = plt.subplots(figsize=(6.5, 4.5))
for rezim, g in traj_all.groupby("rezim"):
    ax.plot(g["hari"], g["acceptance_kumulatif"], marker="o", label=f"acceptance ({rezim})")
ax.set_xlabel("hari"); ax.set_ylabel("acceptance kumulatif")
ax.set_title("Drift acceptance: datar (1x) vs menurun monoton (rezim operasi)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(common.FIGDIR, "01_drift_acceptance.png"))
plt.show()

slope_1x = np.polyfit(df1["hari"], df1["acceptance_kumulatif"], 1)[0]
slope_k = np.polyfit(dfk["hari"], dfk["acceptance_kumulatif"], 1)[0]
print(f"Slope acceptance 1x: {slope_1x:+.5f}/hari (target: ~0, datar)")
print(f"Slope acceptance {REZIM_KANDIDAT}x: {slope_k:+.5f}/hari (target: negatif jelas)")"""),

        md("""## 1.2 Pilih & bekukan rezim operasi

Isi setelah 1.1 & 1.1b dievaluasi terhadap kriteria."""),
        code("""REZIM_OPERASI = REZIM_KANDIDAT  # ubah bila kandidat lain lebih sesuai kriteria
common.SUBSTRAT["rezim_operasi_load_multiplier"] = REZIM_OPERASI
common.SUBSTRAT["rezim_referensi_load_multiplier"] = 1.0
print("Rezim operasi dibekukan:", REZIM_OPERASI)
print("Framing: load_multiplier = skenario PENETRASI EV masa depan, bukan rekalibrasi -- "
     "parameter perilaku (LCMNL, trust, SoC) TIDAK berubah, hanya intensitas kedatangan.")"""),

        md("""## 1.3 Karakterisasi baseline non-RL di rezim operasi

S0–S3 via `harness.py`, ≥10 seed. **`greedy_util` wajib disertakan** (PDQN pernah kalah
signifikan darinya, p=0,0039 — lihat `Metodologi_Perbandingan_PDQN_RRM.md` §1.1)."""),
        code("""ds_operasi = common.generate_load_dataset(REZIM_OPERASI) if REZIM_OPERASI != 1.0 \\
    else common.DATASET_KANONIK

results = common.run_all_baselines(dataset_path=ds_operasi, seeds=range(10))
rows_baseline = []
for name, agg in results.items():
    row = {"skenario": name, "gini_mean": agg["gini_mean"]["mean"],
          "gini_mean_std": agg["gini_mean"]["std"],
          "wait_mean": agg["wait_mean"], "acceptance": agg["acceptance_overall"]["mean"],
          "trust_mean": agg["trust_mean"], "herding": agg["herding_index"]["mean"],
          "wait_by_compliance_ratio": agg["wait_by_compliance"]["ratio"]}
    rows_baseline.append(row)
baseline_df = pd.DataFrame(rows_baseline)
common.save_json(rows_baseline, "01_baseline_S0-S3.json")
display(baseline_df)"""),

        md("""**S2 (diversified) adalah pembanding penentu** — memisahkan kontribusi *mekanisme*
(estimasi jujur + diversifikasi) dari kontribusi *kebijakan yang dipelajari*. Klaim nilai RL
harus dibandingkan terhadap S2, bukan hanya S0."""),

        checklist_md("Ukuran ketercapaian Tahap 1 (ringkasan)", [
            "Rentang trust rezim operasi ≥5× rentang 1×, CI tak tumpang tindih",
            "wait mean rezim operasi di rentang 10-60 menit",
            "Acceptance turun monoton di rezim operasi, datar di 1×",
            "Rezim operasi dibekukan & dicatat di `common.SUBSTRAT`",
            "Tabel S0-S3 lengkap (termasuk greedy_util) tersedia di kedua rezim",
        ]),
        gate_md("""**Cabang 1** (diharapkan): kriteria 1.1 & 1.1b terpenuhi → lanjut ke Tahap 2
dengan rezim operasi terkonfirmasi.

**Cabang 2** (hasil negatif, tetap sah): loop tidak menguat signifikan di beban mana pun yang
diuji → **konsekuensi**: klaim tesis bergeser dari *"RRM diperlukan"* menjadi
*"performativitas dapat diabaikan pada rentang beban realistis untuk konfigurasi klaster
ini"*. **Putuskan cabang ini sebelum menjalankan Tahap 2** — bila Cabang 2 terjadi, Tahap 2-4
tetap dapat dijalankan sebagai eksplorasi tambahan, tetapi kurva Tahap 1 menjadi kontribusi
utama, bukan prasyarat solusi."""),
        kesimpulan_template("Tahap 1"),
    ]
    return make_notebook(cells)


# =============================================================================
# TAHAP 2 -- Replikasi Baseline PDQN (titik A)
# =============================================================================
def build_02():
    cells = [
        md("""# Tahap 2 — Replikasi Baseline PDQN di Substrat Final (titik A)

**Tujuan**: membuktikan PDQN **sehat dan kompeten** di substrat final, trust STATIS
(`constant_trust()` aktif). Ini BUKAN mencari hasil baru — ini asuransi terhadap tuduhan
*strawman*: tanpa tahap ini, kegagalan PDQN di Tahap 3 dapat dituduh sebagai implementasi
yang buruk.

**Preseden** (`archive/docs/LAPORAN_IMPLEMENTASI_PDQN.md` Bagian 4, dataset 7-hari/8 SPKLU):
PDQN > Greedy-queue signifikan pada µ̂≥0,5 (p=0,0098 & 0,0020); **PDQN < Greedy-util
signifikan pada µ̂=0,8 (p=0,0039)** — dilaporkan apa adanya, bukan disembunyikan.

**Prasyarat**: Tahap 0 & 1 lulus gerbang. Rezim operasi & substrat sudah dibekukan."""),
        nav(current="02_Replikasi_Baseline_PDQN"),
        HEADER_SETUP,

        md("""## 2.0 Konfigurasi (dibekukan sebelum uji, mengikuti preseden §4.1 arsip)

Isi sel di bawah dengan konfigurasi final, lalu JANGAN diubah lagi setelah run dimulai."""),
        code("""CONFIG_TAHAP2 = dict(
    dataset_path=None,  # diisi dari common.SUBSTRAT["rezim_operasi_load_multiplier"], lihat di bawah
    mu_hat=0.8,          # sapuan bisa ditambah: [0.2, 0.5, 0.8] mengikuti preseden arsip
    n_train_seed=3,      # >=3, preseden arsip
    n_eval_seed=10,      # >=10, preseden arsip
    anggaran_chunk=100,  # potongan training -- akan DIPAKAI IDENTIK di Tahap 3 & 4
    trust_mode="constant_trust",  # trust_value default 0.5 (lihat ablations.py)
)

rezim_op = common.SUBSTRAT.get("rezim_operasi_load_multiplier", 1.0)
CONFIG_TAHAP2["dataset_path"] = (common.DATASET_KANONIK if rezim_op == 1.0
                                 else common.generate_load_dataset(rezim_op))
print(CONFIG_TAHAP2)
common.save_json(CONFIG_TAHAP2, "02_config_beku.json")
print("\\n[PENTING] anggaran_chunk & seluruh hyperparameter di sini WAJIB dipakai IDENTIK")
print("di Tahap 3 (titik B) dan Tahap 4 (titik E) -- lihat Lapisan 2 (kesetaraan baseline)")
print("di Metodologi_Perbandingan_PDQN_RRM.md.")"""),

        md("""## 2.1 Training PDQN (titik A) — kerangka

Sel ini SKELETON siap-jalan, memakai infrastruktur yang sudah ada
(`PDQNContinuousTrainer` atau varian diskrit sesuai `spesifikasi_teknis_pdqn_baseline.md`).
**Belum dieksekusi penuh di sini** (anggaran waktu training nyata) — jalankan di luar
notebook (skrip terpisah / background) lalu muat hasilnya kembali di sel 2.2."""),
        code("""# from marl_spklu.experiments.ablations import constant_trust
# from marl_spklu.rl.pdqn_continuous_trainer import PDQNContinuousTrainer
# from marl_spklu.rl.forecaster import FormulaForecaster
#
# hasil_per_seed = []
# with common.frozen_trust(value=0.5):
#     for train_seed in range(CONFIG_TAHAP2["n_train_seed"]):
#         tr = PDQNContinuousTrainer(CONFIG_TAHAP2["dataset_path"], k=2,
#                                    rollout_steps=96, seed=train_seed)
#         policy, trace = tr.train(FormulaForecaster(), n_updates=CONFIG_TAHAP2["anggaran_chunk"])
#         # simpan checkpoint: torch.save(policy.state_dict(), f"Eksekusi_RL/outputs/pdqn_A_seed{train_seed}.pt")
#         hasil_per_seed.append(dict(train_seed=train_seed, trace=trace))
#
# common.save_json(hasil_per_seed, "02_training_A_trace.json")
print("[TODO] Jalankan training sesungguhnya di luar notebook interaktif (durasi non-trivial).")
print("Kerangka kode tersedia di atas (dikomentari) -- aktifkan setelah dataset/anggaran final.")"""),

        md("""## 2.2 Evaluasi & uji statistik

Muat checkpoint hasil 2.1, evaluasi pada ≥10 seed, uji Wilcoxon berpasangan vs
Greedy-queue DAN Greedy-util."""),
        code("""# from marl_spklu.rl.rollout import evaluate_policy, InferenceAgent
# from scipy.stats import wilcoxon
#
# def eval_pdqn_vs_baseline(policy, baseline_agent_factory, dataset_path, n_eval_seed=10):
#     pdqn_ginis, base_ginis = [], []
#     for s in range(n_eval_seed):
#         sim_p = common.fresh_sim(dataset_path)
#         random.seed(s); np.random.seed(s)
#         res_p = evaluate_policy(sim_p, policy, FormulaForecaster(), k=2)
#         pdqn_ginis.append(res_p["gini_served"])
#
#         sim_b = common.fresh_sim(dataset_path)
#         random.seed(s); np.random.seed(s)
#         sim_b.run(max_steps=sim_b.max_steps, agent=baseline_agent_factory())
#         served = np.array([sp.total_served for sp in sim_b.spklus.values()], float)
#         base_ginis.append(common.gini(served))
#     stat, p = wilcoxon(base_ginis, pdqn_ginis)  # H0: median selisih = 0
#     return dict(pdqn=pdqn_ginis, baseline=base_ginis, wilcoxon_p=p)
print("[TODO] Isi setelah checkpoint 2.1 tersedia.")"""),

        checklist_md("Ukuran ketercapaian Tahap 2", [
            "PDQN signifikan mengungguli Greedy-queue, Wilcoxon p < 0,05",
            "Spread antar train-seed independen ≤ 0,005 poin Gini",
            "explained_variance kritik > 0,1 DAN naik selama training",
            "Kurva belajar mendatar di akhir anggaran (bukan under-training)",
            "Hasil vs greedy_util dilaporkan apa adanya (menang/kalah/tak beda)",
            "Anggaran pelatihan dicatat & akan dipakai identik di Tahap 3-4",
        ]),
        gate_md("**Tidak melanjutkan ke Tahap 3** bila PDQN tidak mengungguli Greedy-queue di "
               "substrat final. Diagnosis dulu: anggaran kurang, hyperparameter belum sesuai "
               "substrat baru, atau explained_variance≈0 (kritik gagal fit)."),
        kesimpulan_template("Tahap 2"),
    ]
    return make_notebook(cells)


# =============================================================================
# TAHAP 3 -- Eksperimen Pivot (titik B, A')
# =============================================================================
def build_03():
    cells = [
        md("""# Tahap 3 — Eksperimen Pivot: Performativitas Merusak PDQN? (titik B, A′)

**Tujuan**: membangun *kebutuhan* akan RRM. Satu faktor, dan hanya satu faktor, divariasikan:
`constant_trust()` ON vs OFF. Arsitektur, hyperparameter, dan anggaran pelatihan **identik**
dengan Tahap 2.

**Tiga lengan wajib**:

| Lengan | Kondisi trust | Fungsi |
|---|---|---|
| **A** | `constant_trust(0,5)` | Titik acuan (= Tahap 2) |
| **B** | Dinamis penuh | Kondisi yang diteliti |
| **A′** | `constant_trust(≈level-akhir-B)` | **Kontrol penentu** |

**Mengapa A′ wajib**: trust rata-rata pada lengan B **bergerak turun** (terukur pada
GreedyAgent, rezim 4×: 0,4958→0,4802). Tanpa A′, tidak bisa dibedakan apakah kegagalan
PDQN disebabkan trust yang *berubah*, atau semata trust yang *lebih rendah*.

**Prasyarat**: Tahap 2 lulus gerbang (titik A sehat & terdokumentasi)."""),
        nav(current="03_Eksperimen_Pivot"),
        HEADER_SETUP,

        md("""## 3.0 Tentukan level trust akhir lengan B (untuk kalibrasi A′)

Sebelum training PDQN penuh, ukur ke mana trust rata-rata konvergen di bawah kebijakan
sederhana (GreedyAgent) pada rezim operasi — dipakai sebagai nilai `constant_trust()` A′.
Ini BUKAN nilai final (PDQN mungkin berbeda), tapi estimasi awal yang murah."""),
        code("""ds_operasi = (common.DATASET_KANONIK
             if common.SUBSTRAT.get("rezim_operasi_load_multiplier", 1.0) == 1.0
             else common.generate_load_dataset(common.SUBSTRAT["rezim_operasi_load_multiplier"]))

traj = common.performativity_trajectory(ds_operasi, seed=42)
trust_akhir_estimasi = traj[-1]["trust_mean"] if traj else 0.48
print(f"Estimasi trust akhir (GreedyAgent, rezim operasi): {trust_akhir_estimasi:.4f}")
print("Gunakan nilai ini sbg titik awal utk constant_trust() lengan A' -- "
     "SESUAIKAN setelah training B (titik B) selesai & trust akhir SESUNGGUHNYA "
     "(di bawah PDQN, bukan Greedy) terukur.")"""),

        md("""## 3.1 Training tiga lengan — kerangka

Identik dengan Tahap 2.1, hanya konteks trust yang berbeda per lengan. **Anggaran & seed
HARUS identik dengan `CONFIG_TAHAP2`** (dimuat dari `02_config_beku.json`)."""),
        code("""config = json.load(open(os.path.join(common.OUTDIR, "02_config_beku.json"))) \\
    if os.path.exists(os.path.join(common.OUTDIR, "02_config_beku.json")) else None
print("Konfigurasi Tahap 2 (harus dipakai identik):", config)

# Lengan A -- SAMA dgn Tahap 2, tinggal pakai ulang checkpoint/hasilnya, TIDAK perlu dilatih ulang.
#
# Lengan B (dinamis penuh):
# with common.frozen_trust(value=None):  # None/tanpa context manager = trust dinamis default
#     for train_seed in range(config["n_train_seed"]):
#         tr = PDQNContinuousTrainer(config["dataset_path"], k=2, rollout_steps=96, seed=train_seed)
#         policy_B, trace_B = tr.train(FormulaForecaster(), n_updates=config["anggaran_chunk"])
#         # PENTING: pastikan dataset_path pakai horizon yg diputuskan di Tahap 0.2
#         # (30-hari default, atau 60/90-hari bila volume data terbukti kurang).
#         # TIDAK perlu carry-forward -- horizon independen sudah cukup (lihat 0.2).
#
# Lengan A' (statis di level akhir B):
# with common.frozen_trust(value=trust_akhir_terukur_dari_B):
#     for train_seed in range(config["n_train_seed"]):
#         tr = PDQNContinuousTrainer(config["dataset_path"], k=2, rollout_steps=96, seed=train_seed)
#         policy_Ap, trace_Ap = tr.train(FormulaForecaster(), n_updates=config["anggaran_chunk"])
print("[TODO] Jalankan training B & A' di luar notebook interaktif dgn anggaran IDENTIK Tahap 2.")"""),

        md("""## 3.2 Evaluasi tiga lengan & uji statistik

Kriteria: **B signifikan lebih buruk dari A** (p<0,05) **DAN** **A′ tidak sama buruknya
dengan B** (selisih B−A′ signifikan)."""),
        code("""# rows = []
# for lengan, policy in [("A", policy_A), ("B", policy_B), ("A_prime", policy_Ap)]:
#     for eval_seed in range(config["n_eval_seed"]):
#         res = evaluate_policy(common.fresh_sim(config["dataset_path"]), policy, ...)
#         rows.append(dict(lengan=lengan, eval_seed=eval_seed, gini=res["gini_served"]))
# df = pd.DataFrame(rows)
#
# from scipy.stats import wilcoxon
# p_BA  = wilcoxon(df[df.lengan=="A"]["gini"], df[df.lengan=="B"]["gini"]).pvalue
# p_BAp = wilcoxon(df[df.lengan=="A_prime"]["gini"], df[df.lengan=="B"]["gini"]).pvalue
# print(f"B vs A   : p={p_BA:.4f}  (target <0.05, B lebih buruk)")
# print(f"B vs A'  : p={p_BAp:.4f} (target <0.05 JUGA -- A' harus beda nyata dari B)")
print("[TODO] Isi setelah checkpoint 3.1 tersedia.")""")
        ,
        code("""fig, ax = plt.subplots(figsize=(6, 4.5))
# ax.boxplot([df[df.lengan==l]["gini"] for l in ["A","A_prime","B"]], labels=["A (statis-awal)","A' (statis-akhir-B)","B (dinamis)"])
ax.set_ylabel("Gini")
ax.set_title("Tahap 3 -- perbandingan tiga lengan (isi setelah data tersedia)")
fig.tight_layout()
fig.savefig(os.path.join(common.FIGDIR, "03_tiga_lengan.png"))
plt.show()"""),

        checklist_md("Ukuran ketercapaian Tahap 3", [
            "B signifikan lebih buruk dari A (Wilcoxon p<0,05) pada gini_mean",
            "A' TIDAK sama buruknya dengan B (selisih B-A' signifikan)",
            "Ukuran efek dilaporkan (selisih Gini absolut + Cohen's d), bukan hanya p-value",
            "Kurva belajar lengan B mendatar di akhir",
            "Horizon lengan B sesuai keputusan Tahap 0.2 (30-hari default / 60-90-hari bila diperlukan)",
            "Diagnostik: explained_variance lengan B diperiksa apakah turun sepanjang training",
        ]),
        gate_md("""**Cabang 1** (diharapkan) — B≪A signifikan DAN A′≉B: performativitas terbukti
merusak, dan terbukti karena dinamikanya (bukan levelnya). Lanjut ke Tahap 4 dengan klaim kuat.

**Cabang 2** (hasil negatif, tetap sah) — B≈A: performativitas TIDAK merusak PDQN pada rezim
ini. **Laporkan apa adanya.** Konsekuensi: klaim bergeser ke *"performativitas dapat
ditoleransi PDQN sampai intensitas X"*, dengan kurva Tahap 1 sebagai kontribusi utama.
Tahap 4 tetap dapat dijalankan sebagai eksplorasi *"apakah RRM tetap bermanfaat meski tidak
diperlukan"*, bukan sebagai solusi atas kegagalan.

**Putuskan kriteria kedua cabang ini SEBELUM menjalankan training** (bukan setelah melihat
hasil)."""),
        kesimpulan_template("Tahap 3"),
    ]
    return make_notebook(cells)


# =============================================================================
# TAHAP 4 -- Solusi (E -> C, D)
# =============================================================================
def build_04():
    cells = [
        md("""# Tahap 4 — Solusi: RRM dan Arsitektur, Ter-atribusi

**Tujuan**: menunjukkan solusi bekerja, DAN dapat mengatribusikan perbaikan kepada
penyebabnya. **Mulai dari E** (RRM saja, arsitektur identik dengan B) — klaim paling bersih
dengan biaya paling murah. Tambahkan C (arsitektur) dan D (arsitektur+RRM) HANYA bila ingin
mengklaim perubahan arsitektur.

**Tangga ablasi**: A (Tahap 2) · B, A′ (Tahap 3) · **E** = B+RRM, arsitektur identik ·
C (opsional) = B+arsitektur baru, tanpa RRM · D (opsional) = C+RRM.

**Prasyarat**: Tahap 3 selesai (kedua cabang gerbang punya jalur ke sini)."""),
        nav(current="04_Solusi_RRM_Arsitektur"),
        HEADER_SETUP,

        md("""## 4.1 Titik E — RRM saja, arsitektur identik dengan B

`train_rrm()` sudah tersedia di `PDQNContinuousTrainer` (siklus freeze→retrain). Anggaran
pelatihan **harus identik** dengan B — bila fase freeze menambah langkah, berikan anggaran
yang sama kepada B juga (ulangi Tahap 3 bila perlu)."""),
        code("""config = json.load(open(os.path.join(common.OUTDIR, "02_config_beku.json"))) \\
    if os.path.exists(os.path.join(common.OUTDIR, "02_config_beku.json")) else None

# from marl_spklu.rl.pdqn_continuous_trainer import PDQNContinuousTrainer
# from marl_spklu.rl.forecaster import FormulaForecaster
#
# hasil_E = []
# for train_seed in range(config["n_train_seed"]):
#     tr = PDQNContinuousTrainer(config["dataset_path"], k=2, rollout_steps=96, seed=train_seed)
#     policy_E, trace_E = tr.train_rrm(FormulaForecaster(), n_rounds=5,
#                                      freeze_chunks=15, retrain_chunks=10)
#     hasil_E.append(dict(train_seed=train_seed, trace=trace_E))
# common.save_json(hasil_E, "04_training_E_trace.json")
print("[TODO] Jalankan train_rrm() dgn anggaran total SEBANDING config['anggaran_chunk'].")
print("Preseden arsip (dataset sulit terpisah): 3 seed x 5 ronde x [15 freeze+10 retrain] "
     "chunk, PDQN kontinu RRM unggul 3/3 seed tanpa tumpang tindih vs Greedy-util "
     "(0,0666+-0,0016 vs 0,0737+-0,0013).")"""),

        md("""## 4.2 Evaluasi E vs B — kriteria utama

Kriteria: **E signifikan lebih baik dari B** (p<0,05), **memulihkan ≥50% selisih (B−A)**."""),
        code("""# rows_E = [...]  # evaluasi sama pola dgn Tahap 3.2
# gini_A = df[df.lengan=="A"]["gini"].mean()
# gini_B = df[df.lengan=="B"]["gini"].mean()
# gini_E = np.mean([r["gini"] for r in rows_E])
#
# selisih_BA = gini_B - gini_A          # kerusakan akibat performativitas
# pemulihan_E = gini_B - gini_E          # seberapa banyak E memulihkan
# frac_pulih = pemulihan_E / selisih_BA if selisih_BA else float("nan")
# print(f"Selisih (B-A) = {selisih_BA:.4f}  (kerusakan performativitas)")
# print(f"Pemulihan oleh E = {pemulihan_E:.4f}  ({100*frac_pulih:.1f}% dari kerusakan, target >=50%)")
print("[TODO] Isi setelah checkpoint E & B tersedia.")"""),

        md("""## 4.3 (Opsional) Titik C — arsitektur saja, tanpa RRM

**Hanya wajib bila Anda ingin mengklaim perubahan arsitektur.** Setiap perubahan harus
punya justifikasi dari diagnosis (bukan coba-coba), dan perubahan yang sifatnya tuning umum
(jumlah layer, aktivasi) **wajib diberikan juga ke lengan B** agar bukan keuntungan sepihak.
"""),
        code("""# Contoh kerangka: PDQNContinuousPolicy dgn modifikasi arsitektur (mis. hidden_dim lebih
# besar, atau modul rekuren tambahan utk melacak drift trust -- JUSTIFIKASI: non-stasionaritas
# trust terukur di Tahap 1/3, modul ini dirancang utk melacaknya).
#
# policy_C_arch = PDQNContinuousPolicy(obs_dim, critic_obs_dim, N, hidden=256)  # contoh
# ... training C dgn trust DINAMIS (spt B), arsitektur baru, anggaran identik ...
print("[TODO -- opsional] Isi hanya bila arsitektur akan diklaim sbg kontribusi.")"""),

        md("""## 4.4 (Opsional) Titik D — arsitektur + RRM"""),
        code("""# policy_D = ... (arsitektur C) + train_rrm() ...
print("[TODO -- opsional] Isi hanya bila 4.3 dijalankan.")"""),

        md("""## 4.5 Atribusi & interpretasi"""),
        code("""# selisih_CB = gini_B - gini_C   # kontribusi arsitektur saja
# selisih_DC = gini_C - gini_D   # kontribusi RRM DI ATAS arsitektur baru
# selisih_EB = gini_B - gini_E   # kontribusi RRM BERDIRI SENDIRI (dari 4.2)
#
# print(f"(E-B) RRM berdiri sendiri     : {selisih_EB:+.4f}")
# print(f"(C-B) arsitektur saja         : {selisih_CB:+.4f}")
# print(f"(D-C) RRM di atas arsitektur  : {selisih_DC:+.4f}")
#
# if selisih_EB > 0 and selisih_DC > 0 and abs(selisih_EB - selisih_DC) < 0.3 * max(selisih_EB, selisih_DC):
#     print("=> RRM berkontribusi INDEPENDEN dari arsitektur -- klaim TERKUAT: "
#          "'RRM memulihkan kegagalan akibat performativitas'")
# elif selisih_EB <= 0 and selisih_DC > 0:
#     print("=> RRM hanya berguna BERSAMA arsitektur baru -- klaim: 'RRM + arsitektur X', bukan 'RRM' saja")
# elif selisih_EB > 0 and selisih_DC <= 0:
#     print("=> arsitektur sudah menyerap manfaat RRM -- laporkan apa adanya")
print("[TODO] Isi setelah 4.1-4.4 tersedia.")"""),

        checklist_md("Ukuran ketercapaian Tahap 4", [
            "E signifikan lebih baik dari B (p<0,05)",
            "E memulihkan >=50% selisih (B-A)",
            "Anggaran pelatihan E identik dengan B",
            "Arsitektur & hyperparameter E identik dengan B (nol perubahan selain skema training)",
            "Konsisten pada >=3 seed pelatihan, arah kemenangan sama di semua seed",
            "(bila C/D dijalankan) setiap perubahan arsitektur punya justifikasi tertulis dari diagnosis",
            "(bila C/D dijalankan) perubahan tuning umum juga diberikan ke lengan B",
        ]),
        gate_md("**Bila E tidak mengungguli B**: laporkan apa adanya, periksa dulu apakah "
               "penyebabnya implementasi (mis. panjang fase freeze belum cukup) sebelum "
               "menyimpulkan RRM tidak efektif. JANGAN menutupinya dengan menambah "
               "perubahan arsitektur lalu melaporkan selisih gabungan (D-B) tanpa memisahkan."),
        kesimpulan_template("Tahap 4"),
    ]
    return make_notebook(cells)


# =============================================================================
# TAHAP 5 -- Robustness & Stabilitas
# =============================================================================
def build_05():
    cells = [
        md("""# Tahap 5 — Robustness & Stabilitas

**Tujuan**: memastikan hasil bukan artefak seed, konfigurasi, atau rezim tunggal.

**Prasyarat**: Tahap 4 punya minimal titik E terselesaikan (jalur minimum) atau tangga
penuh A–E (+C,D opsional)."""),
        nav(current="05_Robustness_Stabilitas"),
        HEADER_SETUP,

        md("""## 5.1 Multi-seed penuh

Seluruh klaim final: ≥5 seed pelatihan × ≥10 seed evaluasi. CV lintas seed dilaporkan
untuk setiap metrik (preseden: S0 CV = 0,153 — varians antar-seed bisa melebihi selisih
antar-metode yang diharapkan)."""),
        code("""# rows_final = []  # gabungkan seluruh lengan yg relevan (A, B, E, dan C/D bila ada)
# for lengan, policy in lengan_policies.items():
#     for eval_seed in range(10):
#         res = evaluate_policy(...)
#         rows_final.append(dict(lengan=lengan, eval_seed=eval_seed, gini=res["gini_served"], ...))
# df_final = pd.DataFrame(rows_final)
#
# cv_per_lengan = df_final.groupby("lengan")["gini"].agg(lambda x: x.std()/x.mean())
# print(cv_per_lengan)
print("[TODO] Isi setelah seluruh checkpoint Tahap 2-4 tersedia.")"""),

        md("""## 5.2 Verifikasi performative stability

Bekukan kebijakan terlatih (lengan terbaik, mis. E), jalankan beberapa pass TANPA update.
Kriteria: Gini/trust/acceptance **menyetimbang** (bentuk *saturating*), bukan terus
berosilasi. Titik setimbang tidak boleh bergantung pada trust awal."""),
        code("""# from marl_spklu.experiments.ablations import initial_trust
#
# for trust0 in [0.5, 0.7]:
#     with initial_trust(trust0):
#         sim = common.fresh_sim(ds_operasi)
#         # jalankan beberapa pass TANPA ppo.update -- policy dibekukan (eval mode)
#         # catat gini/trust/acceptance per pass
#         ...
print("[TODO] Isi setelah kebijakan final (Tahap 4) tersedia. Preseden arsip: kedua titik "
     "awal trust (0,5 & 0,7) konvergen ke level sebanding (~0,32-0,33) -- konfirmasi ini "
     "sbg tanda performative stability, bukan artefak inisialisasi.")"""),

        md("""## 5.3 Uji lintas rezim

Kebijakan terlatih di rezim operasi dievaluasi juga pada beban 1× (referensi tervalidasi)
dan minimal satu beban lain. Akui eksplisit bila keunggulan hanya muncul pada rezim
latihnya."""),
        code("""# for ds_eval, label in [(common.DATASET_KANONIK, "1x"), (ds_operasi, "operasi")]:
#     res = evaluate_policy(common.fresh_sim(ds_eval), policy_final, ...)
#     print(label, res)
print("[TODO] Isi setelah kebijakan final tersedia.")"""),

        md("""## 5.4 Kriteria trade-off tesis (Objektif 2)

`wait_by_compliance.ratio ≤ 1,2` sebagai **kriteria seleksi model**, bukan sekadar
laporan. Bila kebijakan melanggar ini demi Gini yang lebih baik → *reward hacking*,
dilaporkan sebagai kegagalan memenuhi Objektif 2, bukan disembunyikan."""),
        code("""# res_final = evaluate_policy(...)  # perlu wait_by_compliance dari harness-style eval
# print("wait_by_compliance ratio:", res_final.get("wait_by_compliance", {}).get("ratio"))
# assert res_final["wait_by_compliance"]["ratio"] <= 1.2, "PELANGGARAN OBJEKTIF 2 -- laporkan apa adanya"
print("[TODO] Isi setelah kebijakan final tersedia.")"""),

        checklist_md("Ukuran ketercapaian Tahap 5", [
            ">=5 seed pelatihan x >=10 seed evaluasi untuk klaim final",
            "CV lintas seed dilaporkan tiap metrik",
            "Arah kemenangan konsisten di SELURUH seed, bukan hanya rata-rata",
            "Gini/trust/acceptance menyetimbang saat kebijakan dibekukan (5.2)",
            "Titik setimbang tidak bergantung trust awal (0,5 vs 0,7 konvergen sebanding)",
            "Keunggulan lintas rezim diuji & dilaporkan apa adanya (5.3)",
            "wait_by_compliance.ratio <= 1,2 dipakai sbg kriteria seleksi model (5.4)",
        ]),
        gate_md("Pelanggaran Objektif 2 (5.4) TIDAK BOLEH ditutupi demi Gini yang lebih baik -- "
               "dilaporkan eksplisit sbg kegagalan memenuhi kriteria trade-off, bagian dari "
               "kejujuran hasil (lihat Tahap 6 §6.2)."),
        kesimpulan_template("Tahap 5"),
    ]
    return make_notebook(cells)


# =============================================================================
# TAHAP 6 -- Pelaporan
# =============================================================================
def build_06():
    cells = [
        md("""# Tahap 6 — Pelaporan

**Tujuan**: menyajikan hasil sedemikian rupa sehingga setiap klaim dapat ditelusuri ke
eksperimen yang mengisolasinya.

**Prasyarat**: Tahap 0-5 selesai (minimal jalur minimum: 0→1→2→3→4.1→5.1)."""),
        nav(current="06_Pelaporan"),
        HEADER_SETUP,

        md("""## 6.1 Muat seluruh hasil tahap sebelumnya"""),
        code("""import glob
outputs = {}
for f in sorted(glob.glob(os.path.join(common.OUTDIR, "*.json"))):
    name = os.path.basename(f).replace(".json", "")
    with open(f, encoding="utf-8") as fh:
        outputs[name] = json.load(fh)
print("Berkas hasil tersedia:")
for k in outputs:
    print(" -", k)"""),

        md("""## 6.2 Tangga ablasi lengkap (isi manual berdasar Tahap 2-5)"""),
        code("""tangga = pd.DataFrame([
    dict(titik="A",  deskripsi="PDQN, trust statis (0,5)", gini_mean=np.nan, keterangan="Tahap 2"),
    dict(titik="A'", deskripsi="PDQN, trust statis (level akhir B)", gini_mean=np.nan, keterangan="Tahap 3 (kontrol)"),
    dict(titik="B",  deskripsi="PDQN, trust dinamis", gini_mean=np.nan, keterangan="Tahap 3 (pivot)"),
    dict(titik="E",  deskripsi="B + RRM (arsitektur identik)", gini_mean=np.nan, keterangan="Tahap 4"),
    dict(titik="C",  deskripsi="B + arsitektur baru (tanpa RRM)", gini_mean=np.nan, keterangan="Tahap 4 (opsional)"),
    dict(titik="D",  deskripsi="C + RRM", gini_mean=np.nan, keterangan="Tahap 4 (opsional)"),
])
print("[TODO] Isi kolom gini_mean dari hasil Tahap 2-4 tersimpan.")
display(tangga)

# selisih ter-atribusi
print("\\nSelisih ter-atribusi (isi setelah tangga di atas lengkap):")
print("B-A   = kerusakan akibat performativitas")
print("E-B   = kontribusi RRM berdiri sendiri")
print("C-B   = kontribusi arsitektur (bila dijalankan)")
print("D-C   = kontribusi RRM di atas arsitektur baru (bila dijalankan)")"""),

        md("""## 6.3 Kurva performativitas vs beban (dari Tahap 1)"""),
        code("""if "01_performativity_sweep" in outputs:
    df_perf = pd.DataFrame(outputs["01_performativity_sweep"])
    display(df_perf.groupby("load_multiplier")["trust_mean"].agg(["mean", "std"]))
else:
    print("[belum tersedia -- jalankan Tahap 1]")"""),

        md("""## 6.4 Tabel baseline lengkap S0-S3"""),
        code("""if "01_baseline_S0-S3" in outputs:
    display(pd.DataFrame(outputs["01_baseline_S0-S3"]))
else:
    print("[belum tersedia -- jalankan Tahap 1.3]")"""),

        md("""## 6.5 Checklist kejujuran wajib (§6.2 Rencana Eksekusi)

- [ ] **PDQN kalah dari `greedy_util` bahkan pada trust statis** (p=0,0039, preseden arsip)
      dinyatakan eksplisit, bukan disembunyikan
- [ ] **Sebagian masalah yang ditemukan bukan kelemahan PDQN**, melainkan cacat formulasi
      reward/lingkungan yang diperbaiki di substrat (Tahap 0) dan diberikan juga kepada PDQN
- [ ] **Rezim eksperimen dipilih berdasarkan pengukuran** (Tahap 1), bukan intuisi — kurva
      §6.3 dilampirkan sebagai bukti
- [ ] Keterbatasan simulator yang relevan (`LAPORAN_VALIDASI.md` §V5.3–V5.4) dikutip,
      terutama bahwa klaim tentang dinamika trust pada beban kanonik tidak didukung
- [ ] Anggaran pelatihan & jumlah konfigurasi yang dicoba **per lengan** dicantumkan
      (bukti kesetaraan, Lapisan 2 metodologi)
- [ ] Seluruh uji statistik menyertakan ukuran efek, bukan hanya p-value"""),

        md("""## 6.6 Batas klaim (format `LAPORAN_VALIDASI.md` §V5.4)

Isi manual, contoh kerangka:

> Hasil ini **mendukung** klaim bahwa performativitas trust menurunkan performa PDQN pada
> rezim beban [...], dan bahwa RRM memulihkan [...]% dari kerusakan tersebut, pada substrat
> Klaster 12 dengan konfigurasi reward yang telah diseimbangkan (Tahap 0).
>
> Hasil ini **tidak mendukung** klaim serupa pada beban kanonik (1×), di mana performativitas
> terbukti tidak material.
>
> Hasil ini **tidak mendukung** [...] karena [...]."""),
        code("""print("Isi §6.6 secara manual di sel markdown di atas setelah seluruh tahap selesai.")
print("Ekspor notebook ini (atau ringkasannya) sbg lampiran metodologi tesis.")"""),

        kesimpulan_template("Tahap 6 (Pelaporan akhir)"),
    ]
    return make_notebook(cells)


BUILDERS = {
    "00_Bekukan_Substrat.ipynb": build_00,
    "01_Tetapkan_Rezim.ipynb": build_01,
    "02_Replikasi_Baseline_PDQN.ipynb": build_02,
    "03_Eksperimen_Pivot.ipynb": build_03,
    "04_Solusi_RRM_Arsitektur.ipynb": build_04,
    "05_Robustness_Stabilitas.ipynb": build_05,
    "06_Pelaporan.ipynb": build_06,
}


def main():
    for fname, builder in BUILDERS.items():
        nb = builder()
        path = os.path.join(HERE, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"OK  {fname}  ({len(nb['cells'])} sel)")


if __name__ == "__main__":
    main()
