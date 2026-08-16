import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""
Visualisasi hasil evaluasi High-Load.
Membaca highload_eval_hasil.json dan menghasilkan chart perbandingan.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("highload_eval_hasil.json") as f:
    results = json.load(f)

scenarios = ["S0", "S5_HONEST", "S5_DISHONEST", "S6_HONEST", "S6_DISHONEST"]
labels = ["S0\nNo Rec", "S5 Honest\nFormula", "S5 Dishonest\nFormula", "S6 Honest\nMLP", "S6 Dishonest\nMLP"]
colors = ["#6C7A89", "#2980B9", "#E74C3C", "#27AE60", "#9B59B6"]

# Style
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor": "#1A1D27",
    "axes.edgecolor": "#2E3347",
    "axes.labelcolor": "#C9D1D9",
    "axes.titlecolor": "#E6EDF3",
    "axes.grid": True,
    "grid.color": "#2E3347",
    "grid.linewidth": 0.6,
    "text.color": "#C9D1D9",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "legend.facecolor": "#1A1D27",
    "legend.edgecolor": "#2E3347",
    "font.family": "sans-serif",
    "font.size": 9,
})

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Hasil Pengujian Beban Tinggi (High-Load Evaluation)\n"
    "4x Beban (116 Users, 132 Spawns) · 2 Hari (192 Steps) · Disjoint Users",
    fontsize=12, fontweight="bold", color="#E6EDF3", y=0.98
)

# 1. Mean Wait
mean_waits = [results[s]["mean_wait"] for s in scenarios]
axs[0, 0].bar(labels, mean_waits, color=colors, edgecolor="#2E3347", width=0.6)
axs[0, 0].set_ylabel("Waktu Tunggu Rata-rata (Menit)")
axs[0, 0].set_title("Waktu Tunggu Rata-rata (↓ lebih baik)", fontweight="bold")
for i, v in enumerate(mean_waits):
    axs[0, 0].text(i, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#E6EDF3")

# 2. Gini Served (Utilisation Inequality)
gini_served = [results[s]["gini_served"] for s in scenarios]
axs[0, 1].bar(labels, gini_served, color=colors, edgecolor="#2E3347", width=0.6)
axs[0, 1].set_ylabel("Gini Index")
axs[0, 1].set_title("Inequality Utilisasi SPKLU (↓ lebih baik = lebih adil)", fontweight="bold")
axs[0, 1].set_ylim(0.5, 0.9)
for i, v in enumerate(gini_served):
    axs[0, 1].text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color="#E6EDF3")

# 3. Active SPKLUs
n_active = [results[s]["n_active"] for s in scenarios]
axs[1, 0].bar(labels, n_active, color=colors, edgecolor="#2E3347", width=0.6)
axs[1, 0].set_ylabel("Jumlah SPKLU Aktif (dari total 50)")
axs[1, 0].set_title("Jumlah SPKLU Aktif (↑ lebih baik = load-balancing)", fontweight="bold")
axs[1, 0].set_ylim(0, 45)
for i, v in enumerate(n_active):
    axs[1, 0].text(i, v + 1, f"{v}", ha="center", va="bottom", fontsize=8, color="#E6EDF3")

# 4. User Trust
mean_trusts = [results[s]["mean_trust"] for s in scenarios]
axs[1, 1].bar(labels, mean_trusts, color=colors, edgecolor="#2E3347", width=0.6)
axs[1, 1].set_ylabel("Trust Rata-rata")
axs[1, 1].set_title("Trust Rata-rata Pengguna (↑ lebih baik)", fontweight="bold")
axs[1, 1].set_ylim(0.48, 0.53)
for i, v in enumerate(mean_trusts):
    axs[1, 1].text(i, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color="#E6EDF3")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "highload_eval_comparison.png"
fig.savefig(out, dpi=150, facecolor="#0F1117")
print(f"[OK] Visualisasi high-load disimpan ke: {out}")
