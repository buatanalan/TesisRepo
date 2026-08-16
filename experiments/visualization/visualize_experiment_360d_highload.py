import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""
Visualisasi distribusi metrik eksperimen 360-hari BEBAN TINGGI.
Membaca main_experiment_360d_highload.json dan menghasilkan dashboard.
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
with open("main_experiment_360d_highload.json") as f:
    data = json.load(f)

comp = data["comparison"]
history = data["train_history"]

SCENARIOS = ["S0_no_intervention", "S1_greedy", "S3_opsrl", "S4_marl"]
LABELS = ["S0\nNo-intervensi", "S1\nGreedy", "S3\nOP-SRL", "S4\nMARL"]
COLORS = ["#6C7A89", "#E67E22", "#27AE60", "#2980B9"]

# ─────────────────────────────────────────────
# Pallete & style
# ─────────────────────────────────────────────
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

fig = plt.figure(figsize=(20, 22), facecolor="#0F1117")
fig.suptitle(
    "Analisis Distribusi Metrik — Eksperimen 360 Hari BEBAN TINGGI\n"
    "BETA_QUEUE_OBS=−1.2 · updates=30 · rollout=480 · k=3 · eval_seeds=5 · 2x Load Density",
    fontsize=13, fontweight="bold", color="#E6EDF3", y=0.995
)

gs = gridspec.GridSpec(
    4, 4, figure=fig,
    hspace=0.52, wspace=0.38,
    top=0.968, bottom=0.04, left=0.06, right=0.97
)


# ─── helper ────────────────────────────────────
def bar_mean_std(ax, key, ylabel, title, fmt=".1f", pct=False, ylim=None):
    means, stds, labels = [], [], []
    for sc, lb in zip(SCENARIOS, LABELS):
        v = comp[sc][key]
        if isinstance(v, dict):
            means.append(v["mean"])
            stds.append(v["std"])
        else:
            means.append(v)
            stds.append(0.0)
        labels.append(lb)

    x = np.arange(len(SCENARIOS))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=COLORS, edgecolor="#2E3347", linewidth=0.8,
                  error_kw={"ecolor": "#ffffff55", "elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    if ylim:
        ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    for bar, m, s in zip(bars, means, stds):
        lbl = f"{m:{fmt}}" + (f"\n±{s:{fmt}}" if s > 0 else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s * 1.05,
                lbl, ha="center", va="bottom", fontsize=7, color="#E6EDF3")
    return bars


def scalar_bar(ax, key, ylabel, title, fmt=".1f", ylim=None):
    vals = [comp[sc][key] for sc in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    bars = ax.bar(x, vals, color=COLORS, edgecolor="#2E3347", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=7.5, color="#E6EDF3")
    return bars


# ════════════════════════════════════════════
# ROW 0 — KPI Level
# ════════════════════════════════════════════

# 0-0: Wait Mean
ax00 = fig.add_subplot(gs[0, 0])
bar_mean_std(ax00, "wait_mean", "Menit", "Rata-rata Waktu Tunggu\n(mean lintas seed, ↓ lebih baik)", fmt=".1f")
ax00.set_ylim(0, 55)

# 0-1: Wait P90
ax01 = fig.add_subplot(gs[0, 1])
scalar_bar(ax01, "wait_p90", "Menit", "Waktu Tunggu P90\n(persentil ke-90, ↓ lebih baik)", fmt=".0f")

# 0-2: frac_inactive
ax02 = fig.add_subplot(gs[0, 2])
vals_inactive = [comp[sc]["frac_inactive"] for sc in SCENARIOS]
x = np.arange(len(SCENARIOS))
bars = ax02.bar(x, vals_inactive, color=COLORS, edgecolor="#2E3347", linewidth=0.8)
ax02.set_xticks(x); ax02.set_xticklabels(LABELS, fontsize=8)
ax02.set_ylabel("Proporsi Stasiun Idle", fontsize=8)
ax02.set_title("Fraksi Stasiun Tidak-Aktif\n(↓ lebih baik, 0=semua aktif)", fontsize=9, fontweight="bold", pad=6)
ax02.set_ylim(0, 0.04)
ax02.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
for bar, v in zip(bars, vals_inactive):
    ax02.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
              f"{v:.2%}", ha="center", va="bottom", fontsize=7.5, color="#E6EDF3")

# 0-3: herding_events
ax03 = fig.add_subplot(gs[0, 3])
bar_mean_std(ax03, "herding_events", "Jumlah Event", "Jumlah Herding Events\n(↓ lebih baik)", fmt=".0f")

# ════════════════════════════════════════════
# ROW 1 — Dinamika Gini
# ════════════════════════════════════════════

# 1-0: auc_gini
ax10 = fig.add_subplot(gs[1, 0])
bar_mean_std(ax10, "auc_gini", "AUC Gini", "AUC Gini Utilisasi\n(imbalansi persisten, ↑ = lebih timpang)",
             fmt=".3f", ylim=(0.80, 0.90))
ax10.axhline(0.3, color="#FF6B6B", linestyle="--", linewidth=1, alpha=0.7, label="Gate >0.3")
ax10.axhline(0.8, color="#58D68D", linestyle="--", linewidth=1, alpha=0.7, label="Threshold 0.8")
ax10.legend(fontsize=7)

# 1-1: gini_final
ax11 = fig.add_subplot(gs[1, 1])
bar_mean_std(ax11, "gini_final", "Gini Akhir", "Gini Utilisasi (Akhir Simulasi)\n(mean±std lintas seed)",
             fmt=".3f", ylim=(0.70, 0.90))

# 1-2: gini_slope (×10⁻⁶)
ax12 = fig.add_subplot(gs[1, 2])
means_slope = [comp[sc]["gini_slope"]["mean"] * 1e6 for sc in SCENARIOS]
stds_slope = [comp[sc]["gini_slope"]["std"] * 1e6 for sc in SCENARIOS]
x = np.arange(len(SCENARIOS))
ax12.bar(x, means_slope, yerr=stds_slope, capsize=5, color=COLORS,
         edgecolor="#2E3347", linewidth=0.8,
         error_kw={"ecolor": "#ffffff55", "elinewidth": 1.2})
ax12.set_xticks(x); ax12.set_xticklabels(LABELS, fontsize=8)
ax12.set_ylabel("Slope (×10⁻⁶)", fontsize=8)
ax12.set_title("Slope Gini Utilisasi\n(tren temporal, nilai negatif = membaik)", fontsize=9, fontweight="bold", pad=6)
ax12.axhline(0, color="#FF6B6B", linestyle="--", linewidth=0.8, alpha=0.6)
for i, (m, s) in enumerate(zip(means_slope, stds_slope)):
    ax12.text(i, m - s - 0.05, f"{m:.2f}e-6", ha="center", va="top", fontsize=7, color="#E6EDF3")

# 1-3: time_to_neg_slope
ax13 = fig.add_subplot(gs[1, 3])
tts_median = [comp[sc]["time_to_neg_slope"]["median_day"] or 0 for sc in SCENARIOS]
tts_n = [comp[sc]["time_to_neg_slope"]["n_reached"] for sc in SCENARIOS]
bars13 = ax13.bar(x, tts_median, color=COLORS, edgecolor="#2E3347", linewidth=0.8)
ax13.set_xticks(x); ax13.set_xticklabels(LABELS, fontsize=8)
ax13.set_ylabel("Hari", fontsize=8)
ax13.set_title("Hari ke Slope Gini Negatif\n(kecepatan konvergensi, ↓ lebih baik)", fontsize=9, fontweight="bold", pad=6)
for bar, v, n in zip(bars13, tts_median, tts_n):
    ax13.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
              f"Hari {v:.0f}\n({n}/5 seed)", ha="center", va="bottom", fontsize=7, color="#E6EDF3")

# ════════════════════════════════════════════
# ROW 2 — Koordinasi & Trust
# ════════════════════════════════════════════

# 2-0: herding_index
ax20 = fig.add_subplot(gs[2, 0])
bar_mean_std(ax20, "herding_index", "Indeks Herding", "Indeks Herding\n(↓ lebih baik, 0=tidak ada flocking)",
             fmt=".3f", ylim=(0, 0.15))
ax20.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))

# 2-1: rec_entropy_norm
ax21 = fig.add_subplot(gs[2, 1])
bar_mean_std(ax21, "rec_entropy_norm", "Entropi Ternormalisasi", "Entropi Rekomendasi (Ternormalisasi)\n(↑ = distribusi lebih merata)",
             fmt=".3f", ylim=(0, 0.30))

# 2-2: acceptance_overall
ax22 = fig.add_subplot(gs[2, 2])
bar_mean_std(ax22, "acceptance_overall", "Tingkat Kepatuhan", "Tingkat Kepatuhan Rekomendasi\n(overall, mean±std lintas seed)",
             fmt=".3f", ylim=(0, 1.15))
ax22.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax22.axhline(1.0, color="#FF6B6B", linestyle="--", linewidth=0.8, alpha=0.6, label="Kepatuhan penuh")
ax22.legend(fontsize=7)

# 2-3: trust_mean & trust_frac_below_0.5
ax23 = fig.add_subplot(gs[2, 3])
trust_means = [comp[sc]["trust_mean"] for sc in SCENARIOS]
trust_fracs = [comp[sc]["trust_frac_below_0.5"] for sc in SCENARIOS]
x = np.arange(len(SCENARIOS))
width = 0.38
b1 = ax23.bar(x - width/2, trust_means, width, color=COLORS, edgecolor="#2E3347", linewidth=0.8, label="Trust mean")
b2 = ax23.bar(x + width/2, trust_fracs, width, color=[c + "99" for c in COLORS],
              edgecolor="#2E3347", linewidth=0.8, hatch="//", label="Frac trust<0.5")
ax23.set_xticks(x); ax23.set_xticklabels(LABELS, fontsize=8)
ax23.set_ylabel("Nilai Trust", fontsize=8)
ax23.set_title("Distribusi Trust Pengguna\n(mean & fraksi trust < 0.5)", fontsize=9, fontweight="bold", pad=6)
ax23.axhline(0.5, color="#FFD700", linestyle="--", linewidth=0.8, alpha=0.7)
ax23.set_ylim(0, 0.95)
ax23.legend(fontsize=7)
for bar, v in zip(b1, trust_means):
    ax23.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#E6EDF3")
for bar, v in zip(b2, trust_fracs):
    ax23.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#E6EDF3")

# ════════════════════════════════════════════
# ROW 3 — Training Curve S4 & Radar Summary
# ════════════════════════════════════════════

# 3-0~1: Training curve (reward, gini, EV, KL)
ax_train = fig.add_subplot(gs[3, :2])
iters = [h["iter"] for h in history]
rewards = [h["mean_reward"] for h in history]
gini_tr = [h["gini_served"] for h in history]
ev_tr = [h["explained_var"] for h in history]
kl_tr = [h["approx_kl"] for h in history]

ax_train.plot(iters, rewards, color="#2980B9", linewidth=1.8, label="Mean Reward", zorder=3)
ax_train.fill_between(iters, rewards, alpha=0.15, color="#2980B9")
ax_train.plot(iters, gini_tr, color="#E67E22", linewidth=1.5, linestyle="--", label="Gini Served (training)", zorder=3)
ax_train.plot(iters, ev_tr, color="#27AE60", linewidth=1.2, linestyle=":", label="Explained Var (critic)", zorder=2)
ax_train.plot(iters, kl_tr, color="#9B59B6", linewidth=1.0, linestyle="-.", label="Approx KL", zorder=2)
ax_train.axhline(0, color="#ffffff33", linewidth=0.7)
ax_train.set_xlabel("Chunk-Update (Iterasi)", fontsize=9)
ax_train.set_ylabel("Nilai", fontsize=9)
ax_train.set_title("Kurva Training S4 MARL — 30 Chunk-Update\n(Reward, Gini, Explained Variance, KL-Divergence)",
                   fontsize=9, fontweight="bold", pad=6)
ax_train.legend(fontsize=8, loc="upper left")
ax_train.set_xlim(min(iters), max(iters))

# Annotate final values
for label, vals, col in [("R final", rewards, "#2980B9"), ("Gini final", gini_tr, "#E67E22")]:
    ax_train.annotate(f"{label}={vals[-1]:.3f}",
                      xy=(iters[-1], vals[-1]),
                      xytext=(-40, 10),
                      textcoords="offset points",
                      color=col, fontsize=7.5,
                      arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

# 3-2~3: Radar chart
ax_radar = fig.add_subplot(gs[3, 2:], polar=True)

metrics_radar = {
    "Wait\n(↓)": [comp[sc]["wait_mean"] for sc in SCENARIOS],
    "Herding\n(↓)": [comp[sc]["herding_index"]["mean"] for sc in SCENARIOS],
    "AUC Gini\n(↑)": [comp[sc]["auc_gini"]["mean"] for sc in SCENARIOS],
    "Acceptance\n(↑)": [comp[sc]["acceptance_overall"]["mean"] for sc in SCENARIOS],
    "Rec\nEntropy(↑)": [comp[sc]["rec_entropy_norm"]["mean"] for sc in SCENARIOS],
    "Inactive\n(↓)": [comp[sc]["frac_inactive"] for sc in SCENARIOS],
}

def normalize(vals, lower_is_better=False):
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [0.5] * len(vals)
    norm = [(v - mn) / (mx - mn) for v in vals]
    return [1 - n for n in norm] if lower_is_better else norm

radar_data = {}
for (label, vals), lob in zip(metrics_radar.items(),
                               [True, True, False, False, False, True]):
    radar_data[label] = normalize(vals, lower_is_better=lob)

categories = list(radar_data.keys())
N = len(categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]  # close

for i, (sc, lb, col) in enumerate(zip(SCENARIOS, LABELS, COLORS)):
    values = [radar_data[cat][i] for cat in categories]
    values += values[:1]
    ax_radar.plot(angles, values, color=col, linewidth=1.8, label=lb.replace("\n", " "))
    ax_radar.fill(angles, values, color=col, alpha=0.10)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=8, color="#C9D1D9")
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6, color="#8B949E")
ax_radar.set_facecolor("#1A1D27")
ax_radar.spines["polar"].set_color("#2E3347")
ax_radar.grid(color="#2E3347", linewidth=0.6)
ax_radar.set_title("Radar: Perbandingan Multi-Metrik (360 Hari Beban Tinggi)\n(dinormalisasi 0→1, 1=terbaik per metrik)",
                   fontsize=9, fontweight="bold", pad=18, color="#E6EDF3")
ax_radar.legend(loc="lower right", bbox_to_anchor=(1.35, -0.08), fontsize=8)

# ─── Save ────────────────────────────────────
out = "experiment_360d_highload_dashboard.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0F1117")
print(f"[OK] Dashboard disimpan ke: {out}")
