"""Hasilkan seluruh gambar prioritas Bab V + Lampiran (2026-09-02), sesuai spesifikasi
`RENCANA_VISUALISASI.md`. Keluaran: PNG ke `Dokumen tambahan/gambar_bab5/`.

Tidak butuh server -- seluruh data sumber (berkas JSON hasil evaluasi) sudah ter-pull
lokal; hanya *checkpoint* aktor (.pt) yang tak tersedia lokal, dan tak diperlukan di
sini (kita memvisualisasikan hasil yang SUDAH dievaluasi, bukan menjalankan ulang).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _analisis_bab5 as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "Dokumen tambahan", "gambar_bab5")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------- palet (§0)
W = dict(greedy="#8C8C8C", master="#E8A33D", sel2="#D9B36C", sel3="#7FA8C9",
         pmaster="#1B4F72", untung="#1E7A4C", rugi="#B33F3F", acuan="#4A4A4A")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

B4 = "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout"
SUF = {"MASTER": "_noattn_pure3", "MASTER+Atensi": "_pure3",
       "MASTER+P": "_pg0.1_noattn_pure3", "P-MASTER": "_pg0.1_pure3"}
WARNA4 = {"MASTER": W["master"], "MASTER+Atensi": W["sel2"],
         "MASTER+P": W["sel3"], "P-MASTER": W["pmaster"]}


def simpan(fig, nama):
    p = os.path.join(OUT, nama)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nama}")


# ===================================================================== Gambar 1
# TK1-3 (3 subplot) -- A.1a
def gambar_TK():
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    data = [("TK1: Gini utilisasi (lebih rendah lebih baik)",
            [A.gini_stasiun(B4 + SUF["MASTER"], "gini_utilisasi", "90d")[0],
             A.gini_stasiun(B4 + SUF["P-MASTER"], "gini_utilisasi", "90d")[0],
             A.gini_stasiun("greedy_setara_k3", "gini_utilisasi", "90d", "greedy_queue")[0]]),
           ("TK2: probabilitas penerimaan",
            [A.unit_stat(B4 + SUF["MASTER"], "acc", "90d")[0],
             A.unit_stat(B4 + SUF["P-MASTER"], "acc", "90d")[0],
             A.unit_stat("greedy_setara_k3", "acc", "90d", "greedy_queue")[0]]),
           ("TK3: |laju turun trust|/hari (×10⁻³)",
            [1.466, 0.763, 0.718])]
    label = ["MASTER", "P-MASTER", "greedy"]
    warna = [W["master"], W["pmaster"], W["greedy"]]
    for ax, (judul, vals) in zip(axes, data):
        bars = ax.bar(label, vals, color=warna, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4f}" if v < 1 else f"{v:.3f}",
                       (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                       fontsize=8)
        ax.set_title(judul, fontsize=9)
    fig.suptitle("Tujuan Kinerja 1-3, skenario beban 4×", y=1.04)
    simpan(fig, "gambar_5_1_TK1-3.png")


# ===================================================================== Gambar 2
# Progresi 4 konfigurasi: wait & nilai-mengikuti-rekomendasi -- A.1c
def gambar_progresi():
    urut = ["MASTER", "MASTER+Atensi", "MASTER+P", "P-MASTER"]
    wait = [A.unit_stat(B4 + SUF[n], "wait", "90d")[0] for n in urut]
    nilai_ikut = []
    for n in urut:
        wp = A.unit_stat(B4 + SUF[n], "wpatuh_mean", "90d")[0]
        wt = A.unit_stat(B4 + SUF[n], "wtolak_mean", "90d")[0]
        nilai_ikut.append(wt - wp)
    warna = [WARNA4[n] for n in urut]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    bars = ax1.bar(urut, wait, color=warna, edgecolor="white")
    for b, v in zip(bars, wait):
        ax1.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=8)
    ax1.axhline(wait[0], color=W["acuan"], ls="--", lw=1, alpha=0.6)
    ax1.set_ylabel("waktu tunggu (menit)"); ax1.set_title("(a) Waktu tunggu")
    ax1.annotate(f"-{(1 - wait[-1] / wait[0]) * 100:.0f}%", xy=(3, wait[-1]),
                xytext=(2.3, wait[0] * 0.7), fontsize=9, color=W["pmaster"],
                arrowprops=dict(arrowstyle="->", color=W["pmaster"]))
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    bars2 = ax2.bar(urut, nilai_ikut, color=[W["rugi"] if v < 0 else W["untung"]
                                            for v in nilai_ikut], edgecolor="white")
    for b, v in zip(bars2, nilai_ikut):
        ax2.annotate(f"{v:+.1f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v > 0 else "top", fontsize=8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("menit (+ = mengikuti menguntungkan)")
    ax2.set_title("(b) Nilai mengikuti rekomendasi")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    simpan(fig, "gambar_5_2_progresi_konfigurasi.png")


# ===================================================================== Gambar 3
# Lintasan trust harian + ambang keruntuhan -- A.4a
def gambar_trust_lintasan():
    fig, ax = plt.subplots(figsize=(7, 4))
    for nm, tag, ln, warna in [("greedy", "greedy_setara_k3", "greedy_queue", W["greedy"]),
                               ("MASTER", B4 + SUF["MASTER"], None, W["master"]),
                               ("P-MASTER", B4 + SUF["P-MASTER"], None, W["pmaster"])]:
        d = A.muat(tag, "90d"); k = A._pilih_lengan(d, ln)
        h = d["harian"][k]
        hari = np.array([r["hari"] for r in h])
        tr = np.array([r["trust_mean"] for r in h])
        m = hari >= 0
        ax.plot(hari[m], tr[m], color=warna, label=nm, lw=1.8)
    ax.axhline(0.5, color=W["acuan"], ls=":", lw=1)
    ax.plot([0, 90], [0.5, 0.0], color=W["acuan"], ls="--", lw=1,
           label="ambang keruntuhan (0,5/90 per hari)")
    ax.set_xlabel("hari"); ax.set_ylabel("trust rata-rata")
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Lintasan kepercayaan pengguna, skenario beban 4×")
    simpan(fig, "gambar_5_3_lintasan_trust.png")


# ===================================================================== Gambar 4
# Interaction plot grid 2x2, 4 metrik -- A.2a
def gambar_interaksi():
    metrik = [("gini_UTIL", "Gini utilisasi"), ("wait", "Waktu tunggu (menit)"),
             ("acc", "Probabilitas penerimaan"), ("herding", "Herding index")]

    def nilai(m, attn, p):
        suf = {(False, False): "MASTER", (True, False): "MASTER+Atensi",
              (False, True): "MASTER+P", (True, True): "P-MASTER"}[(attn, p)]
        tag = B4 + SUF[suf]
        if m == "gini_UTIL":
            return A.gini_stasiun(tag, "gini_utilisasi", "90d")
        return A.unit_stat(tag, m, "90d")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, (m, judul) in zip(axes.flat, metrik):
        for p, warna, label in [(False, W["master"], "P nonaktif"),
                                (True, W["pmaster"], "P aktif")]:
            y = [nilai(m, False, p)[0], nilai(m, True, p)[0]]
            yerr = [nilai(m, False, p)[1], nilai(m, True, p)[1]]
            ax.errorbar(["atensi OFF", "atensi ON"], y, yerr=yerr, marker="o",
                       color=warna, label=label, capsize=3, lw=1.8)
        ax.set_title(judul, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Interaksi atensi × Modul P (garis berimpit = tanpa interaksi;\n"
                "menyilang/merenggang = interaksi kuat)", y=1.02, fontsize=10)
    fig.tight_layout()
    simpan(fig, "gambar_5_4_interaksi_faktorial.png")


# ===================================================================== Gambar 5
# Diverging bar ketahanan lintas kondisi -- A.3a
def gambar_ketahanan():
    kondisi = [("trust 0,3", -0.0484), ("baku", -0.0419), ("gw ×0,5", -0.0043),
              ("trust 0,7", +0.0156), ("gw ×2", +0.0761)]
    kondisi = sorted(kondisi, key=lambda x: x[1])
    nm = [k[0] for k in kondisi]; val = [k[1] for k in kondisi]
    warna = [W["untung"] if v < 0 else W["rugi"] for v in val]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.barh(nm, val, color=warna, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    for b, v in zip(bars, val):
        ax.annotate(f"{v:+.4f}", (v, b.get_y() + b.get_height() / 2),
                   ha="left" if v > 0 else "right", va="center", fontsize=8,
                   xytext=(4 if v > 0 else -4, 0), textcoords="offset points")
    ax.set_xlabel("interaksi atensi×P pada Gini utilisasi\n(negatif = menguntungkan)")
    ax.set_title("Ketahanan interaksi lintas kondisi lingkungan")
    simpan(fig, "gambar_5_5_ketahanan_kondisi.png")


# ===================================================================== Gambar 6
# PPO vs DDPG -- B.2a (titik tunggal, tanpa kurva gradien -- data proses per-iterasi
# DDPG tak tersedia lokal utk lengan yg persis dipakai Blok B)
def gambar_ppo_ddpg():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    label = ["PPO", "DDPG", "DDPG\n(baku)"]
    warna = [W["pmaster"], W["master"], W["rugi"]]
    for ax, (vals, judul) in zip(axes, [([0.0303, 0.1922, 0.3398], "Gini utilisasi"),
                                        ([85.1, 479.3, 896.0], "Waktu tunggu (menit)")]):
        bars = ax.bar(label, vals, color=warna, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}" if v < 1 else f"{v:.0f}",
                       (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                       fontsize=8)
        ax.set_title(judul, fontsize=9)
    fig.suptitle("Backbone algoritma: PPO vs DDPG\n(MASTER dasar, tanpa atensi/Modul P — rezim kebijakan penuh)",
                y=1.08, fontsize=10)
    fig.tight_layout()
    simpan(fig, "gambar_5_6_ppo_vs_ddpg.png")


# ===================================================================== Gambar 7
# Modul P: pref_dalam/pref_primer + garis ambang kebetulan -- B.3a
def gambar_modul_p():
    urut = ["MASTER", "MASTER+Atensi", "MASTER+P", "P-MASTER"]
    warna = [WARNA4[n] for n in urut]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, (m, ambang, judul) in zip(axes, [("pref_dalam", 1 / 2, "pref_dalam (kebetulan k/6=0,500)"),
                                             ("pref_primer", 1 / 6, "pref_primer (kebetulan 1/6=0,167)")]):
        try:
            vals = [A.unit_stat(B4 + SUF[n], m, "90d")[0] for n in urut]
        except Exception:
            vals = None
        if vals is None:
            continue
        bars = ax.bar(urut, vals, color=warna, edgecolor="white")
        ax.axhline(ambang, color=W["acuan"], ls="--", lw=1.2, label="kebetulan (agen buta-user)")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                       va="bottom", fontsize=8)
        ax.set_title(judul, fontsize=9)
        ax.legend(fontsize=7)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Pemahaman preferensi: hampir menempel garis kebetulan", y=1.03)
    fig.tight_layout()
    simpan(fig, "gambar_5_7_modul_p_preferensi.png")


# ===================================================================== Gambar 8
# Kurva antrean cembung + titik greedy/P-MASTER -- B.1c (paling penting §5.4)
def gambar_kurva_antrean():
    rho = np.linspace(0, 0.95, 400)
    f = rho / (1 - rho)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(rho, f, color=W["acuan"], lw=1.2)
    titik = [("greedy", 0.620, W["greedy"]), ("P-MASTER", 0.714, W["pmaster"])]
    for nm, r, warna in titik:
        y = r / (1 - r)
        ax.scatter([r], [y], color=warna, zorder=5, s=70, edgecolor="white")
        ax.annotate(f"{nm}\nρ={r:.3f}\nf(ρ)={y:.2f}", (r, y), xytext=(10, 10),
                   textcoords="offset points", fontsize=8, color=warna)
        ax.plot([r, r], [0, y], color=warna, ls=":", lw=1)
        ax.plot([0, r], [y, y], color=warna, ls=":", lw=1)
    ax.set_xlabel(r"$\rho$ (beban tawaran / kapasitas)")
    ax.set_ylabel(r"$f(\rho)=\rho/(1-\rho)$")
    ax.set_title("Kecembungan model antrean: mengapa kenaikan lama layan\n"
                "berlipat ganda pada kenaikan waktu tunggu")
    ax.set_ylim(0, 4)
    simpan(fig, "gambar_5_8_kurva_antrean.png")


# ===================================================================== Gambar 9
# Utilisasi per stasiun greedy vs P-MASTER, berlabel komposisi konektor -- B.1a
def gambar_utilisasi_stasiun():
    KAP = {"SPKLU_00": "0AC+1DC", "SPKLU_01": "1AC+1DC", "SPKLU_02": "4AC+0DC",
          "SPKLU_03": "4AC+0DC", "SPKLU_04": "3AC+1DC", "SPKLU_05": "1AC+1DC"}
    sids = list(KAP)

    def util(tag, ln=None):
        d = A.muat(tag, "90d"); k = A._pilih_lengan(d, ln)
        return {s: np.mean([r["_stasiun"][s]["util_mean"] for r in d["per_seed"][k]])
               for s in sids}

    metode = [("greedy", util("greedy_setara_k3", "greedy_queue"), W["greedy"]),
             ("MASTER", util(B4 + SUF["MASTER"]), W["master"]),
             ("MASTER+Atensi", util(B4 + SUF["MASTER+Atensi"]), W["sel2"]),
             ("MASTER+P", util(B4 + SUF["MASTER+P"]), W["sel3"]),
             ("P-MASTER", util(B4 + SUF["P-MASTER"]), W["pmaster"])]

    y = np.arange(len(sids))
    n = len(metode)
    h = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (nm, u, warna) in enumerate(metode):
        offset = (n - 1) / 2 * h - i * h
        ax.barh(y + offset, [u[s] for s in sids], height=h, color=warna, label=nm)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} ({KAP[s]})" for s in sids])
    ax.set_xlabel("utilisasi (waktu pakai / waktu pakai maksimum)")
    ax.legend(fontsize=8, ncol=1, loc="lower right")
    ax.set_title("Utilisasi per stasiun, seluruh metode, berlabel komposisi konektor")
    simpan(fig, "gambar_5_9_utilisasi_stasiun.png")


if __name__ == "__main__":
    print(f"Menyimpan ke: {os.path.abspath(OUT)}\n")
    for fn in [gambar_TK, gambar_progresi, gambar_trust_lintasan, gambar_interaksi,
              gambar_ketahanan, gambar_ppo_ddpg, gambar_modul_p, gambar_kurva_antrean,
              gambar_utilisasi_stasiun]:
        try:
            fn()
        except Exception as e:
            print(f"  GAGAL {fn.__name__}: {e}")
    print("\nSELESAI.")
