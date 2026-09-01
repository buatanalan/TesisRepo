"""Analisis bobot attention antar-stasiun (Uji A', bagian 2) -- baca `attensi_*.npz`
dan `collision_*.json` hasil `_ekstrak_attensi_stasiun.py`, uji tiga sinyal (lih.
docstring skrip ekstraksi utk rumusan lengkap tiap sinyal).

Pemakaian:
    python _analisis_attensi_stasiun.py \
        master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3 \
        master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pure3 \
        master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3 \
        greedy
(argumen pertama & kedua = lengan BERATENSI utk Sinyal 1&3; SEMUA argumen dipakai
utk tabel Sinyal 2 collision rate.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, common
from scipy.stats import pearsonr

ARMS = sys.argv[1:] or [
    "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3",
    "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pure3",
    "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3",
    "greedy",
]


def sinyal_1(tag_arm):
    """Korelasi weights[i,j] (i != j) dgn utilisasi kandidat j SAAT keputusan itu.
    Prediksi: NEGATIF -- attention mengecil ke kandidat yg sudah ramai."""
    f = os.path.join(common.OUTDIR, f"attensi_{tag_arm}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    w, mask, util = d["w"], d["mask"], d["util"]   # (M,N,N), (M,N), (M,N)
    N = w.shape[1]
    xs, ys = [], []
    for b in range(w.shape[0]):
        for i in range(N):
            if not mask[b, i]:
                continue
            for j in range(N):
                if i == j or not mask[b, j]:
                    continue
                xs.append(util[b, j])
                ys.append(w[b, i, j])
    xs, ys = np.array(xs), np.array(ys)
    r, p = pearsonr(xs, ys)
    return dict(n_pasangan=len(xs), r=float(r), p=float(p),
               arah="NEGATIF (sesuai prediksi)" if r < 0 else "POSITIF (berlawanan prediksi)")


def sinyal_3(tag_arm):
    """Rasio varians ANTAR-komposisi (pola mask berbeda) vs DALAM-komposisi (pola
    mask sama, ulangan berbeda), per stasiun query i, dirata-ratakan. Rasio >> 1
    berarti bobot BERUBAH bermakna mengikuti siapa pesaingnya (peka konteks);
    rasio ~1 berarti bobot nyaris konstan berapa pun komposisinya."""
    f = os.path.join(common.OUTDIR, f"attensi_{tag_arm}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    w, mask = d["w"], d["mask"]
    N = w.shape[1]
    rasio_per_stasiun = []
    for i in range(N):
        aktif = mask[:, i]
        if aktif.sum() < 10:
            continue
        wi = w[aktif, i, :]              # (M_i, N) -- vektor bobot stasiun i tiap kejadian
        pola = [tuple(m) for m in mask[aktif]]
        grup = {}
        for k, p in enumerate(pola):
            grup.setdefault(p, []).append(wi[k])
        if len(grup) < 2:
            continue
        semua = np.concatenate([np.array(v) for v in grup.values()])
        var_total = semua.var(axis=0).mean()
        var_dalam = np.mean([np.array(v).var(axis=0).mean() if len(v) > 1 else 0.0
                             for v in grup.values()])
        var_antar = max(var_total - var_dalam, 0.0)
        if var_dalam > 1e-8:
            rasio_per_stasiun.append(var_antar / var_dalam)
    if not rasio_per_stasiun:
        return None
    return dict(n_stasiun=len(rasio_per_stasiun), rasio_mean=float(np.mean(rasio_per_stasiun)),
               rasio_median=float(np.median(rasio_per_stasiun)))


def sinyal_2():
    print("SINYAL 2 -- collision rate (proporsi rekomendasi berbagi stasiun tujuan "
         "dgn rekomendasi lain pd langkah yang sama)\n")
    print(f"{'lengan':70s}{'collision_rate':>15s}")
    baris = []
    for arm in ARMS:
        f = os.path.join(common.OUTDIR, f"collision_{arm}.json")
        if not os.path.exists(f):
            print(f"{arm:70s}{'(tak ada data)':>15s}")
            continue
        import json
        d = json.load(open(f, encoding="utf-8"))
        cr = d["collision_rate_mean"]
        baris.append((arm, cr))
        print(f"{arm:70s}{cr:15.4f}")
    if len(baris) >= 2:
        acuan = dict(baris)
        if "greedy" in acuan:
            base = acuan["greedy"]
            print(f"\nSelisih thd greedy (negatif = attention MENGURANGI tabrakan):")
            for nm, cr in baris:
                if nm != "greedy":
                    print(f"   {nm:60s} {cr - base:+.4f}")


def main():
    berattensi = [a for a in ARMS if a != "greedy"]
    print("UJI A' -- attention antar-stasiun (2026-09-01)\n")
    print("SINYAL 1 -- korelasi bobot attention vs utilisasi kandidat yang dituju\n")
    print(f"{'lengan':60s}{'n pasangan':>12s}{'r':>8s}{'p':>10s}  arah")
    for arm in berattensi:
        s1 = sinyal_1(arm)
        if s1 is None:
            print(f"{arm:60s}  (tak ada data attensi -- mungkin `use_station_attn=False`)")
            continue
        print(f"{arm:60s}{s1['n_pasangan']:12,d}{s1['r']:8.4f}{s1['p']:10.2e}  {s1['arah']}")
    print()

    print("SINYAL 3 -- rasio varians ANTAR-komposisi vs DALAM-komposisi (>>1 = peka konteks)\n")
    print(f"{'lengan':60s}{'n stasiun':>10s}{'rasio (mean)':>14s}{'rasio (median)':>16s}")
    for arm in berattensi:
        s3 = sinyal_3(arm)
        if s3 is None:
            print(f"{arm:60s}  (tak ada data cukup)")
            continue
        print(f"{arm:60s}{s3['n_stasiun']:10d}{s3['rasio_mean']:14.3f}{s3['rasio_median']:16.3f}")
    print()

    sinyal_2()

    print("\nPembacaan: Sinyal 1 negatif & signifikan (p<0.05) DAN Sinyal 3 rasio >> 1")
    print("mendukung klaim Bab IV bahwa attention peka thd persaingan antar-kandidat.")
    print("Sinyal 2 mengonfirmasi/menyanggah pd level PERILAKU AGREGAT (independen dari")
    print("Sinyal 1/3 -- tak butuh bobot attention sama sekali, jadi bisa jadi silang-cek).")


if __name__ == "__main__":
    main()
