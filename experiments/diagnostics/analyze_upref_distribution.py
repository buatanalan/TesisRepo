import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Distribusi u_pref per-SPKLU dan kapasitas rekomendasi untuk MENGGESER pilihan pribadi.

Dasar teori (lihat User.decide_spklu mode `binary_utility`):
    score(j) = (1 - mu_hat) * u_pref(j) + mu_hat * f_rec(j),   pilihan = argmax score
Dengan f_rec one-hot pada j=a_hat, rekomendasi MENANG persis bila

    mu_hat / (1 - mu_hat)  >  u_pref(favorit) - u_pref(a_hat)          ... (*)

Jadi yang menentukan bukan level absolut u_pref, melainkan SELISIH (gap) terhadap favorit
pribadi pengguna pada saat itu. Skrip ini mengukur distribusi gap tersebut dari konteks
keputusan NYATA (kandidat feasible setelah filter jangkauan + balking, lokasi/SoC/kebiasaan
aktual), lalu menghitung berapa persen keputusan yang dapat digeser tiap SPKLU pada tiap
mu_hat.

Dijalankan pada skenario NATURAL (tanpa rekomendasi) -- inilah lanskap preferensi organik
yang harus dilawan perekomendasi saat mulai bekerja.

Pakai:
    python experiments/diagnostics/analyze_upref_distribution.py [--dataset ...] [--seeds 3]
"""
import argparse
import random
from collections import defaultdict

import numpy as np

from marl_spklu.env.user import User
from marl_spklu.rl.training import _fresh_sim

MUS = (0.2, 0.5, 0.8)


def collect(dataset, seeds):
    """Jalankan simulasi Natural, rekam u_pref tiap keputusan lewat hook User.last_u_pref."""
    recs = []          # (candidate_ids, u_pref array)
    orig_decide = User.decide_spklu

    def spy(self, *a, **kw):
        out = orig_decide(self, *a, **kw)
        if self.last_u_pref is not None:
            recs.append((list(self.last_candidate_ids), np.array(self.last_u_pref)))
        return out

    User.decide_spklu = spy
    try:
        for seed in seeds:
            random.seed(seed); np.random.seed(seed)
            sim = _fresh_sim(dataset)
            sim.run(max_steps=sim.max_steps, agent=None)
            sids = list(sim.spklus.keys())
    finally:
        User.decide_spklu = orig_decide
    return recs, sids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scenario_dataset_3d.json")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    recs, sids = collect(args.dataset, range(args.seeds))
    n_dec = len(recs)
    print(f"Dataset {args.dataset} | {args.seeds} seed | {n_dec} keputusan terekam\n")

    # ---- 1. Distribusi u_pref per SPKLU (hanya saat SPKLU itu jadi kandidat) ----
    vals = defaultdict(list)      # sid -> nilai u_pref
    gaps = defaultdict(list)      # sid -> gap thd favorit pribadi
    n_cand = defaultdict(int)     # sid -> berapa kali jadi kandidat
    n_fav = defaultdict(int)      # sid -> berapa kali jadi favorit pribadi
    for cand, u in recs:
        best = float(u.max())
        fav = cand[int(u.argmax())]
        n_fav[fav] += 1
        for i, sid in enumerate(cand):
            vals[sid].append(float(u[i]))
            gaps[sid].append(best - float(u[i]))
            n_cand[sid] += 1

    print("=== 1. Distribusi u_pref per SPKLU (saat menjadi kandidat) ===")
    hdr = (f"{'SPKLU':<11}{'%kandidat':>10}{'%favorit':>10}"
           f"{'u_mean':>9}{'u_sd':>8}{'u_p10':>9}{'u_p90':>9}")
    print(hdr); print("-" * len(hdr))
    for sid in sids:
        v = np.array(vals[sid]) if vals[sid] else np.array([np.nan])
        print(f"{sid:<11}{100*n_cand[sid]/n_dec:>9.1f}%{100*n_fav[sid]/n_dec:>9.1f}%"
              f"{v.mean():>9.2f}{v.std():>8.2f}{np.percentile(v,10):>9.2f}{np.percentile(v,90):>9.2f}")

    # ---- 2. Gap terhadap favorit pribadi -> daya geser ----
    print("\n=== 2. Gap ke favorit pribadi & % keputusan yang BISA digeser ===")
    print("    (rekomendasi j menang bila mu/(1-mu) > gap_j; gap=0 berarti j memang favorit)")
    hdr = (f"{'SPKLU':<11}{'gap_med':>9}{'gap_p25':>9}{'gap_p75':>9}"
           + "".join(f"{'mu='+str(m):>9}" for m in MUS))
    print(hdr); print("-" * len(hdr))
    shift_by_mu = {m: [] for m in MUS}
    for sid in sids:
        g = np.array(gaps[sid]) if gaps[sid] else np.array([np.nan])
        row = f"{sid:<11}{np.median(g):>9.2f}{np.percentile(g,25):>9.2f}{np.percentile(g,75):>9.2f}"
        for m in MUS:
            thr = m / (1 - m)
            # % dari SELURUH keputusan: hanya berlaku saat sid memang kandidat
            frac = float((g < thr).sum()) / n_dec
            shift_by_mu[m].append(frac)
            row += f"{100*frac:>8.1f}%"
        print(row)

    # ---- 3. Ringkasan daya geser sistem ----
    print("\n=== 3. Ringkasan: seberapa besar rekomendasi bisa menggeser pilihan ===")
    for m in MUS:
        thr = m / (1 - m)
        arr = np.array(shift_by_mu[m])
        # Rekomendasi TERBAIK-KASUS: selalu pilih SPKLU yang paling mungkin diterima
        best_sid = sids[int(arr.argmax())]
        # Rekomendasi yang benar2 MENGGESER (gap>0, bukan sekadar menyetujui favorit)
        moved = []
        for cand, u in recs:
            best = float(u.max())
            g = best - u
            # bisa digeser ke SPKLU LAIN (gap>0) yang masih di bawah ambang?
            moved.append(bool(((g > 1e-9) & (g < thr)).any()))
        print(f"  mu={m} (ambang {thr:.2f}): "
              f"rata2 {100*arr.mean():.1f}% keputusan per-SPKLU dapat diterima | "
              f"terbaik {best_sid} {100*arr.max():.1f}% | "
              f"{100*np.mean(moved):.1f}% keputusan BISA digeser ke SPKLU lain")


if __name__ == "__main__":
    main()
