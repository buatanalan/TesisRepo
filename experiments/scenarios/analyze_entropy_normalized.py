import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Analisis entropy TERNORMALISASI thd ln(n_kandidat) -- item terbuka §5 LAPORAN_SKENARIO_TINGKAT_KESULITAN.md.

feasible_candidates() (penentu mask/kandidat per keputusan) HANYA bergantung pada lokasi user,
SoC, fitur SPKLU statis, & willingness_ratio -- TIDAK pada state dinamis (antrean/charging).
Jadi distribusi n_kandidat per keputusan bisa dihitung LANGSUNG dari jadwal spawn dataset,
tanpa menjalankan simulasi/RL penuh -- cukup replay jadwal & panggil feasible_candidates().

Menghitung E[ln(n_kandidat)] utk tiap willingness_ratio yang dipakai di eksperimen kita
(2.0 dan 5.0), lalu menormalisasi entropy_final yang sudah dikumpulkan dari berbagai skenario.
"""
import json

import numpy as np

from marl_spklu.env.user import feasible_candidates

DATASET = "scenario_dataset.json"
RATIOS = [2.0, 5.0]


def compute_ln_ncand_stats(dataset_path, ratio):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    spklu_features = {}
    for s in data["spklus"]:
        spklu_features[s["id"]] = {
            'loc': tuple(s["location"]), 'pop': float(s.get("popularity", 1.0)),
            'conn': sum(s["capacities"].values()),
        }
    n_cands = []
    for ev in data["schedule"]:
        soc = ev.get("soc", 50.0)
        loc = tuple(ev["spawn_loc"])
        feas = feasible_candidates(loc, soc, spklu_features, None, ratio)
        n_cands.append(len(feas))
    n_cands = np.array(n_cands)
    ln_n = np.log(np.maximum(n_cands, 1))
    return {
        "ratio": ratio, "n_decisions": len(n_cands),
        "mean_n_candidates": float(n_cands.mean()),
        "median_n_candidates": float(np.median(n_cands)),
        "mean_ln_n_candidates": float(ln_n.mean()),   # E[ln(n)] -- pembagi normalisasi
        "ln_of_mean_n_candidates": float(np.log(n_cands.mean())),  # utk pembanding
    }


# Entropy_final (mean 5 hari terakhir) yang sudah kita kumpulkan dari eksperimen2 sebelumnya,
# dikelompokkan menurut willingness_ratio yang dipakai.
COLLECTED_ENTROPY = {
    2.0: {
        "0_easy_all": 2.4173, "1a_popularity": 2.4208, "1b_forecaster": 2.4201,
        "1c_heterogeneity": 2.4270, "1d_congestion": 2.4154,
        "0_fixed": 2.4279, "1a_popularity_fixed": 2.4293, "1b_forecaster_fixed": 2.4113,
    },
    5.0: {
        "1e_action_space": 3.4006,
        "2_high_trust_static (30hari)": 3.4246,  # dari test_30d_curriculum_summary.json
        "3_baseline (30hari)": 3.4155,           # dari test_30d_curriculum_summary.json
        "single_seed run pertama (30hari)": 3.42,  # ~rata2 first5/last5 run_test_30d_single_seed
    },
}


def main():
    stats_by_ratio = {}
    for ratio in RATIOS:
        st = compute_ln_ncand_stats(DATASET, ratio)
        stats_by_ratio[ratio] = st
        print(f"\n=== willingness_ratio={ratio} ===")
        print(json.dumps(st, indent=2))

    print(f"\n\n{'=' * 100}")
    print("=== ENTROPY MENTAH vs TERNORMALISASI (entropy_final / E[ln(n_kandidat)]) ===")
    print(f"{'=' * 100}")
    header = f"{'skenario':<38}{'ratio':>7}{'entropy_mentah':>16}{'E[ln(n)]':>12}{'entropy_norm':>14}"
    print(header)
    print("-" * len(header))

    results = {"ln_ncand_stats": stats_by_ratio, "normalized": {}}
    for ratio, entries in COLLECTED_ENTROPY.items():
        denom = stats_by_ratio[ratio]["mean_ln_n_candidates"]
        for name, ent in entries.items():
            norm = ent / denom if denom > 0 else None
            results["normalized"][name] = {
                "ratio": ratio, "entropy_raw": ent, "ln_n_denom": denom, "entropy_normalized": norm,
            }
            print(f"{name:<38}{ratio:>7}{ent:>16.4f}{denom:>12.4f}{norm:>14.4f}")

    with open("test_entropy_normalized_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[INFO] Ringkasan -> test_entropy_normalized_summary.json")


if __name__ == "__main__":
    main()
