"""Verifikasi multi-seed: apakah pola konvergensi (Gini turun mulus + trust plateau)
pada horizon kontinu 90-hari (lihat percakapan) konsisten lintas train_seed, atau cuma
kebetulan satu run. TIDAK menyentuh trainer/policy -- murni pengulangan `PDQNContinuousTrainer`
dgn seed berbeda, merekam kurva gini/trust per-chunk utk tiap seed.

Pemakaian:
    python -m marl_spklu.experiments.diagnostics.pdqn_continuous_multiseed
"""
import json

import numpy as np

from marl_spklu.rl.pdqn_continuous_trainer import PDQNContinuousTrainer
from marl_spklu.rl.forecaster import FormulaForecaster

DATASET = "scenario_dataset_90d.json"
SEEDS = (0, 1, 2)
N_UPDATES = 90


def run_seed(seed: int):
    tr = PDQNContinuousTrainer(DATASET, k=3, rollout_steps=96, seed=seed, verbose=False)
    tr.train(FormulaForecaster(), n_updates=N_UPDATES)
    gini = np.array([h["gini_served"] for h in tr.history])
    trust = np.array([h["trust_mean"] for h in tr.history])
    return gini, trust


def main():
    results = {}
    for seed in SEEDS:
        print(f"--- train_seed={seed} ---")
        gini, trust = run_seed(seed)
        results[seed] = {"gini": gini.tolist(), "trust": trust.tolist()}
        print(f"  gini[0]={gini[0]:.3f} gini[-1]={gini[-1]:.3f}  "
             f"trust[0]={trust[0]:.3f} trust[-1]={trust[-1]:.3f}")
        # plateau: rata2 10 chunk terakhir
        print(f"  plateau (10 chunk akhir): gini={gini[-10:].mean():.3f}+-{gini[-10:].std():.3f}  "
             f"trust={trust[-10:].mean():.3f}+-{trust[-10:].std():.3f}")

    with open("hasil_pdqn_continuous_multiseed_90d.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nRINGKASAN LINTAS SEED\n{'='*70}")
    gini_start = [results[s]["gini"][0] for s in SEEDS]
    gini_end_plateau = [np.mean(results[s]["gini"][-10:]) for s in SEEDS]
    trust_start = [results[s]["trust"][0] for s in SEEDS]
    trust_end_plateau = [np.mean(results[s]["trust"][-10:]) for s in SEEDS]
    print(f"Gini awal   : {np.mean(gini_start):.3f} +- {np.std(gini_start):.3f}  (per-seed: {[f'{v:.3f}' for v in gini_start]})")
    print(f"Gini plateau: {np.mean(gini_end_plateau):.3f} +- {np.std(gini_end_plateau):.3f}  (per-seed: {[f'{v:.3f}' for v in gini_end_plateau]})")
    print(f"Trust awal   : {np.mean(trust_start):.3f} +- {np.std(trust_start):.3f}")
    print(f"Trust plateau: {np.mean(trust_end_plateau):.3f} +- {np.std(trust_end_plateau):.3f}  (per-seed: {[f'{v:.3f}' for v in trust_end_plateau]})")


if __name__ == "__main__":
    main()
