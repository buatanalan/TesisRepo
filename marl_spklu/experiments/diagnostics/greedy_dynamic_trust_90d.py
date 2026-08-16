"""Baseline Greedy di lingkungan PERFORMATIF yang SAMA dgn PDQN kontinu: rec_mode="estwait"
(default User, TIDAK di-override), trust DINAMIS (TIDAK dibekukan via constant_trust) --
beda dari PDQN diskrit Tahap 0 yang sengaja membekukan trust. Dijalankan pada dataset
90-hari kontinu yang SAMA (scenario_dataset_90d.json), disampel per-chunk (96 langkah = 1
hari) dgn metrik IDENTIK yang dipakai PDQNContinuousTrainer.history (gini_served dari
total_served KUMULATIF, trust_mean populasi) supaya kurva bisa dibandingkan apple-to-apple.

Greedy TIDAK dilatih (rule-based) -- tidak ada "seed training", tapi seed dataset/RNG
tetap divariasikan (3 seed) supaya draw populasi (beta_dist dst, MXL heterogen per-user)
sebanding dgn 3 run PDQN kontinu yang sudah ada (lihat pdqn_continuous_multiseed.py).

Pemakaian:
    python -m marl_spklu.experiments.diagnostics.greedy_dynamic_trust_90d
"""
import json
import random

import numpy as np

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.env.simulator import Simulator
from marl_spklu.experiments import harness

DATASET = "scenario_dataset_90d.json"
SEEDS = (0, 1, 2)
CHUNK = 96


def _gini(a):
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def run_seed(mode: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    sim = Simulator({}, [], None, user_willingness_radius_km=None,
                    user_willingness_ratio=harness.DEFAULT_WILLINGNESS_RATIO)
    sim.load_from_dataset(DATASET)
    max_steps = getattr(sim, "max_steps", max(sim.spawn_schedule) + 1)
    sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=max_steps)
    agent = GreedyAgent(mode=mode)   # rec_mode="estwait" default User, trust DINAMIS (tak dibekukan)

    gini_curve, trust_curve = [], []
    step = 0
    while step < max_steps:
        for _ in range(CHUNK):
            if step >= max_steps:
                break
            sim.step_once(step, agent=agent)
            step += 1
        served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
        trusts = np.array([u.trust for u in sim.users], dtype=float)
        gini_curve.append(_gini(served))
        trust_curve.append(float(trusts.mean()) if trusts.size else 0.5)
    return np.array(gini_curve), np.array(trust_curve)


def main():
    results = {}
    for mode in ("utilization", "queue"):
        results[mode] = {}
        for seed in SEEDS:
            print(f"--- Greedy(mode={mode}) seed={seed} ---")
            gini, trust = run_seed(mode, seed)
            results[mode][seed] = {"gini": gini.tolist(), "trust": trust.tolist()}
            print(f"  gini[0]={gini[0]:.3f} gini[-1]={gini[-1]:.3f}  "
                 f"trust[0]={trust[0]:.3f} trust[-1]={trust[-1]:.3f}  "
                 f"plateau(10 akhir): gini={gini[-10:].mean():.3f} trust={trust[-10:].mean():.3f}")

    with open("hasil_greedy_dynamic_trust_90d.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nRINGKASAN LINTAS SEED (Greedy, trust dinamis, 90 hari)\n{'='*70}")
    for mode in results:
        gini_end = [np.mean(results[mode][s]["gini"][-10:]) for s in SEEDS]
        trust_end = [np.mean(results[mode][s]["trust"][-10:]) for s in SEEDS]
        print(f"Greedy({mode}): gini_plateau={np.mean(gini_end):.3f}+-{np.std(gini_end):.3f}  "
             f"trust_plateau={np.mean(trust_end):.3f}+-{np.std(trust_end):.3f}")
    print("\nPembanding PDQN kontinu (dari hasil_pdqn_continuous_multiseed_90d.json, sudah ada):")
    print("  gini_plateau=0.159+-0.004  trust_plateau=0.787+-0.008")


if __name__ == "__main__":
    main()
