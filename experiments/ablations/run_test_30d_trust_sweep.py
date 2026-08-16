import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Kurva dosis-respons init trust: sapu beberapa nilai trust awal (reward fairness TETAP
default, alpha_gini=0.5/alpha_flock=0.3 -- sesuai temuan run_test_30d_bootstrap_experiments.py
bahwa trust tinggi SENDIRIAN sudah menaikkan acceptance & memperbaiki Gini tanpa perlu
melonggarkan fairness). 30 hari/seed=0, dataset & arsitektur identik dgn eksperimen sebelumnya.

Tujuan: apakah manfaat trust tinggi LINEAR terhadap nilainya, atau ada titik jenuh/ambang.
"""
import json

import numpy as np

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator

DATASET = "scenario_dataset.json"
SEED = 0
CHUNK = 96
N_UPDATES = 30
TRUST_LEVELS = [0.5, 0.65, 0.8, 0.9, 1.0]


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def run_variant(init_trust):
    print(f"\n{'=' * 70}\n=== init_trust={init_trust} ===\n{'=' * 70}")
    rc = RewardCalculator(alpha_gini=0.5, alpha_flock=0.3)
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, reward_calc=rc, honest_estwait=True)
    forecaster = FormulaForecaster()
    sim = tr._fresh_sim()
    for u in sim.users:
        u.trust = float(init_trust)

    agent = RLRolloutAgent(tr.policy, sim, tr.rc, forecaster, tr.k, honest_estwait=tr.honest_estwait)

    day_records = []
    step = 0
    for day in range(N_UPDATES):
        boundary = False
        for _ in range(CHUNK):
            sim.step_once(step, agent=agent)
            step += 1
            if step >= sim.max_steps:
                boundary = True
                break
        if boundary:
            for t in agent.transitions:
                t.resolved = True
        resolved = [t for t in agent.transitions if t.resolved]
        pending = [t for t in agent.transitions if not t.resolved]

        if resolved:
            stats = tr.ppo.update(resolved)
            rewards = np.array([t.reward for t in resolved])
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            trust_vals = [u.trust for u in sim.users]
            day_rec = {
                "day": day,
                "mean_reward": float(rewards.mean()),
                "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                "mean_trust_allusers": float(np.mean(trust_vals)),
                "gini_served_cum": _gini(served),
                "total_served_cum": int(served.sum()),
                "entropy_final": stats.get("entropy_final", 0.0),
                "explained_var": stats.get("explained_var", 0.0),
            }
            day_records.append(day_rec)
            print(f"  [Hari {day + 1:02d}/{N_UPDATES}] R={day_rec['mean_reward']:+.4f} "
                 f"accept={day_rec['acceptance_rate']:.2f} trust={day_rec['mean_trust_allusers']:.3f} "
                 f"gini_served={day_rec['gini_served_cum']:.3f} served={day_rec['total_served_cum']}")

        if boundary:
            break
        agent.transitions = pending

    n_report = min(5, len(day_records))
    served_final = np.array([s.total_served for s in sim.spklus.values()], float)
    result = {
        "init_trust": init_trust,
        "reward_last%d" % n_report: float(np.mean([d["mean_reward"] for d in day_records[-n_report:]])),
        "acceptance_first%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[:n_report]])),
        "acceptance_last%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[-n_report:]])),
        "trust_last%d" % n_report: float(np.mean([d["mean_trust_allusers"] for d in day_records[-n_report:]])),
        "gini_served_final": _gini(served_final),
        "herding_events": sim.herding_events,
        "total_served_final": int(served_final.sum()),
        "entropy_last%d" % n_report: float(np.mean([d["entropy_final"] for d in day_records[-n_report:]])),
        "day_records": day_records,
    }
    return result


def main():
    all_results = {}
    for lvl in TRUST_LEVELS:
        all_results[str(lvl)] = run_variant(lvl)

    with open("test_30d_trust_sweep_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'=' * 100}\n=== KURVA DOSIS-RESPON INIT TRUST ===\n{'=' * 100}")
    cols = ["acceptance_last5", "trust_last5", "gini_served_final", "herding_events",
           "total_served_final", "reward_last5", "entropy_last5"]
    header = f"{'init_trust':<12}" + "".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))
    for lvl in TRUST_LEVELS:
        r = all_results[str(lvl)]
        row = f"{lvl:<12}"
        for c in cols:
            v = r.get(c)
            row += f"{v:>18.4f}" if isinstance(v, float) else f"{str(v):>18}"
        print(row)

    print("\n[INFO] Ringkasan lengkap -> test_30d_trust_sweep_summary.json")


if __name__ == "__main__":
    main()
