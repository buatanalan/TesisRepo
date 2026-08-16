import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Curriculum learning: bukan trust ATAU fairness statis sepanjang horizon, tapi bobot
fairness (alpha_gini, alpha_flock) di-NAIKKAN BERTAHAP dari 0 (hari 1) ke nilai penuh
(hari 30) -- sementara trust awal tetap tinggi (pemenang tunggal dari trust-sweep).

Rasional: biarkan agen dulu membangun kegunaan (wait-improvement, compliance) SEBELUM
dibebani tekanan pemerataan -- beda dari varian "combined" sebelumnya yang mematikan
fairness SELAMANYA (terbukti lebih buruk dari trust-tinggi-saja). Di sini fairness cuma
DITUNDA, diperkenalkan bertahap begitu agen sudah punya pengaruh (trust tinggi ->
weight_ai besar) utk benar-benar bisa memakainya.

4 varian dibandingkan (semua 30 hari/seed=0, dataset & arsitektur identik):
  1. baseline            : trust=0.5, fairness penuh statis (alpha_gini=0.5, alpha_flock=0.3)
  2. high_trust_static   : trust=1.0, fairness penuh statis SEJAK HARI 1
  3. fairness_off_static : trust=1.0, fairness OFF SELAMANYA (kontrol negatif)
  4. curriculum          : trust=1.0, fairness naik LINEAR 0 -> penuh dari hari 1 ke 30
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
FULL_ALPHA_GINI = 0.5
FULL_ALPHA_FLOCK = 0.3


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def run_variant(name, init_trust, fairness_schedule):
    """fairness_schedule(day) -> (alpha_gini, alpha_flock) utk hari tsb (0-indexed)."""
    print(f"\n{'=' * 70}\n=== VARIAN: {name} (init_trust={init_trust}) ===\n{'=' * 70}")
    rc = RewardCalculator(alpha_gini=0.0, alpha_flock=0.0)   # nilai awal, diubah tiap hari
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, reward_calc=rc, honest_estwait=True)
    forecaster = FormulaForecaster()
    sim = tr._fresh_sim()
    if init_trust is not None:
        for u in sim.users:
            u.trust = float(init_trust)

    agent = RLRolloutAgent(tr.policy, sim, tr.rc, forecaster, tr.k, honest_estwait=tr.honest_estwait)

    day_records = []
    step = 0
    for day in range(N_UPDATES):
        ag, af = fairness_schedule(day)
        tr.rc.alpha_gini = ag
        tr.rc.alpha_flock = af

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
                "day": day, "alpha_gini": ag, "alpha_flock": af,
                "mean_reward": float(rewards.mean()),
                "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                "mean_trust_allusers": float(np.mean(trust_vals)),
                "gini_served_cum": _gini(served),
                "total_served_cum": int(served.sum()),
                "entropy_final": stats.get("entropy_final", 0.0),
            }
            day_records.append(day_rec)
            print(f"  [Hari {day + 1:02d}/{N_UPDATES}] a_gini={ag:.3f} a_flock={af:.3f} "
                 f"R={day_rec['mean_reward']:+.4f} accept={day_rec['acceptance_rate']:.2f} "
                 f"trust={day_rec['mean_trust_allusers']:.3f} gini_served={day_rec['gini_served_cum']:.3f} "
                 f"served={day_rec['total_served_cum']}")

        if boundary:
            break
        agent.transitions = pending

    n_report = min(5, len(day_records))
    served_final = np.array([s.total_served for s in sim.spklus.values()], float)
    result = {
        "variant": name, "init_trust": init_trust,
        "acceptance_first%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[:n_report]])),
        "acceptance_last%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[-n_report:]])),
        "trust_last%d" % n_report: float(np.mean([d["mean_trust_allusers"] for d in day_records[-n_report:]])),
        "gini_served_final": _gini(served_final),
        "herding_events": sim.herding_events,
        "total_served_final": int(served_final.sum()),
        "reward_last%d" % n_report: float(np.mean([d["mean_reward"] for d in day_records[-n_report:]])),
        "entropy_last%d" % n_report: float(np.mean([d["entropy_final"] for d in day_records[-n_report:]])),
        "day_records": day_records,
    }
    return result


def main():
    variants = {
        "baseline": (0.5, lambda day: (FULL_ALPHA_GINI, FULL_ALPHA_FLOCK)),
        "high_trust_static": (1.0, lambda day: (FULL_ALPHA_GINI, FULL_ALPHA_FLOCK)),
        "fairness_off_static": (1.0, lambda day: (0.0, 0.0)),
        "curriculum": (1.0, lambda day: (FULL_ALPHA_GINI * day / (N_UPDATES - 1),
                                         FULL_ALPHA_FLOCK * day / (N_UPDATES - 1))),
    }

    all_results = {}
    for name, (init_trust, sched) in variants.items():
        all_results[name] = run_variant(name, init_trust, sched)

    with open("test_30d_curriculum_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'=' * 100}\n=== TABEL PERBANDINGAN AKHIR ===\n{'=' * 100}")
    cols = ["acceptance_last5", "trust_last5", "gini_served_final", "herding_events",
           "total_served_final", "reward_last5", "entropy_last5"]
    header = f"{'variant':<22}" + "".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        row = f"{name:<22}"
        for c in cols:
            v = r.get(c)
            row += f"{v:>18.4f}" if isinstance(v, float) else f"{str(v):>18}"
        print(row)

    print("\n[INFO] Ringkasan lengkap -> test_30d_curriculum_summary.json")


if __name__ == "__main__":
    main()
