import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Validasi LINTAS-DATASET temuan §10 (timeline kontinu menembus plateau trust tanpa merusak
Gini). Menjalankan dua varian INTI -- reset_baseline vs continuous -- pada tiga dataset
berbeda topologi/populasi (§9 batasan: sebelumnya hanya dataset 5x seed-42).

Datasets:
  5x   : scenario_dataset_5x.json   (seed 42, 8 SPKLU, rasio ~40%) -- referensi §10
  altA : scenario_dataset_altA.json (seed 7,  8 SPKLU, rasio ~40%) -- variasi seed murni
  altB : scenario_dataset_altB.json (seed 13, 12 SPKLU, rasio ~55%) -- topologi & beban beda

Konfigurasi rekomendasi identik (ent_coef=0,3, Formula, rolling-Gini, critic baseline, actor
lokal, lambda=0,25). Kalau di ketiga dataset 'continuous' konsisten menaikkan trust & acceptance
di atas 'reset' tanpa memperburuk Gini, temuan robust lintas dataset. NOL edit ke marl_spklu/.
"""
import json
import random

import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.ppo import PPOTrainer
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator

from run_rolling_gini_reward import RollingBaseCriticAgent

SEED = 0
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
ENT_COEF = 0.3
LAMBDA = 0.25
INIT_TRUST = 0.5

DATASETS = [
    ("5x",   "scenario_dataset_5x.json"),
    ("altA", "scenario_dataset_altA.json"),
    ("altB", "scenario_dataset_altB.json"),
]


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def build_sim(dataset, trust_carry=None):
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(dataset)
    for u in sim.users:
        u.trust = trust_carry.get(u.user_id, INIT_TRUST) if trust_carry else INIT_TRUST
    return sim


def run(dataset, carry_trust):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    N = len(build_sim(dataset).spklus)
    policy = HPPOPolicy(4 + 4 * N, 2 * N + 1, N, delta_max=10.0)
    ppo = PPOTrainer(policy, ent_coef=ENT_COEF)
    rc = RewardCalculator(alpha_wait=1.0, beta_prox=0.1, alpha_honesty=1.0,
                          alpha_gini=0.5 * LAMBDA, alpha_flock=0.3 * LAMBDA)
    forecaster = FormulaForecaster()

    trust_carry = None
    pass_records = []
    for pass_idx in range(N_PASSES):
        sim = build_sim(dataset, trust_carry if carry_trust else None)
        agent = RollingBaseCriticAgent(policy, sim, rc, forecaster, k=3, honest_estwait=True)
        step = 0
        accepts = []
        for _ in range(DAYS_PER_PASS):
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
                ppo.update(resolved)
                accepts.append(float(np.mean([t.complied for t in resolved])))
            if boundary:
                break
            agent.transitions = pending
        trust_carry = {u.user_id: float(u.trust) for u in sim.users}
        served = np.array([s.total_served for s in sim.spklus.values()], float)
        waits = [L["wait_time"] for L in sim.logs]
        pass_records.append({
            "acceptance": float(np.mean(accepts)),
            "trust_end": float(np.mean([u.trust for u in sim.users])),
            "gini_served": _gini(served),
            "mean_wait": float(np.mean(waits)) if waits else 0.0,
        })
    last5 = pass_records[-5:]
    return {
        "acceptance": float(np.mean([p["acceptance"] for p in last5])),
        "trust": float(np.mean([p["trust_end"] for p in last5])),
        "gini": float(np.mean([p["gini_served"] for p in last5])),
        "wait": float(np.mean([p["mean_wait"] for p in last5])),
        "trust_per_pass": [p["trust_end"] for p in pass_records],
    }


def main():
    results = {}
    for label, dataset in DATASETS:
        print(f"\n{'=' * 60}\n=== DATASET {label} ({dataset}) ===\n{'=' * 60}")
        for variant, carry in [("reset", False), ("continuous", True)]:
            r = run(dataset, carry)
            results[f"{label}_{variant}"] = {"dataset": label, "variant": variant, **r}
            print(f"  [{variant:<10}] accept={r['acceptance']:.3f} trust={r['trust']:.3f} "
                 f"gini={r['gini']:.3f} wait={r['wait']:.1f}  trust1->10={r['trust_per_pass'][0]:.3f}->{r['trust_per_pass'][-1]:.3f}")

    with open("test_crossval_timeline_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 88}\n=== VALIDASI LINTAS-DATASET: reset vs continuous (rata-rata 5 pass terakhir) ===\n{'=' * 88}")
    print(f"{'dataset':<8}{'varian':<12}{'accept':>9}{'trust':>8}{'gini':>8}{'wait':>8}{'d_accept':>10}{'d_trust':>9}")
    print("-" * 72)
    for label, _ in DATASETS:
        rr = results[f"{label}_reset"]; rc_ = results[f"{label}_continuous"]
        for v in (rr, rc_):
            print(f"{v['dataset']:<8}{v['variant']:<12}{v['acceptance']:>9.3f}{v['trust']:>8.3f}"
                 f"{v['gini']:>8.3f}{v['wait']:>8.1f}", end="")
            if v["variant"] == "continuous":
                print(f"{v['acceptance']-rr['acceptance']:>+10.3f}{v['trust']-rr['trust']:>+9.3f}")
            else:
                print(f"{'':>10}{'':>9}")
    print("\n[INFO] Ringkasan -> test_crossval_timeline_summary.json")


if __name__ == "__main__":
    main()
