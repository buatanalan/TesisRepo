import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Validasi MULTI-SEED konfigurasi rekomendasi (LAPORAN_KONFIGURASI_REWARD.md §6) untuk
memastikan hasil robust (bukan kebetulan seed tunggal).

Konfigurasi terkunci:
  ent_coef=0.3, dataset=5x-interaksi, Formula, rolling-Gini, critic baseline, actor lokal,
  trust 0.5 natural, lambda=0.25 (alpha_gini=0.125, alpha_flock=0.075; suku individual dikunci).

Seed {0,1,2,3,4} -> laporkan mean +/- std dari metrik (acceptance, trust, gini, wait, herding).
Seluruh sumber acak diseed: torch, np.random, random.
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

DATASET = "scenario_dataset_5x.json"
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
ENT_COEF = 0.3
LAMBDA = 0.25
SEEDS = [0, 1, 2, 3, 4]


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def fresh_sim():
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)   # trust 0.5 natural
    return sim


def run_seed(seed):
    print(f"\n{'=' * 56}\n=== SEED {seed} ===\n{'=' * 56}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    N = len(fresh_sim().spklus)
    policy = HPPOPolicy(4 + 4 * N, 2 * N + 1, N, delta_max=10.0)
    ppo = PPOTrainer(policy, ent_coef=ENT_COEF)
    rc = RewardCalculator(alpha_wait=1.0, beta_prox=0.1, alpha_honesty=1.0,
                          alpha_gini=0.5 * LAMBDA, alpha_flock=0.3 * LAMBDA)
    forecaster = FormulaForecaster()

    pass_records = []
    for pass_idx in range(N_PASSES):
        sim = fresh_sim()
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
        served = np.array([s.total_served for s in sim.spklus.values()], float)
        waits = [L["wait_time"] for L in sim.logs]
        pass_records.append({
            "acceptance": float(np.mean(accepts)),
            "trust_end": float(np.mean([u.trust for u in sim.users])),
            "gini_served": _gini(served),
            "mean_wait": float(np.mean(waits)) if waits else 0.0,
            "herding": sim.herding_events,
        })
        pr = pass_records[-1]
        print(f"  [Pass {pass_idx+1:02d}] accept={pr['acceptance']:.3f} trust={pr['trust_end']:.3f} "
             f"gini={pr['gini_served']:.3f} wait={pr['mean_wait']:.1f}")

    last5 = pass_records[-5:]
    return {
        "seed": seed,
        "acceptance": float(np.mean([p["acceptance"] for p in last5])),
        "trust": float(np.mean([p["trust_end"] for p in last5])),
        "gini": float(np.mean([p["gini_served"] for p in last5])),
        "wait": float(np.mean([p["mean_wait"] for p in last5])),
        "herding": float(np.mean([p["herding"] for p in last5])),
        "gini_per_pass": [p["gini_served"] for p in pass_records],
    }


def main():
    results = [run_seed(s) for s in SEEDS]
    with open("test_multiseed_validation_summary.json", "w") as f:
        json.dump({"config": {"dataset": DATASET, "ent_coef": ENT_COEF, "lambda": LAMBDA,
                              "n_passes": N_PASSES, "days_per_pass": DAYS_PER_PASS, "seeds": SEEDS},
                  "per_seed": results}, f, indent=2)

    print(f"\n\n{'=' * 80}\n=== VALIDASI MULTI-SEED (5 seed, konfigurasi rekomendasi) ===\n{'=' * 80}")
    print(f"{'seed':<6}{'accept':>9}{'trust':>8}{'gini':>8}{'wait':>8}{'herding':>9}")
    print("-" * 48)
    for r in results:
        print(f"{r['seed']:<6}{r['acceptance']:>9.3f}{r['trust']:>8.3f}{r['gini']:>8.3f}"
             f"{r['wait']:>8.1f}{r['herding']:>9.0f}")
    print("-" * 48)
    for key, label in [("acceptance", "acceptance"), ("trust", "trust"), ("gini", "GINI"),
                       ("wait", "wait"), ("herding", "herding")]:
        vals = np.array([r[key] for r in results])
        print(f"{label:<20} mean={vals.mean():.3f}  std={vals.std():.3f}  "
             f"[{vals.min():.3f}, {vals.max():.3f}]")
    print("\n[INFO] Ringkasan -> test_multiseed_validation_summary.json")


if __name__ == "__main__":
    main()
