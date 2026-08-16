import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Sweep bobot reward INDIVIDUAL (one-factor-at-a-time) untuk menutup pertanyaan
'apakah bobot reward berpengaruh, atau parameter struktural yang dominan?' (lihat
LAPORAN_KONFIGURASI_REWARD.md §8).

Dikunci (struktural, nilai terbaik tervalidasi): ent_coef=0.3, dataset 5x, Formula, rolling-Gini,
critic baseline, actor lokal, trust natural, lambda fairness=0.25 (alpha_gini=0.125, flock=0.075).

Disapu OFAT dari referensi (alpha_wait=1, beta_prox=0.1, alpha_honesty=1):
  alpha_wait     : 0, 1(ref), 3      (harusnya pengaruhi wait/steering)
  beta_prox       : 0, 0.1(ref), 0.5  (harusnya pengaruhi acceptance)
  alpha_honesty    : 0, 1(ref), 3      (harusnya pengaruhi trust)

seed=0 (varians antar-seed sudah terbukti sangat kecil, std~0.001 -> seed tunggal cukup utk
mendeteksi apakah suatu bobot berpengaruh; efek harus >> 0.005 utk nyata).
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
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
ENT_COEF = 0.3
LAMBDA = 0.25

RUNS = [
    ("ref",         dict(w=1.0, p=0.1, h=1.0)),
    ("wait_0",      dict(w=0.0, p=0.1, h=1.0)),
    ("wait_3",      dict(w=3.0, p=0.1, h=1.0)),
    ("prox_0",      dict(w=1.0, p=0.0, h=1.0)),
    ("prox_0.5",    dict(w=1.0, p=0.5, h=1.0)),
    ("honesty_0",   dict(w=1.0, p=0.1, h=0.0)),
    ("honesty_3",   dict(w=1.0, p=0.1, h=3.0)),
]


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def fresh_sim():
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)
    return sim


def run(name, wt):
    print(f"\n{'=' * 58}\n=== {name}  (wait={wt['w']} prox={wt['p']} honesty={wt['h']}) ===\n{'=' * 58}")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    N = len(fresh_sim().spklus)
    policy = HPPOPolicy(4 + 4 * N, 2 * N + 1, N, delta_max=10.0)
    ppo = PPOTrainer(policy, ent_coef=ENT_COEF)
    rc = RewardCalculator(alpha_wait=wt["w"], beta_prox=wt["p"], alpha_honesty=wt["h"],
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
    last5 = pass_records[-5:]
    r = {
        "name": name, "weights": wt,
        "acceptance": float(np.mean([p["acceptance"] for p in last5])),
        "trust": float(np.mean([p["trust_end"] for p in last5])),
        "gini": float(np.mean([p["gini_served"] for p in last5])),
        "wait": float(np.mean([p["mean_wait"] for p in last5])),
        "herding": float(np.mean([p["herding"] for p in last5])),
    }
    print(f"  -> accept={r['acceptance']:.3f} trust={r['trust']:.3f} gini={r['gini']:.3f} "
         f"wait={r['wait']:.1f} herding={r['herding']:.0f}")
    return r


def main():
    results = [run(name, wt) for name, wt in RUNS]
    with open("test_individual_weights_sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    ref = results[0]
    print(f"\n\n{'=' * 92}\n=== SWEEP BOBOT INDIVIDUAL (OFAT) -- delta vs referensi ===\n{'=' * 92}")
    print(f"{'run':<12}{'accept':>9}{'trust':>8}{'gini':>8}{'wait':>8}{'herding':>9}   {'|d_accept|':>10}")
    print("-" * 76)
    for r in results:
        d_acc = abs(r["acceptance"] - ref["acceptance"])
        print(f"{r['name']:<12}{r['acceptance']:>9.3f}{r['trust']:>8.3f}{r['gini']:>8.3f}"
             f"{r['wait']:>8.1f}{r['herding']:>9.0f}   {d_acc:>10.3f}")
    print(f"\n(Referensi = {ref['name']}: accept={ref['acceptance']:.3f} trust={ref['trust']:.3f} "
         f"gini={ref['gini']:.3f} wait={ref['wait']:.1f})")
    print("\n[INFO] Ringkasan -> test_individual_weights_sweep_summary.json")


if __name__ == "__main__":
    main()
