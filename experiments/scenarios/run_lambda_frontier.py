import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Langkah 3 metodologi tuning (lihat LAPORAN_KONFIGURASI_REWARD.md): telusuri Pareto frontier
acceptance<->Gini dengan menyapu SATU dial trade-off, parameter struktural dikunci di nilai
terbaik yang sudah ditemukan.

DIKUNCI (parameter struktural/gerbang):
  - ent_coef = 0.3        (lever entropi -> menjaga Gini rendah, terbukti)
  - dataset  = 5x-interaksi (scenario_dataset_5x.json, median 12 interaksi/user -> trust bisa tumbuh)
  - forecaster = Formula   (terbukti >= Learned)
  - reward Gini = rolling-window 24 jam
  - critic = baseline (enhanced terbukti memperburuk)
  - actor = LOKAL (tak diubah)
  - trust = 0.5 NATURAL (reset per-pass; 5x interaksi memberi ruang tumbuh dalam tiap pass)

DISAPU (dial trade-off): lambda menskalakan (alpha_gini, alpha_flock) BERSAMA.
  reward individual (alpha_wait=1, beta_prox=0.1, alpha_honesty=1) DIKUNCI sbg patokan rasio.
  lambda in {0, 0.25, 0.5, 1.0, 2.0} -> alpha_gini=0.5*lambda, alpha_flock=0.3*lambda.
"""
import json

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
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
BASE_ALPHA_GINI = 0.5
BASE_ALPHA_FLOCK = 0.3


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def fresh_sim():
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)   # trust 0.5 natural (tak di-bootstrap)
    return sim


def run(lam):
    print(f"\n{'=' * 60}\n=== lambda={lam} (a_gini={BASE_ALPHA_GINI*lam:.3f} a_flock={BASE_ALPHA_FLOCK*lam:.3f}) ===\n{'=' * 60}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    N = len(fresh_sim().spklus)
    policy = HPPOPolicy(4 + 4 * N, 2 * N + 1, N, delta_max=10.0)
    ppo = PPOTrainer(policy, ent_coef=ENT_COEF)
    rc = RewardCalculator(alpha_wait=1.0, beta_prox=0.1, alpha_honesty=1.0,
                          alpha_gini=BASE_ALPHA_GINI * lam, alpha_flock=BASE_ALPHA_FLOCK * lam)
    forecaster = FormulaForecaster()

    pass_records = []
    for pass_idx in range(N_PASSES):
        sim = fresh_sim()
        agent = RollingBaseCriticAgent(policy, sim, rc, forecaster, k=3, honest_estwait=True)
        step = 0
        accepts = []
        prev_logs = 0
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
            "pass": pass_idx,
            "acceptance": float(np.mean(accepts)),
            "trust_end": float(np.mean([u.trust for u in sim.users])),
            "gini_served": _gini(served),
            "herding": sim.herding_events,
            "mean_wait": float(np.mean(waits)) if waits else 0.0,
        })
        pr = pass_records[-1]
        print(f"  [Pass {pass_idx+1:02d}] accept={pr['acceptance']:.3f} trust={pr['trust_end']:.3f} "
             f"gini={pr['gini_served']:.3f} wait={pr['mean_wait']:.1f} herd={pr['herding']}")

    last5 = pass_records[-5:]
    return {
        "lambda": lam,
        "alpha_gini": BASE_ALPHA_GINI * lam, "alpha_flock": BASE_ALPHA_FLOCK * lam,
        "acceptance_last5": float(np.mean([p["acceptance"] for p in last5])),
        "trust_last5": float(np.mean([p["trust_end"] for p in last5])),
        "gini_last5": float(np.mean([p["gini_served"] for p in last5])),
        "wait_last5": float(np.mean([p["mean_wait"] for p in last5])),
        "herding_last5": float(np.mean([p["herding"] for p in last5])),
        "acceptance_per_pass": [p["acceptance"] for p in pass_records],
        "gini_per_pass": [p["gini_served"] for p in pass_records],
        "trust_per_pass": [p["trust_end"] for p in pass_records],
    }


def main():
    results = {}
    for lam in LAMBDAS:
        results[f"lambda_{lam}"] = run(lam)
    with open("test_lambda_frontier_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 90}\n=== PARETO FRONTIER acceptance<->Gini (5x dataset, ent_coef=0.3, trust natural) ===\n{'=' * 90}")
    print(f"{'lambda':<9}{'a_gini':>8}{'accept':>9}{'trust':>8}{'GINI':>8}{'wait':>8}{'herding':>9}")
    print("-" * 59)
    for name, r in results.items():
        print(f"{r['lambda']:<9}{r['alpha_gini']:>8.3f}{r['acceptance_last5']:>9.3f}{r['trust_last5']:>8.3f}"
             f"{r['gini_last5']:>8.3f}{r['wait_last5']:>8.1f}{r['herding_last5']:>9.0f}")
    print("\n[INFO] Ringkasan -> test_lambda_frontier_summary.json")


if __name__ == "__main__":
    main()
