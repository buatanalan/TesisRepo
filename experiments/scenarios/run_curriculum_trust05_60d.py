import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Curriculum fairness DI BAWAH TRUST 0.5 NATURAL (belum pernah diuji -- semua curriculum
sebelumnya pakai trust=1.0 bootstrap). 60d x 10 pass, reset per-pass, rolling-Gini + Formula
+ critic baseline.

Mekanisme yg diuji: dgn reset-per-pass, curriculum per-pass = "trust-first DI DALAM tiap pass"
-- tiap pass mulai fairness=0 (agen bangun kegunaan+trust lewat rekomendasi bagus yg dipatuhi),
fairness naik bertahap setelah trust sedikit terbentuk. Ini versi trust-first yg kompatibel
dgn reset-per-pass.

  A_static_full     : alpha_gini=0.5, alpha_flock=0.3 KONSTAN sepanjang tiap pass
  B_curriculum       : alpha ramp 0->penuh sepanjang tiap pass (reset ke 0 tiap pass baru)

Dilacak: progresi trust dalam tiap pass (hari 1 / tengah / akhir) utk cek apakah fairness-off
di awal benar2 membangun trust lebih tinggi.
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

DATASET = "scenario_dataset_60d.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 60
N_PASSES = 10
FULL_ALPHA_GINI = 0.5
FULL_ALPHA_FLOCK = 0.3


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


def run_variant(name, fairness_schedule):
    print(f"\n{'=' * 70}\n=== {name} (trust 0.5 natural, {N_PASSES}p x {DAYS_PER_PASS}d) ===\n{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    N = len(fresh_sim().spklus)
    policy = HPPOPolicy(4 + 4 * N, 2 * N + 1, N, delta_max=10.0)
    ppo = PPOTrainer(policy)
    rc = RewardCalculator()
    forecaster = FormulaForecaster()

    pass_records = []
    for pass_idx in range(N_PASSES):
        sim = fresh_sim()
        agent = RollingBaseCriticAgent(policy, sim, rc, forecaster, k=3, honest_estwait=True)
        step = 0
        accepts = []
        trust_trace = {}
        for day_in_pass in range(DAYS_PER_PASS):
            ag, af = fairness_schedule(day_in_pass)
            rc.alpha_gini = ag
            rc.alpha_flock = af
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
            if day_in_pass in (0, DAYS_PER_PASS // 2, DAYS_PER_PASS - 1):
                trust_trace[day_in_pass] = float(np.mean([u.trust for u in sim.users]))
            if boundary:
                break
            agent.transitions = pending

        served = np.array([s.total_served for s in sim.spklus.values()], float)
        pr = {
            "pass": pass_idx,
            "acceptance_mean": float(np.mean(accepts)),
            "gini_end": _gini(served),
            "trust_day0": trust_trace.get(0),
            "trust_mid": trust_trace.get(DAYS_PER_PASS // 2),
            "trust_end": trust_trace.get(DAYS_PER_PASS - 1),
            "herding_events": sim.herding_events,
        }
        pass_records.append(pr)
        print(f"  [Pass {pass_idx+1:02d}] accept={pr['acceptance_mean']:.3f} gini={pr['gini_end']:.3f} "
             f"trust {pr['trust_day0']:.3f}->{pr['trust_mid']:.3f}->{pr['trust_end']:.3f} "
             f"herding={pr['herding_events']}")

    return {
        "variant": name,
        "acceptance_per_pass": [p["acceptance_mean"] for p in pass_records],
        "gini_per_pass": [p["gini_end"] for p in pass_records],
        "trust_end_per_pass": [p["trust_end"] for p in pass_records],
        "herding_pass10": pass_records[-1]["herding_events"],
        "pass_records": pass_records,
    }


def main():
    results = {
        "A_static_full": run_variant("A_static_full",
                                    lambda d: (FULL_ALPHA_GINI, FULL_ALPHA_FLOCK)),
        "B_curriculum": run_variant("B_curriculum",
                                   lambda d: (FULL_ALPHA_GINI * d / (DAYS_PER_PASS - 1),
                                              FULL_ALPHA_FLOCK * d / (DAYS_PER_PASS - 1))),
    }
    with open("test_curriculum_trust05_60d_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 100}\n=== CURRICULUM @ TRUST 0.5 NATURAL (60d x 10pass) ===\n{'=' * 100}")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  acceptance/pass: {['%.3f'%a for a in r['acceptance_per_pass']]}")
        print(f"  gini/pass      : {['%.3f'%g for g in r['gini_per_pass']]}")
        print(f"  trust_end/pass : {['%.3f'%t for t in r['trust_end_per_pass']]}")
        print(f"  gini pass1->pass10: {r['gini_per_pass'][0]:.3f}->{r['gini_per_pass'][-1]:.3f}  "
             f"herding_pass10={r['herding_pass10']}")
    print("\n[INFO] Ringkasan -> test_curriculum_trust05_60d_summary.json")


if __name__ == "__main__":
    main()
