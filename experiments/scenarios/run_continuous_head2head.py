import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Head-to-head di TIMELINE KONTINU: apakah MARL (yang trust-nya kini tumbuh ke ~0,79, §10)
menyaingi/mengalahkan greedy S1 pada acceptance & wait, sambil unggul di Gini?

Perbandingan lama (run_4method_trust05_60d.py) memakai evaluasi single-horizon trust-reset,
sehingga acceptance MARL (~0,34) tertinggal. TAPI di timeline kontinu trust menumpuk untuk
SEMUA metode yang memberi rekomendasi (kepatuhan user = trust x w_i), termasuk S1. Head-to-head
yang ADIL harus menjalankan semua metode pada timeline kontinu yang sama.

Metode (semua: trust DIBAWA lintas-pass, dataset 5x, 10 pass x 30 hari):
  S0  no-intervention   : tanpa agen (acuan Gini/wait alami; acceptance tak berlaku)
  S1  greedy least-loaded: GreedyAgent (herding alfabetis -> Gini buruk secara teori)
  S3  OP-SRL             : OPSRLAgent + kepatuhan penuh eksogen (batas atas jika selalu dipatuhi)
  S4  MARL              : RollingBaseCriticAgent, dilatih PPO (ent_coef=0,3, rolling-Gini, lambda=0,25)

Metrik seragam dari sim tiap akhir pass (bukan bookkeeping agen): acceptance (compliance_history
user), trust mean, Gini atas total_served, wait mean, herding. NOL edit ke marl_spklu/.
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
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.agents.opsrl_agent import OPSRLAgent
from marl_spklu.experiments.harness import force_full_compliance
from marl_spklu.experiments import metrics as M

from run_rolling_gini_reward import RollingBaseCriticAgent

DATASET = "scenario_dataset_5x.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
ENT_COEF = 0.3
LAMBDA = 0.25
INIT_TRUST = 0.5


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def build_sim(trust_carry=None):
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)
    for u in sim.users:
        u.trust = trust_carry.get(u.user_id, INIT_TRUST) if trust_carry else INIT_TRUST
    return sim


def _pass_metrics(sim, pass_idx):
    served = np.array([s.total_served for s in sim.spklus.values()], float)
    waits = [L["wait_time"] for L in sim.logs]
    return {
        "pass": pass_idx,
        "acceptance": M.overall_acceptance_rate(sim.users),
        "trust_end": float(np.mean([u.trust for u in sim.users])),
        "gini_served": _gini(served),
        "mean_wait": float(np.mean(waits)) if waits else 0.0,
        "herding": sim.herding_events,
    }


def _run_pass(sim, agent, force_compliance, ppo=None):
    """Jalankan satu pass (30 hari). Kalau ppo != None -> latih MARL. Return None."""
    step = 0
    for _ in range(DAYS_PER_PASS):
        boundary = False
        for _ in range(CHUNK):
            if force_compliance:
                with force_full_compliance():
                    sim.step_once(step, agent=agent)
            else:
                sim.step_once(step, agent=agent)
            step += 1
            if step >= sim.max_steps:
                boundary = True
                break
        if ppo is not None:
            if boundary:
                for t in agent.transitions:
                    t.resolved = True
            resolved = [t for t in agent.transitions if t.resolved]
            pending = [t for t in agent.transitions if not t.resolved]
            if resolved:
                ppo.update(resolved)
            if boundary:
                break
            agent.transitions = pending
        elif boundary:
            break


def run_method(name, kind):
    """kind: 's0' | 's1' | 's3' | 's4'."""
    print(f"\n{'=' * 66}\n=== {name} ({kind}) -- timeline kontinu ===\n{'=' * 66}")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    policy = ppo = forecaster = rc = None
    if kind == "s4":
        N = len(build_sim().spklus)
        policy = HPPOPolicy(4 + 4 * N, 2 * N + 1, N, delta_max=10.0)
        ppo = PPOTrainer(policy, ent_coef=ENT_COEF)
        rc = RewardCalculator(alpha_wait=1.0, beta_prox=0.1, alpha_honesty=1.0,
                              alpha_gini=0.5 * LAMBDA, alpha_flock=0.3 * LAMBDA)
        forecaster = FormulaForecaster()

    trust_carry = None
    pass_records = []
    for pass_idx in range(N_PASSES):
        sim = build_sim(trust_carry)
        if kind == "s0":
            agent, force = None, False
        elif kind == "s1":
            agent, force = GreedyAgent(), False
        elif kind == "s3":
            agent, force = OPSRLAgent(), True
        else:  # s4
            agent, force = RollingBaseCriticAgent(policy, sim, rc, forecaster, k=3, honest_estwait=True), False

        _run_pass(sim, agent, force, ppo=ppo if kind == "s4" else None)
        trust_carry = {u.user_id: float(u.trust) for u in sim.users}
        pr = _pass_metrics(sim, pass_idx)
        pass_records.append(pr)
        print(f"  [Pass {pass_idx+1:02d}] accept={pr['acceptance']:.3f} trust={pr['trust_end']:.3f} "
             f"gini={pr['gini_served']:.3f} wait={pr['mean_wait']:.1f} herd={pr['herding']}")

    last5 = pass_records[-5:]
    return {
        "name": name, "kind": kind,
        "acceptance": float(np.mean([p["acceptance"] for p in last5])),
        "trust": float(np.mean([p["trust_end"] for p in last5])),
        "gini": float(np.mean([p["gini_served"] for p in last5])),
        "wait": float(np.mean([p["mean_wait"] for p in last5])),
        "herding": float(np.mean([p["herding"] for p in last5])),
        "trust_per_pass": [p["trust_end"] for p in pass_records],
        "acceptance_per_pass": [p["acceptance"] for p in pass_records],
        "gini_per_pass": [p["gini_served"] for p in pass_records],
        "wait_per_pass": [p["mean_wait"] for p in pass_records],
    }


def main():
    results = [
        run_method("S0_no_intervention", "s0"),
        run_method("S1_greedy",          "s1"),
        run_method("S3_opsrl",           "s3"),
        run_method("S4_marl",            "s4"),
    ]
    with open("test_continuous_head2head_summary.json", "w") as f:
        json.dump({"config": {"dataset": DATASET, "ent_coef": ENT_COEF, "lambda": LAMBDA,
                              "n_passes": N_PASSES, "days_per_pass": DAYS_PER_PASS, "seed": SEED,
                              "note": "trust dibawa lintas-pass utk SEMUA metode"},
                  "methods": results}, f, indent=2)

    print(f"\n\n{'=' * 84}\n=== HEAD-TO-HEAD TIMELINE KONTINU (rata-rata 5 pass terakhir) ===\n{'=' * 84}")
    print(f"{'metode':<20}{'accept':>9}{'trust':>8}{'gini':>8}{'wait':>8}{'herding':>9}")
    print("-" * 62)
    for r in results:
        print(f"{r['name']:<20}{r['acceptance']:>9.3f}{r['trust']:>8.3f}{r['gini']:>8.3f}"
             f"{r['wait']:>8.1f}{r['herding']:>9.0f}")
    print("\n(S0 acceptance tak berlaku: tanpa rekomendasi. Fokus: S4 vs S1 pada accept/wait/gini.)")
    print("\n[INFO] Ringkasan -> test_continuous_head2head_summary.json")


if __name__ == "__main__":
    main()
