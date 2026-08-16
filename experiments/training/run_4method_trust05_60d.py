import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "scenarios"))
"""Perbandingan 4-METODE versi diperbaiki (trust 0.5 NATURAL, reset per-pass, 60 hari x 10 pass):

  S0  Tanpa intervensi
  S1  Greedy least-loaded
  S3  OP-SRL (kepatuhan penuh eksogen)
  S4_base  MARL, critic BASELINE, reward rolling-Gini, Formula forecaster
  S4_enh    MARL, critic DIPERKAYA (traveling-EV+slot+temporal), reward rolling-Gini

MARL dilatih trust 0.5 NATURAL (TIDAK di-bootstrap 1.0) -- tiap pass sim baru dgn trust default
0.5, reset per-pass (sesuai keputusan user). Bobot policy dibawa lintas pass. Actor TETAP lokal.
Evaluasi lewat harness yg sama (KPI sebanding). Critic tak dipakai saat evaluasi (EvalAgent pass
critic_obs=None) -> policy base & enhanced dievaluasi seragam.
"""
import json

import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.ppo import PPOTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.experiments.harness import compare_scenarios, format_comparison

from run_rolling_gini_reward import RollingBaseCriticAgent, RollingEnhCriticAgent

DATASET = "scenario_dataset_60d.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 60
N_PASSES = 10
EVAL_SEEDS = 3
WILLINGNESS_RATIO = 5.0


def fresh_sim():
    """Trust 0.5 NATURAL (default User) -- TIDAK di-bootstrap."""
    sim = Simulator({}, [], None, user_willingness_radius_km=None,
                    user_willingness_ratio=WILLINGNESS_RATIO)
    sim.load_from_dataset(DATASET)
    return sim


def train_policy(name, agent_cls, critic_dim_fn):
    print(f"\n=== Melatih MARL: {name} ({N_PASSES} pass x {DAYS_PER_PASS} hari, trust 0.5 natural) ===")
    torch.manual_seed(SEED); np.random.seed(SEED)
    N = len(fresh_sim().spklus)
    policy = HPPOPolicy(4 + 4 * N, critic_dim_fn(N), N, delta_max=10.0)
    ppo = PPOTrainer(policy)
    rc = RewardCalculator()
    forecaster = FormulaForecaster()
    for pass_idx in range(N_PASSES):
        sim = fresh_sim()   # trust 0.5, reset per-pass
        agent = agent_cls(policy, sim, rc, forecaster, k=3, honest_estwait=True)
        step = 0
        pass_accepts = []
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
                pass_accepts.append(float(np.mean([t.complied for t in resolved])))
            if boundary:
                break
            agent.transitions = pending
        trust_end = float(np.mean([u.trust for u in sim.users]))
        print(f"  [Pass {pass_idx+1:02d}/{N_PASSES}] accept={np.mean(pass_accepts):.3f} trust_end={trust_end:.3f}")
    return policy, forecaster


class EvalAgent(RLRolloutAgent):
    """Agen evaluasi: critic TAK dipakai (return None) -> seragam utk policy base & enhanced,
    tanpa masalah dimensi critic_obs. Actor tetap lokal apa adanya."""
    def _build_critic_obs(self):
        return None


class _DeferredMarlAgent:
    def __init__(self, policy, forecaster, k=3, honest=True):
        self._policy = policy; self._forecaster = forecaster; self._k = k
        self._honest = honest; self._real = None

    def bind_to_sim(self, sim):
        self._real = EvalAgent(self._policy, sim, RewardCalculator(), self._forecaster,
                              k=self._k, honest_estwait=self._honest)

    def get_recommendation(self, feasible_spklus):
        return self._real.get_recommendation(feasible_spklus)

    def predict_waits(self, feasible_spklus):
        return self._real.predict_waits(feasible_spklus)


def main():
    policy_base, fc = train_policy("S4_base (critic baseline)", RollingBaseCriticAgent, lambda N: 2 * N + 1)
    policy_enh, _ = train_policy("S4_enh (critic diperkaya)", RollingEnhCriticAgent, lambda N: 4 * N + 5)

    extra = {
        "S4_base": (lambda: _DeferredMarlAgent(policy_base, fc), False),
        "S4_enh": (lambda: _DeferredMarlAgent(policy_enh, fc), False),
    }
    names = ["S0_no_intervention", "S1_greedy", "S3_opsrl", "S4_base", "S4_enh"]
    print(f"\n=== Evaluasi 5 skenario x {EVAL_SEEDS} seed (dataset 60d) ===")
    res = compare_scenarios(DATASET, scenario_names=names, seeds=range(EVAL_SEEDS),
                            willingness_ratio=WILLINGNESS_RATIO, extra_scenarios=extra)
    print("\n" + format_comparison(res))

    slim = {n: {k: v for k, v in agg.items() if k != "_runs"} for n, agg in res.items()}
    with open("main_experiment_trust05_60d_10pass.json", "w") as f:
        json.dump({"config": {"dataset": DATASET, "n_passes": N_PASSES, "days_per_pass": DAYS_PER_PASS,
                              "trust": "0.5 natural (reset per-pass)", "reward": "rolling-Gini",
                              "forecaster": "Formula", "eval_seeds": EVAL_SEEDS}, "comparison": slim}, f, indent=2)
    print("\n[INFO] Hasil -> main_experiment_trust05_60d_10pass.json")


if __name__ == "__main__":
    main()
