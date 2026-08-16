import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Uji HORIZON PANJANG (10 pass x 30 hari = 300 hari) config yg direkomendasikan:
FormulaForecaster + critic DIPERKAYA, vs critic baseline sbg pembanding. Tujuan: apakah
keunggulan explained-variance critic diperkaya (terbukti di run 30-hari) akhirnya menerjemah
jadi acceptance/Gini yg lebih baik begitu diberi waktu training cukup panjang.

Multi-pass: tiap pass = sim baru (trust user di-reset ke 1.0, state segar), TAPI bobot policy
& PPO DIBAWA lintas pass (tak di-reset). Actor TETAP lokal (tak disentuh). FormulaForecaster
(dipertahankan krn terbukti tak kalah dari LearnedForecaster di setup ini -- lihat
test_integrated_stack_summary.json).
"""
import json
import math

import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.user import UserState
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.ppo import PPOTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator, _gini as _gini_calc

DATASET = "scenario_dataset.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
INIT_TRUST = 1.0


class EnhancedCriticAgent(RLRolloutAgent):
    def _build_critic_obs(self):
        utilisasi = np.array([self.sim.spklus[s].get_utilization() for s in self.sids])
        antrean = np.array([self.sim.spklus[s].get_queue_length() for s in self.sids]) / self.q_scale
        gini_u = _gini_calc(utilisasi)
        remaining = []
        for s in self.sids:
            times = [ev["remaining_time"] for c in self.sim.spklus[s].charging.values() for ev in c]
            remaining.append(float(np.mean(times)) if times else 0.0)
        remaining = np.array(remaining) / self.wait_scale
        traveling = np.zeros(self.N)
        for u in self.sim.users:
            if u.state == UserState.TRAVELING and u.target_spklu in self.sid_to_idx:
                traveling[self.sid_to_idx[u.target_spklu]] += 1
        traveling = traveling / self.q_scale
        time_now = self.sim.current_step * self.sim.dt_minutes
        hour = (time_now / 60.0) % 24.0
        dow = (time_now / (60.0 * 24.0)) % 7.0
        temporal = np.array([math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
                            math.sin(2 * math.pi * dow / 7.0), math.cos(2 * math.pi * dow / 7.0)])
        return np.concatenate([utilisasi, antrean, [gini_u], remaining, traveling, temporal]).astype(np.float32)


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def build_sim():
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)
    for u in sim.users:
        u.trust = INIT_TRUST
    return sim


def run_variant(name, agent_cls, critic_obs_dim_fn):
    print(f"\n{'=' * 70}\n=== VARIAN: {name} ({N_PASSES} pass x {DAYS_PER_PASS} hari) ===\n{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    sim0 = build_sim()
    N = len(sim0.spklus)
    policy = HPPOPolicy(4 + 4 * N, critic_obs_dim_fn(N), N, delta_max=10.0)
    ppo = PPOTrainer(policy)
    rc = RewardCalculator()
    forecaster = FormulaForecaster()

    day_records = []
    pass_records = []
    for pass_idx in range(N_PASSES):
        sim = build_sim()
        agent = agent_cls(policy, sim, rc, forecaster, k=3, honest_estwait=True)
        step = 0
        for day_in_pass in range(DAYS_PER_PASS):
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
                stats = ppo.update(resolved)
                rewards = np.array([t.reward for t in resolved])
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                day_records.append({
                    "pass": pass_idx, "day_in_pass": day_in_pass,
                    "mean_reward": float(rewards.mean()),
                    "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                    "gini_served_cum": _gini(served),
                    "explained_var": stats.get("explained_var", 0.0),
                })
            if boundary:
                break
            agent.transitions = pending

        recs_this = [d for d in day_records if d["pass"] == pass_idx]
        served_pass = np.array([s.total_served for s in sim.spklus.values()], float)
        pr = {
            "pass": pass_idx,
            "acceptance_mean": float(np.mean([d["acceptance_rate"] for d in recs_this])),
            "gini_end": _gini(served_pass),
            "explained_var_mean": float(np.mean([d["explained_var"] for d in recs_this])),
            "herding_events": sim.herding_events,
        }
        pass_records.append(pr)
        print(f"  [Pass {pass_idx+1:02d}/{N_PASSES}] accept={pr['acceptance_mean']:.3f} "
             f"gini_end={pr['gini_end']:.3f} EV={pr['explained_var_mean']:+.3f} herding={pr['herding_events']}")

    return {
        "variant": name,
        "acceptance_pass1": pass_records[0]["acceptance_mean"],
        "acceptance_pass10": pass_records[-1]["acceptance_mean"],
        "gini_pass1": pass_records[0]["gini_end"],
        "gini_pass10": pass_records[-1]["gini_end"],
        "ev_pass1": pass_records[0]["explained_var_mean"],
        "ev_pass10": pass_records[-1]["explained_var_mean"],
        "herding_pass10": pass_records[-1]["herding_events"],
        "acceptance_per_pass": [p["acceptance_mean"] for p in pass_records],
        "gini_per_pass": [p["gini_end"] for p in pass_records],
        "ev_per_pass": [p["explained_var_mean"] for p in pass_records],
        "pass_records": pass_records,
    }


def main():
    results = {
        "baseline_critic": run_variant("baseline_critic", RLRolloutAgent, lambda N: 2 * N + 1),
        "enhanced_critic": run_variant("enhanced_critic", EnhancedCriticAgent, lambda N: 4 * N + 5),
    }
    with open("test_enhanced_critic_longhorizon_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 100}\n=== CRITIC baseline vs diperkaya @ 300 HARI (10 pass) ===\n{'=' * 100}")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  acceptance/pass: {['%.3f'%a for a in r['acceptance_per_pass']]}")
        print(f"  gini/pass      : {['%.3f'%g for g in r['gini_per_pass']]}")
        print(f"  expl_var/pass  : {['%.2f'%e for e in r['ev_per_pass']]}")
        print(f"  pass1->pass10  : accept {r['acceptance_pass1']:.3f}->{r['acceptance_pass10']:.3f}  "
             f"gini {r['gini_pass1']:.3f}->{r['gini_pass10']:.3f}  EV {r['ev_pass1']:.2f}->{r['ev_pass10']:.2f}  "
             f"herding_pass10={r['herding_pass10']}")
    print("\n[INFO] Ringkasan lengkap -> test_enhanced_critic_longhorizon_summary.json")


if __name__ == "__main__":
    main()
