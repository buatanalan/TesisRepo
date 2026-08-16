import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Distribusi (mean/median/p90/std) tiap metrik untuk head-to-head timeline kontinu -- karena
rata-rata saja bisa menyembunyikan ekor (mis. wait p90, ketimpangan served antar-stasiun).

Menjalankan ulang 4 metode (S0/S1/S3/S4) di timeline kontinu (trust dibawa lintas-pass, identik
run_continuous_head2head.py), lalu pada PASS TERAKHIR mengumpulkan populasi mentah tiap metrik:
  - wait     : waktu tunggu SEMUA trip (menit)              -> distribusi per-trip
  - trust    : trust SEMUA user di akhir pass               -> distribusi per-user
  - served   : total_served SEMUA stasiun                   -> distribusi per-stasiun (basis Gini)
  - accept   : compliance-rate PER-USER (mean riwayat)      -> distribusi per-user
  - herd/step: jumlah rekomendasi ke SPKLU sama per step    -> distribusi per-step

Untuk tiap populasi dilaporkan mean, median, p90, std (+ p10 utk wait/served agar ekor bawah
terlihat). NOL edit ke marl_spklu/.
"""
import json
import random
from collections import Counter

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

from run_rolling_gini_reward import RollingBaseCriticAgent

DATASET = "scenario_dataset_5x.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
ENT_COEF = 0.3
LAMBDA = 0.25
INIT_TRUST = 0.5


def build_sim(trust_carry=None):
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)
    for u in sim.users:
        u.trust = trust_carry.get(u.user_id, INIT_TRUST) if trust_carry else INIT_TRUST
    return sim


def _stats(x, lo_tail=False):
    x = np.asarray(x, float)
    if x.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "std": 0.0, "n": 0}
    d = {"mean": float(x.mean()), "median": float(np.median(x)),
         "p90": float(np.percentile(x, 90)), "std": float(x.std()), "n": int(x.size)}
    if lo_tail:
        d["p10"] = float(np.percentile(x, 10))
    return d


def _run_pass(sim, agent, force, ppo=None):
    step = 0
    for _ in range(DAYS_PER_PASS):
        boundary = False
        for _ in range(CHUNK):
            if force:
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


def _distributions(sim):
    waits = [L["wait_time"] for L in sim.logs]
    trust = [u.trust for u in sim.users]
    served = [s.total_served for s in sim.spklus.values()]
    accept = [float(np.mean(u.compliance_history)) for u in sim.users if u.compliance_history]
    herd = [d.get("n_recs", 0) for d in sim.rec_distribution_log]  # rekomendasi/step (proksi tekanan herding)
    # herding "sesungguhnya": maksimum rekomendasi ke satu SPKLU per step
    max_same = []
    for d in sim.rec_distribution_log:
        counts = d.get("counts", {})
        if counts:
            max_same.append(max(counts.values()))
    return {
        "wait": _stats(waits, lo_tail=True),
        "trust": _stats(trust),
        "served": _stats(served, lo_tail=True),
        "accept_per_user": _stats(accept),
        "recs_per_step": _stats(herd),
        "max_same_spklu_per_step": _stats(max_same),
    }


def run_method(name, kind):
    print(f"  [{name}] menjalankan {N_PASSES} pass...")
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
    last_sim = None
    for pass_idx in range(N_PASSES):
        sim = build_sim(trust_carry)
        if kind == "s0":
            agent, force = None, False
        elif kind == "s1":
            agent, force = GreedyAgent(), False
        elif kind == "s3":
            agent, force = OPSRLAgent(), True
        else:
            agent, force = RollingBaseCriticAgent(policy, sim, rc, forecaster, k=3, honest_estwait=True), False
        _run_pass(sim, agent, force, ppo=ppo if kind == "s4" else None)
        trust_carry = {u.user_id: float(u.trust) for u in sim.users}
        last_sim = sim
    return {"name": name, "kind": kind, "dist": _distributions(last_sim)}


def _fmt(s, lo_tail=False):
    base = f"mean={s['mean']:>7.2f}  median={s['median']:>7.2f}  p90={s['p90']:>7.2f}  std={s['std']:>7.2f}"
    if lo_tail and "p10" in s:
        base = f"p10={s['p10']:>7.2f}  " + base
    return base


def main():
    methods = [("S0_no_intervention", "s0"), ("S1_greedy", "s1"),
               ("S3_opsrl", "s3"), ("S4_marl", "s4")]
    print("=== Mengumpulkan distribusi (pass terakhir, timeline kontinu) ===")
    results = [run_method(n, k) for n, k in methods]
    with open("test_head2head_distributions_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    metric_order = [
        ("wait", "WAIT (menit, per-trip)", True),
        ("served", "SERVED (per-stasiun)", True),
        ("trust", "TRUST (per-user)", False),
        ("accept_per_user", "ACCEPT (rate per-user)", False),
        ("max_same_spklu_per_step", "HERDING (max rec ke 1 SPKLU/step)", False),
    ]
    for key, title, lo in metric_order:
        print(f"\n{'=' * 96}\n=== {title} ===\n{'=' * 96}")
        for r in results:
            print(f"  {r['name']:<20} {_fmt(r['dist'][key], lo_tail=lo)}  (n={r['dist'][key]['n']})")
    print("\n[INFO] Ringkasan -> test_head2head_distributions_summary.json")


if __name__ == "__main__":
    main()
