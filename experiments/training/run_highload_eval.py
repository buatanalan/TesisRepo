import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""
run_highload_eval.py
Mengevaluasi S0 (baseline), S5_HONEST, S5_DISHONEST, S6_HONEST, S6_DISHONEST
pada scenario_test_highload.json (4x beban, 2x horizon = 2 hari).

Model di-TRAIN ulang pada scenario_train_1w.json (210 update) kemudian
diuji pada scenario_test_highload.json (max_steps=192).
"""
import json, os, random
import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.training import train_mode_a, train_mode_b, _fresh_sim
from marl_spklu.rl.rollout import evaluate_policy, RLRolloutAgent
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.agents.greedy_agent import GreedyAgent

TRAIN_DS  = "scenario_train_1w.json"
TEST_DS   = "scenario_test_highload.json"
SEED      = 0
UPDATES   = 210

def get_test_stats(sim):
    waits    = [log["wait_time"]   for log in sim.logs]
    trusts   = [log["trust_after"] for log in sim.logs]
    complied = [log["complied"]    for log in sim.logs]
    served   = np.array([s.total_served for s in sim.spklus.values()], float)
    def _gini(a):
        a = np.clip(a, 0, None)
        if a.sum() == 0: return 0.0
        a = np.sort(a); n = len(a); idx = np.arange(1, n+1)
        return float(np.sum((2*idx - n - 1)*a) / (n*a.sum()))
    return {
        "gini_served":     _gini(served),
        "total_served":    int(served.sum()),
        "n_active":        int((served > 0).sum()),
        "herding_events":  sim.herding_events,
        "mean_wait":       float(np.mean(waits))    if waits    else 0.0,
        "p90_wait":        float(np.quantile(waits, 0.9)) if waits else 0.0,
        "mean_trust":      float(np.mean(trusts))   if trusts   else 0.5,
        "acceptance_rate": float(np.mean(complied)) if complied else 0.0,
    }

def make_fresh_highload_sim():
    """Buat simulator baru dari high-load test dataset."""
    with open(TEST_DS) as f:
        ds = json.load(f)
    spklu_ids = [s["id"] for s in ds["spklus"]]
    sim = Simulator({}, [], HistoryBuffer(spklu_ids, window_size_15m=192), log_actor_states=True)
    sim.load_from_dataset(TEST_DS)
    return sim

def eval_on_highload(policy, forecaster, honest, max_steps=192):
    """Evaluasi policy pada high-load test dataset."""
    sim = make_fresh_highload_sim()
    evaluate_policy(sim, policy, forecaster, k=3, max_steps=max_steps, honest_estwait=honest)
    return get_test_stats(sim)

def main():
    if not os.path.exists(TRAIN_DS):
        print("ERROR: scenario_train_1w.json tidak ditemukan.")
        return
    if not os.path.exists(TEST_DS):
        print("ERROR: scenario_test_highload.json tidak ditemukan. Jalankan generate_highload_test.py.")
        return

    results = {}
    rc = RewardCalculator()

    # ── S0: Baseline tanpa rekomendasi ─────────────────────────────────────────
    print("\n" + "="*60)
    print("S0 — Baseline (tanpa rekomendasi)")
    print("="*60)
    sim_s0 = make_fresh_highload_sim()
    sim_s0.run(max_steps=192)
    results["S0"] = get_test_stats(sim_s0)
    for k, v in results["S0"].items(): print(f"  {k}: {v}")

    # ── S5_HONEST ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S5_HONEST — RL H-PPO Formula, honest=True, {UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr5h = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                   reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                   minibatch=64, epochs=4, honest_estwait=True)
    policy5h, forecaster5h = train_mode_a(TRAIN_DS, tr5h, total_updates=UPDATES)
    results["S5_HONEST"] = eval_on_highload(policy5h, forecaster5h, honest=True)
    print("  Metrik:")
    for k, v in results["S5_HONEST"].items(): print(f"    {k}: {v}")

    # ── S5_DISHONEST ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S5_DISHONEST — RL H-PPO Formula, honest=False, {UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr5d = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                   reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                   minibatch=64, epochs=4, honest_estwait=False)
    policy5d, forecaster5d = train_mode_a(TRAIN_DS, tr5d, total_updates=UPDATES)
    results["S5_DISHONEST"] = eval_on_highload(policy5d, forecaster5d, honest=False)
    print("  Metrik:")
    for k, v in results["S5_DISHONEST"].items(): print(f"    {k}: {v}")

    # ── S6_HONEST ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S6_HONEST — RL H-PPO + MLP Forecaster, honest=True, {UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr6h = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                   reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                   minibatch=64, epochs=4, honest_estwait=True)
    policy6h, forecaster6h, nrows = train_mode_b(
        TRAIN_DS, tr6h, total_updates=UPDATES,
        baseline_agent_factory=GreedyAgent, collect_steps=672
    )
    print(f"  MLP Forecaster dilatih pada {nrows[0]} pasangan.")
    results["S6_HONEST"] = eval_on_highload(policy6h, forecaster6h, honest=True)
    print("  Metrik:")
    for k, v in results["S6_HONEST"].items(): print(f"    {k}: {v}")

    # ── S6_DISHONEST ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S6_DISHONEST — RL H-PPO + MLP Forecaster, honest=False, {UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr6d = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                   reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                   minibatch=64, epochs=4, honest_estwait=False)
    policy6d, forecaster6d, nrows = train_mode_b(
        TRAIN_DS, tr6d, total_updates=UPDATES,
        baseline_agent_factory=GreedyAgent, collect_steps=672
    )
    results["S6_DISHONEST"] = eval_on_highload(policy6d, forecaster6d, honest=False)
    print("  Metrik:")
    for k, v in results["S6_DISHONEST"].items(): print(f"    {k}: {v}")

    # ── Tabel perbandingan ─────────────────────────────────────────────────────
    metrics   = ["mean_wait", "p90_wait", "gini_served", "herding_events",
                 "mean_trust", "acceptance_rate", "total_served", "n_active"]
    scenarios = ["S0", "S5_HONEST", "S5_DISHONEST", "S6_HONEST", "S6_DISHONEST"]

    print("\n\n" + "="*95)
    print("TABEL PERBANDINGAN HIGH-LOAD (4x Beban, 2x Horizon = 2 Hari, Disjoint Users)")
    print("="*95)
    hdr = f"{'Metrik':<20}" + "".join(f"| {s:<16}" for s in scenarios)
    print(hdr)
    print("-" * len(hdr))
    for m in metrics:
        row = f"{m:<20}"
        for s in scenarios:
            row += f"| {results[s][m]:<16.4f}"
        print(row)
    print("="*95)

    # Simpan
    with open("highload_eval_hasil.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Disimpan ke: highload_eval_hasil.json")


if __name__ == "__main__":
    main()
