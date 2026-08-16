import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""
run_honest_comparison.py
Melatih S5 dan S6 dalam dua mode: honest_estwait=True dan False
kemudian membandingkan semua hasil terhadap baseline S0.

Output: honest_comparison_hasil.json
"""

import os
import json
import numpy as np
import random
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.training import train_mode_a, train_mode_b, _fresh_sim
from marl_spklu.rl.rollout import evaluate_policy
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.agents.greedy_agent import GreedyAgent

TRAIN_DS = "scenario_train_1w.json"
TEST_DS  = "scenario_test_1d.json"
SEED     = 0
UPDATES  = 210   # 30 epoch penuh


# ---------------------------------------------------------------------------
# Helper: kumpulkan metrik dari simulator setelah evaluasi
# ---------------------------------------------------------------------------
def get_test_stats(sim):
    waits    = [log["wait_time"]   for log in sim.logs]
    trusts   = [log["trust_after"] for log in sim.logs]
    complied = [log["complied"]    for log in sim.logs]

    mean_wait = float(np.mean(waits))       if waits    else 0.0
    p90_wait  = float(np.quantile(waits, 0.9)) if waits else 0.0
    mean_trust = float(np.mean(trusts))     if trusts   else 0.5
    acc_rate   = float(np.mean(complied))   if complied else 0.0

    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)

    def _gini(a):
        a = np.clip(np.asarray(a, float), 0, None)
        if a.sum() == 0:
            return 0.0
        a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
        return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))

    return {
        "gini_served":    _gini(served),
        "total_served":   int(served.sum()),
        "n_active":       int((served > 0).sum()),
        "herding_events": sim.herding_events,
        "mean_wait":      mean_wait,
        "p90_wait":       p90_wait,
        "mean_trust":     mean_trust,
        "acceptance_rate": acc_rate,
    }


# ---------------------------------------------------------------------------
# Helper: evaluasi policy pada test dataset
# ---------------------------------------------------------------------------
def eval_on_test(policy, forecaster, honest):
    sim = _fresh_sim(TEST_DS)
    sim.log_actor_states = True
    sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=96)
    evaluate_policy(sim, policy, forecaster, k=3, max_steps=96, honest_estwait=honest)
    return get_test_stats(sim)


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(TRAIN_DS) or not os.path.exists(TEST_DS):
        print("ERROR: Dataset tidak ditemukan. Jalankan generate_train_test.py terlebih dahulu.")
        return

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    results = {}
    rc = RewardCalculator(alpha_wait=1.0, alpha_fair=0.3, alpha_accept=0.1,
                          alpha_shape=0.5, xi=0.5)

    # ==========================================================================
    # S0 – Baseline tanpa rekomendasi
    # ==========================================================================
    print("\n" + "="*60)
    print("S0 — Baseline (tanpa rekomendasi RL)")
    print("="*60)
    with open(TEST_DS) as f:
        tds = json.load(f)
    spklu_ids = [s["id"] for s in tds["spklus"]]
    sim_s0 = Simulator({}, [], HistoryBuffer(spklu_ids, window_size_15m=96), log_actor_states=True)
    sim_s0.load_from_dataset(TEST_DS)
    sim_s0.run(max_steps=96)
    results["S0"] = get_test_stats(sim_s0)
    for k, v in results["S0"].items():
        print(f"  {k}: {v}")

    # ==========================================================================
    # S5_HONEST – H-PPO + Formula Forecaster, honest_estwait=True
    # ==========================================================================
    print("\n" + "="*60)
    print(f"S5_HONEST — RL H-PPO (Formula Forecaster), honest=True, {UPDATES} updates")
    print("="*60)
    tr5h = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                  reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                  minibatch=64, epochs=4, honest_estwait=True)
    policy5h, forecaster5h = train_mode_a(TRAIN_DS, tr5h, total_updates=UPDATES)
    results["S5_HONEST"] = eval_on_test(policy5h, forecaster5h, honest=True)
    print("  Selesai. Metrik:")
    for k, v in results["S5_HONEST"].items():
        print(f"    {k}: {v}")

    # ==========================================================================
    # S5_DISHONEST – H-PPO + Formula Forecaster, honest_estwait=False
    # ==========================================================================
    print("\n" + "="*60)
    print(f"S5_DISHONEST — RL H-PPO (Formula Forecaster), honest=False, {UPDATES} updates")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr5d = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                  reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                  minibatch=64, epochs=4, honest_estwait=False)
    policy5d, forecaster5d = train_mode_a(TRAIN_DS, tr5d, total_updates=UPDATES)
    results["S5_DISHONEST"] = eval_on_test(policy5d, forecaster5d, honest=False)
    print("  Selesai. Metrik:")
    for k, v in results["S5_DISHONEST"].items():
        print(f"    {k}: {v}")

    # ==========================================================================
    # S6_HONEST – H-PPO + MLP Forecaster, honest_estwait=True
    # ==========================================================================
    print("\n" + "="*60)
    print(f"S6_HONEST — RL H-PPO + MLP Forecaster, honest=True, {UPDATES} updates")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr6h = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                  reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                  minibatch=64, epochs=4, honest_estwait=True)
    policy6h, forecaster6h, nrows6h = train_mode_b(
        TRAIN_DS, tr6h, total_updates=UPDATES,
        baseline_agent_factory=GreedyAgent, collect_steps=672
    )
    print(f"  MLP Forecaster pre-trained on {nrows6h[0]} pairs.")
    results["S6_HONEST"] = eval_on_test(policy6h, forecaster6h, honest=True)
    print("  Selesai. Metrik:")
    for k, v in results["S6_HONEST"].items():
        print(f"    {k}: {v}")

    # ==========================================================================
    # S6_DISHONEST – H-PPO + MLP Forecaster, honest_estwait=False
    # ==========================================================================
    print("\n" + "="*60)
    print(f"S6_DISHONEST — RL H-PPO + MLP Forecaster, honest=False, {UPDATES} updates")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr6d = TorchContinuingTrainer(TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=False,
                                  reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                  minibatch=64, epochs=4, honest_estwait=False)
    policy6d, forecaster6d, nrows6d = train_mode_b(
        TRAIN_DS, tr6d, total_updates=UPDATES,
        baseline_agent_factory=GreedyAgent, collect_steps=672
    )
    print(f"  MLP Forecaster pre-trained on {nrows6d[0]} pairs.")
    results["S6_DISHONEST"] = eval_on_test(policy6d, forecaster6d, honest=False)
    print("  Selesai. Metrik:")
    for k, v in results["S6_DISHONEST"].items():
        print(f"    {k}: {v}")

    # ==========================================================================
    # Simpan dan cetak tabel perbandingan
    # ==========================================================================
    with open("honest_comparison_hasil.json", "w") as f:
        json.dump(results, f, indent=2)

    metrics = ["mean_wait", "p90_wait", "gini_served", "herding_events",
               "mean_trust", "acceptance_rate", "total_served", "n_active"]
    scenarios = ["S0", "S5_HONEST", "S5_DISHONEST", "S6_HONEST", "S6_DISHONEST"]

    print("\n\n" + "="*90)
    print("TABEL PERBANDINGAN — Honest vs Dishonest (Uji 1 Hari, Disjoint Users)")
    print("="*90)
    hdr = f"{'Metrik':<20}" + "".join(f"| {s:<16}" for s in scenarios)
    print(hdr)
    print("-" * len(hdr))
    for m in metrics:
        row = f"{m:<20}"
        for s in scenarios:
            val = results[s][m]
            row += f"| {val:<16.4f}"
        print(row)
    print("="*90)
    print("Disimpan ke: honest_comparison_hasil.json")


if __name__ == "__main__":
    main()
