import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""
run_hard_train_experiment.py
Melatih S5 (RL H-PPO Formula) dan S6 (RL H-PPO + MLP Forecaster)
pada scenario_train_hard.json (4x beban, 2x horizon = 2 minggu)
lalu menguji generalisasi pada scenario_test_1d.json (disjoint users).

Penyesuaian parameter vs. pelatihan standar:
  - rollout_steps: 192 (2 hari per chunk, proporsional dengan horizon 2x)
  - total_updates : 210 (tetap 30 kali putaran data, konsisten dengan standar)
    -> 1344 step / 192 chunk = 7 chunk/pass -> 30 pass = 210 update
"""
import json, os, random
import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.training import train_mode_a, train_mode_b, _fresh_sim
from marl_spklu.rl.rollout import evaluate_policy
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.agents.greedy_agent import GreedyAgent

TRAIN_DS  = "scenario_train_hard.json"   # 2 minggu, 4x beban
TEST_DS   = "scenario_test_1d.json"      # 1 hari, disjoint users
SEED      = 0
# 1344 step / 192 chunk = 7 chunk/pass; 30 pass = 210 update (sama dengan standar)
ROLLOUT_STEPS = 192
TOTAL_UPDATES = 210


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
        "mean_wait":       float(np.mean(waits))           if waits else 0.0,
        "p90_wait":        float(np.quantile(waits, 0.9))  if waits else 0.0,
        "mean_trust":      float(np.mean(trusts))          if trusts else 0.5,
        "acceptance_rate": float(np.mean(complied))        if complied else 0.0,
    }


def eval_on_test(policy, forecaster, honest):
    sim = _fresh_sim(TEST_DS)
    sim.log_actor_states = True
    sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=96)
    evaluate_policy(sim, policy, forecaster, k=3, max_steps=96, honest_estwait=honest)
    return get_test_stats(sim)


def main():
    if not os.path.exists(TRAIN_DS):
        print("ERROR: scenario_train_hard.json tidak ditemukan.")
        print("       Jalankan generate_hard_train.py terlebih dahulu.")
        return
    if not os.path.exists(TEST_DS):
        print("ERROR: scenario_test_1d.json tidak ditemukan.")
        return

    results = {}
    rc = RewardCalculator(alpha_wait=1.0, alpha_fair=0.3, alpha_accept=0.1,
                          alpha_shape=0.5, xi=0.5)

    print(f"\nDataset latih  : {TRAIN_DS}")
    print(f"Dataset uji    : {TEST_DS}")
    print(f"rollout_steps  : {ROLLOUT_STEPS} (2 hari per chunk)")
    print(f"total_updates  : {TOTAL_UPDATES} (~30 pass x 7 chunk/pass)")

    # ── S0 Baseline ────────────────────────────────────────────────────────────
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
    for k, v in results["S0"].items(): print(f"  {k}: {v}")

    # ── S5_HONEST (hard train) ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S5_HONEST — H-PPO Formula, hard train, {TOTAL_UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr5h = TorchContinuingTrainer(
        TRAIN_DS, k=3, rollout_steps=ROLLOUT_STEPS, seed=SEED, verbose=True,
        reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
        minibatch=64, epochs=4, honest_estwait=True
    )
    policy5h, forecaster5h = train_mode_a(TRAIN_DS, tr5h, total_updates=TOTAL_UPDATES)
    results["S5_HONEST_HARD"] = eval_on_test(policy5h, forecaster5h, honest=True)
    print("Metrik TEST:")
    for k, v in results["S5_HONEST_HARD"].items(): print(f"  {k}: {v}")

    # ── S5_DISHONEST (hard train) ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S5_DISHONEST — H-PPO Formula, hard train, {TOTAL_UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr5d = TorchContinuingTrainer(
        TRAIN_DS, k=3, rollout_steps=ROLLOUT_STEPS, seed=SEED, verbose=False,
        reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
        minibatch=64, epochs=4, honest_estwait=False
    )
    policy5d, forecaster5d = train_mode_a(TRAIN_DS, tr5d, total_updates=TOTAL_UPDATES)
    results["S5_DISHONEST_HARD"] = eval_on_test(policy5d, forecaster5d, honest=False)
    print("Metrik TEST:")
    for k, v in results["S5_DISHONEST_HARD"].items(): print(f"  {k}: {v}")

    # ── S6_HONEST (hard train) ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S6_HONEST — H-PPO + MLP, hard train, {TOTAL_UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr6h = TorchContinuingTrainer(
        TRAIN_DS, k=3, rollout_steps=ROLLOUT_STEPS, seed=SEED, verbose=False,
        reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
        minibatch=64, epochs=4, honest_estwait=True
    )
    policy6h, forecaster6h, nrows = train_mode_b(
        TRAIN_DS, tr6h, total_updates=TOTAL_UPDATES,
        baseline_agent_factory=GreedyAgent, collect_steps=1344   # 1 pass penuh 2 minggu
    )
    print(f"  MLP Forecaster dilatih pada {nrows[0]} pasangan.")
    results["S6_HONEST_HARD"] = eval_on_test(policy6h, forecaster6h, honest=True)
    print("Metrik TEST:")
    for k, v in results["S6_HONEST_HARD"].items(): print(f"  {k}: {v}")

    # ── S6_DISHONEST (hard train) ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"S6_DISHONEST — H-PPO + MLP, hard train, {TOTAL_UPDATES} update")
    print("="*60)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr6d = TorchContinuingTrainer(
        TRAIN_DS, k=3, rollout_steps=ROLLOUT_STEPS, seed=SEED, verbose=False,
        reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
        minibatch=64, epochs=4, honest_estwait=False
    )
    policy6d, forecaster6d, nrows = train_mode_b(
        TRAIN_DS, tr6d, total_updates=TOTAL_UPDATES,
        baseline_agent_factory=GreedyAgent, collect_steps=1344
    )
    results["S6_DISHONEST_HARD"] = eval_on_test(policy6d, forecaster6d, honest=False)
    print("Metrik TEST:")
    for k, v in results["S6_DISHONEST_HARD"].items(): print(f"  {k}: {v}")

    # ── Tabel perbandingan ─────────────────────────────────────────────────────
    metrics   = ["mean_wait", "p90_wait", "gini_served", "herding_events",
                 "mean_trust", "acceptance_rate", "total_served", "n_active"]
    scenarios = ["S0",
                 "S5_HONEST_HARD", "S5_DISHONEST_HARD",
                 "S6_HONEST_HARD", "S6_DISHONEST_HARD"]

    print("\n\n" + "="*100)
    print("TABEL — Hard Train (2 Minggu, 4x Beban) | Uji: 1 Hari Disjoint Users")
    print("="*100)
    hdr = f"{'Metrik':<20}" + "".join(f"| {s:<20}" for s in scenarios)
    print(hdr)
    print("-" * len(hdr))
    for m in metrics:
        row = f"{m:<20}"
        for s in scenarios:
            row += f"| {results[s][m]:<20.4f}"
        print(row)
    print("="*100)

    with open("hard_train_hasil.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Disimpan ke: hard_train_hasil.json")


if __name__ == "__main__":
    main()
