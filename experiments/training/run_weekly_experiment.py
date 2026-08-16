import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
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

def get_test_stats(sim):
    # Calculates additional metrics from sim.logs
    waits = [log["wait_time"] for log in sim.logs]
    trusts = [log["trust_after"] for log in sim.logs]
    complied = [log["complied"] for log in sim.logs]
    
    mean_wait = float(np.mean(waits)) if waits else 0.0
    p90_wait = float(np.quantile(waits, 0.9)) if waits else 0.0
    mean_trust = float(np.mean(trusts)) if trusts else 0.5
    acceptance_rate = float(np.mean(complied)) if complied else 0.0
    
    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
    def _gini(a):
        a = np.clip(np.asarray(a, float), 0, None)
        if a.sum() == 0:
            return 0.0
        a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
        return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))
        
    return {
        "gini_served": _gini(served),
        "total_served": int(served.sum()),
        "n_active": int((served > 0).sum()),
        "herding_events": sim.herding_events,
        "mean_wait": mean_wait,
        "p90_wait": p90_wait,
        "mean_trust": mean_trust,
        "acceptance_rate": acceptance_rate
    }

def main():
    train_dataset = "scenario_train_1w.json"
    test_dataset = "scenario_test_1d.json"
    
    if not os.path.exists(train_dataset) or not os.path.exists(test_dataset):
        print("Error: Train/Test datasets not found. Run generate_train_test.py first.")
        return
        
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    results = {}
    
    # ----------------------------- 1. RUN S0 BASELINE ON TEST DATASET -----------------------------
    print("\n--- Running S0 Baseline on Test Dataset ---")
    with open(test_dataset, "r") as f:
        tds = json.load(f)
    spklu_ids = [s["id"] for s in tds["spklus"]]
    
    sim_s0 = Simulator({}, [], HistoryBuffer(spklu_ids, window_size_15m=96), log_actor_states=True)
    sim_s0.load_from_dataset(test_dataset)
    sim_s0.run(max_steps=96)
    
    results["S0"] = get_test_stats(sim_s0)
    print("S0 Baseline Results:")
    for k, v in results["S0"].items():
        print(f"  {k}: {v}")
        
    # ----------------------------- 2. TRAIN & RUN S5 (RL Mode A) ON TEST DATASET -----------------------------
    print("\n--- Training S5 RL H-PPO (Formula Forecaster) for 1 Week ---")
    rc = RewardCalculator(alpha_wait=1.0, alpha_fair=0.3, alpha_accept=0.1, alpha_shape=0.5, xi=0.5)
    tr_s5 = TorchContinuingTrainer(train_dataset, k=3, rollout_steps=96, seed=seed, verbose=False,
                                   reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                   minibatch=64, epochs=4, honest_estwait=True)
    
    policy_s5, forecaster_s5 = train_mode_a(train_dataset, tr_s5, total_updates=210)
    
    print("Evaluating S5 on Test Dataset...")
    sim_s5 = _fresh_sim(test_dataset)
    sim_s5.log_actor_states = True
    sim_s5.history = HistoryBuffer(list(sim_s5.spklus.keys()), window_size_15m=96)
    evaluate_policy(sim_s5, policy_s5, forecaster_s5, k=3, max_steps=96, honest_estwait=True)
    
    results["S5"] = get_test_stats(sim_s5)
    print("S5 RL H-PPO Results:")
    for k, v in results["S5"].items():
        print(f"  {k}: {v}")
        
    # ----------------------------- 3. TRAIN & RUN S6 (RL Mode B + MLP) ON TEST DATASET -----------------------------
    print("\n--- Training S6 RL H-PPO + MLP Forecaster for 1 Week ---")
    tr_s6 = TorchContinuingTrainer(train_dataset, k=3, rollout_steps=96, seed=seed, verbose=False,
                                   reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
                                   minibatch=64, epochs=4, honest_estwait=True)
                                   
    policy_s6, forecaster_s6, nrows = train_mode_b(
        train_dataset, tr_s6, total_updates=210,
        baseline_agent_factory=GreedyAgent, collect_steps=672
    )
    print(f"MLP Forecaster pre-trained on {nrows[0]} pairs.")
    
    print("Evaluating S6 on Test Dataset...")
    sim_s6 = _fresh_sim(test_dataset)
    sim_s6.log_actor_states = True
    sim_s6.history = HistoryBuffer(list(sim_s6.spklus.keys()), window_size_15m=96)
    evaluate_policy(sim_s6, policy_s6, forecaster_s6, k=3, max_steps=96, honest_estwait=True)
    
    results["S6"] = get_test_stats(sim_s6)
    print("S6 RL + MLP Forecaster Results:")
    for k, v in results["S6"].items():
        print(f"  {k}: {v}")
        
    # ----------------------------- 4. SAVE COMPARISON RESULTS -----------------------------
    with open("weekly_experiment_hasil.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n=== COMPARISON SUMMARY (TESTED ON DISJOINT USERS, 1 DAY) ===")
    print(f"{'Metric':<20} | {'S0 (Baseline)':<15} | {'S5 (RL Formula)':<15} | {'S6 (RL MLP)':<15}")
    print("-" * 75)
    for m in ["mean_wait", "gini_served", "herding_events", "mean_trust", "acceptance_rate", "total_served"]:
        print(f"{m:<20} | {results['S0'][m]:<15.4f} | {results['S5'][m]:<15.4f} | {results['S6'][m]:<15.4f}")
    print("-" * 75)
    print("Saved to weekly_experiment_hasil.json")

if __name__ == "__main__":
    main()
