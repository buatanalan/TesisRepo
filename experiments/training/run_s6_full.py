import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""S6 — sistem penuh: EV Agent RL (H-PPO) + forecaster TERPELAJAR (Mode B).

Pretrain forecaster (LearnedForecaster) offline pada trajektori baseline greedy,
bekukan, lalu latih policy H-PPO. Evaluasi pada episode penuh.
"""
import argparse
import json

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.training import train_mode_b, _fresh_sim
from marl_spklu.rl.rollout import evaluate_policy
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.env.history_buffer import HistoryBuffer


def run_s6(dataset="scenario_dataset.json", updates=540, k=3, rollout_steps=96,
           collect_steps=2880, seed=0, lr=1e-4, ent_coef=0.01, alpha_wait=1.0, alpha_fair=0.3,
           alpha_accept=0.1, alpha_shape=0.5, xi=0.5, gamma=0.99, minibatch=64, epochs=4,
           eval_steps=None, honest_estwait=True):
    mode = "EstWait JUJUR (default)" if honest_estwait else "EstWait manipulatif (ablasi)"
    print(f"=== S6: sistem penuh — RL H-PPO + forecaster terpelajar (Mode B) | {mode} ===")
    rc = RewardCalculator(alpha_wait=alpha_wait, alpha_fair=alpha_fair,
                          alpha_accept=alpha_accept, alpha_shape=alpha_shape, xi=xi)
    tr = TorchContinuingTrainer(dataset, k=k, rollout_steps=rollout_steps, seed=seed, verbose=True,
                                reward_calc=rc, lr=lr, ent_coef=ent_coef, gamma=gamma,
                                minibatch=minibatch, epochs=epochs, honest_estwait=honest_estwait)
    print(f"Pretrain forecaster (baseline greedy) lalu latih {updates} chunk-update...")
    policy, forecaster, nrows = train_mode_b(
        dataset, tr, total_updates=updates,
        baseline_agent_factory=GreedyAgent, collect_steps=collect_steps)
    print(f"Forecaster terlatih pada {nrows[0]} baris.")

    sim = _fresh_sim(dataset)
    sim.log_actor_states = True
    sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=sim.max_steps)
    metrics = evaluate_policy(sim, policy, forecaster, k=k, max_steps=eval_steps,
                              honest_estwait=honest_estwait)
    sim.export_actor_logs("s6_actor_trace")
    print("\n=== Hasil Evaluasi S6 ===")
    for kk, vv in metrics.items():
        print(f"  {kk}: {vv}")
    with open("simulation_s6_metrics.json", "w") as f:
        json.dump({"train_history": tr.history, "eval": metrics,
                   "forecaster_rows": nrows[0]}, f, indent=2)
    with open("simulation_s6_eval_logs.json", "w") as f:
        json.dump(sim.logs, f, indent=2)
    print("\n[INFO] Disimpan ke simulation_s6_metrics.json dan simulation_s6_eval_logs.json")
    return policy, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scenario_dataset.json")
    ap.add_argument("--updates", type=int, default=540, help="jumlah chunk-update")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rollout-steps", type=int, default=96, help="ukuran chunk (step); 96=1 hari")
    ap.add_argument("--collect-steps", type=int, default=2880)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--alpha-wait", type=float, default=1.0, help="bobot objektif WAIT (primer)")
    ap.add_argument("--alpha-fair", type=float, default=0.3, help="bobot disparitas dalam-klaster")
    ap.add_argument("--alpha-accept", type=float, default=0.1)
    ap.add_argument("--alpha-shape", type=float, default=0.5, help="bobot shaping Phi")
    ap.add_argument("--xi", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=None)
    ap.add_argument("--manipulative", action="store_true",
                    help="Ablasi: EstWait manipulatif (a2) alih-alih jujur (default jujur)")
    a = ap.parse_args()
    run_s6(a.dataset, a.updates, a.k, a.rollout_steps, a.collect_steps, a.seed, a.lr,
           a.ent_coef, a.alpha_wait, a.alpha_fair, a.alpha_accept, a.alpha_shape, a.xi,
           a.gamma, a.minibatch, a.epochs, a.eval_steps, honest_estwait=not a.manipulative)
