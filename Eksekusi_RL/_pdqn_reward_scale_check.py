"""Cek skala reward PDQN (analog reward_five_way H-PPO, tp jalur pdqn_agent.py) --
DQN TIDAK menormalisasi advantage/return spt PPO, jadi skala MENTAH reward langsung
memengaruhi besaran target Bellman (y = r + gamma*max Q'). Kalau skala terlalu besar,
Q meledak (overestimasi mengompoundING via max operator); terlalu kecil, sinyal
tenggelam dlm noise eksplorasi/loss numerik.

Dijalankan dgn kebijakan HAMPIR ACAK (epsilon=1.0, tanpa training) supaya statistik
mencerminkan STRUKTUR reward apa adanya, bukan hasil optimasi."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.pdqn_agent import PDQNRolloutAgent
from marl_spklu.rl.pdqn_policy import PDQNQNetwork
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.experiments.ablations import constant_trust

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")

torch.manual_seed(0); np.random.seed(0)
with constant_trust(value=0.5):
    sim = common.fresh_sim(DS)
    N = len(sim.spklus)
    q_net = PDQNQNetwork(6 + 7 * N, N, n_types=1, use_preference=True)
    rc = RewardCalculator(alpha_wait=1.0, beta_prox=0.1, alpha_gini=1.0, alpha_flock=1.0,
                          use_delta_gini=True)
    agent = PDQNRolloutAgent(q_net, sim, rc, FormulaForecaster(), epsilon=1.0)

    # Instrumentasi: sadap tiap suku reward SAAT dipanggil (sama pola reward_five_way)
    prox_terms, gini_terms, flock_terms, wait_terms = [], [], [], []
    orig_flock_rolling = rc.flock_reward_rolling
    def patched_flock(cnt, scale=10.0):
        v = orig_flock_rolling(cnt, scale)
        flock_terms.append(v)
        return v
    rc.flock_reward_rolling = patched_flock

    orig_gini = rc.gini_reward
    def patched_gini(u):
        v = orig_gini(u)
        gini_terms.append(v)
        return v
    rc.gini_reward = patched_gini

    orig_wait = rc.wait_reward
    def patched_wait(wd, wa, de=0.0):
        v = orig_wait(wd, wa, de)
        wait_terms.append(v)
        return v
    rc.wait_reward = patched_wait

    for step in range(sim.max_steps):
        sim.step_once(step, agent=agent)

# Total reward PER TRANSISI (akumulasi semua suku) -- ini yg langsung masuk Bellman target.
total_rewards = np.array([t.reward for t in agent.transitions if t.resolved])
gini_arr = np.array(gini_terms); flock_arr = np.array(flock_terms); wait_arr = np.array(wait_terms)

print(f"n_transitions_resolved = {len(total_rewards)}")
print()
print("=== Statistik suku reward MENTAH (alpha=1, epsilon=1.0/hampir acak) ===")
for name, arr in [("gini", gini_arr), ("flock", flock_arr), ("wait", wait_arr)]:
    if len(arr):
        print(f"  {name:8s}: n={len(arr):5d} mean={arr.mean():+.4f} std={arr.std():.4f} "
             f"min={arr.min():+.4f} max={arr.max():+.4f}")
    else:
        print(f"  {name:8s}: KOSONG (tak pernah terpanggil)")

print()
print("=== TOTAL reward per transisi (yg masuk y = r + gamma*maxQ') ===")
if len(total_rewards):
    print(f"  n={len(total_rewards)} mean={total_rewards.mean():+.4f} std={total_rewards.std():.4f} "
         f"min={total_rewards.min():+.4f} max={total_rewards.max():+.4f}")
    print(f"  |r| P50={np.median(np.abs(total_rewards)):.4f} P90={np.quantile(np.abs(total_rewards), 0.9):.4f} "
         f"P99={np.quantile(np.abs(total_rewards), 0.99):.4f}")

# Estimasi kasar magnitude Q* teoretis di bawah reward RATA-RATA konstan (deret geometri):
# Q* ~ r_mean / (1 - gamma) -- referensi kasar "seharusnya berapa besar Q wajar".
gamma = 0.95
if len(total_rewards):
    q_scale_estimate = abs(total_rewards.mean()) / (1 - gamma) if total_rewards.mean() != 0 else total_rewards.std() / (1-gamma)
    print()
    print(f"Estimasi kasar skala Q* wajar (|r_mean|/(1-gamma), gamma={gamma}): ~{q_scale_estimate:.2f}")
    print("(Q_mean yg TERAMATI saat training divergen sebelumnya mencapai 9-10 -- "
         "bandingkan dgn estimasi ini utk menilai wajar/tidaknya.)")

common.save_json(dict(
    gini_mean=float(gini_arr.mean()) if len(gini_arr) else None,
    gini_std=float(gini_arr.std()) if len(gini_arr) else None,
    flock_mean=float(flock_arr.mean()) if len(flock_arr) else None,
    flock_std=float(flock_arr.std()) if len(flock_arr) else None,
    wait_mean=float(wait_arr.mean()) if len(wait_arr) else None,
    wait_std=float(wait_arr.std()) if len(wait_arr) else None,
    total_reward_mean=float(total_rewards.mean()) if len(total_rewards) else None,
    total_reward_std=float(total_rewards.std()) if len(total_rewards) else None,
), "pdqn_reward_scale_check.json")
print("DONE")
