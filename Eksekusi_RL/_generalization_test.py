import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import torch
from scipy.stats import wilcoxon
import common
from marl_spklu.rl.policy import HPPOPolicy, STATION_FEAT_DIM, CRITIC_STATION_FEAT_DIM
from marl_spklu.rl.rollout import evaluate_policy
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.experiments.ablations import constant_trust

# Uji generalisasi: dataset HELD-OUT (seed BEDA dari training, TAK PERNAH dilihat policy),
# load_multiplier & horizon SAMA dgn dataset training (1.0, 30-hari) -- isolasi murni
# soal "populasi/jadwal baru", bukan soal beban/horizon berbeda.
HELDOUT_SEED = 999
K = 2
N_EVAL_SEED = 10
TRUST_VALUE = 0.65

heldout_path = common.generate_load_dataset(1.0, seed=HELDOUT_SEED, n_users=2636, horizon_days=30,
                                            out_path=os.path.join(common.OUTDIR, "_heldout_seed999.json"))
print("Dataset held-out:", heldout_path)
ds = json.load(open(heldout_path))
print("n_users:", len(ds["users"]), "n_events:", len(ds["schedule"]),
     "load_multiplier:", ds["metadata"].get("load_multiplier"))

# Muat policy Tahap 2 (trust=0.65, ketiga train-seed -- evaluasi semuanya, bukan cuma median,
# supaya kesimpulan generalisasi tak bergantung pilihan satu checkpoint).
results_A = [r for r in json.load(open(os.path.join(common.OUTDIR, "02_training_results.json")))
            if r["trust_value"] == TRUST_VALUE]
N = results_A[0]["N"]

def load_policy(ckpt_path, N):
    obs_dim = 6 + STATION_FEAT_DIM * N
    critic_obs_dim = CRITIC_STATION_FEAT_DIM * N + 10
    pol = HPPOPolicy(obs_dim, critic_obs_dim, N)
    pol.load_state_dict(torch.load(ckpt_path))
    pol.eval()
    return pol

def eval_policy_gini(policy, dataset_path, n_eval_seed, trust_value):
    ginis = []
    for s in range(n_eval_seed):
        with constant_trust(value=trust_value):
            sim = common.fresh_sim(dataset_path)
            random.seed(s); np.random.seed(s)
            res = evaluate_policy(sim, policy, FormulaForecaster(), k=K)
        ginis.append(res["gini_served"])
    return ginis

def eval_baseline_gini(dataset_path, trust_value, agent_factory, n_eval_seed):
    ginis = []
    for s in range(n_eval_seed):
        with constant_trust(value=trust_value):
            sim = common.fresh_sim(dataset_path)
            random.seed(s); np.random.seed(s)
            sim.run(max_steps=sim.max_steps, agent=agent_factory())
        served = np.array([sp.total_served for sp in sim.spklus.values()], float)
        ginis.append(common.gini(served))
    return ginis

print()
print("=== Evaluasi DI DATASET TRAINING (in-sample, konfirmasi angka lama) ===")
gq_in = eval_baseline_gini(common.DATASET_KANONIK, TRUST_VALUE, lambda: GreedyAgent(mode="queue"), N_EVAL_SEED)
gu_in = eval_baseline_gini(common.DATASET_KANONIK, TRUST_VALUE, lambda: GreedyAgent(mode="utilization"), N_EVAL_SEED)
print(f"  greedy_queue in-sample: {np.mean(gq_in):.4f}")
print(f"  greedy_util  in-sample: {np.mean(gu_in):.4f}")

print()
print("=== Evaluasi DI DATASET HELD-OUT (out-of-sample, seed=999, TAK PERNAH dilatih) ===")
gq_out = eval_baseline_gini(heldout_path, TRUST_VALUE, lambda: GreedyAgent(mode="queue"), N_EVAL_SEED)
gu_out = eval_baseline_gini(heldout_path, TRUST_VALUE, lambda: GreedyAgent(mode="utilization"), N_EVAL_SEED)
print(f"  greedy_queue held-out: {np.mean(gq_out):.4f}")
print(f"  greedy_util  held-out: {np.mean(gu_out):.4f}")

print()
all_results = {}
for run in results_A:
    pol = load_policy(run["ckpt"], N)
    ginis_in = eval_policy_gini(pol, common.DATASET_KANONIK, N_EVAL_SEED, TRUST_VALUE)
    ginis_out = eval_policy_gini(pol, heldout_path, N_EVAL_SEED, TRUST_VALUE)
    w_in_gq = wilcoxon(ginis_in, gq_in)
    w_out_gq = wilcoxon(ginis_out, gq_out)
    w_in_gu = wilcoxon(ginis_in, gu_in)
    w_out_gu = wilcoxon(ginis_out, gu_out)
    print(f"train_seed={run['seed']}:")
    print(f"  IN-SAMPLE : policy={np.mean(ginis_in):.4f} vs gq p={w_in_gq.pvalue:.4f} "
         f"({'MENANG' if np.mean(ginis_in)<np.mean(gq_in) else 'KALAH'}) | "
         f"vs gu p={w_in_gu.pvalue:.4f} ({'MENANG' if np.mean(ginis_in)<np.mean(gu_in) else 'KALAH'})")
    print(f"  HELD-OUT  : policy={np.mean(ginis_out):.4f} vs gq p={w_out_gq.pvalue:.4f} "
         f"({'MENANG' if np.mean(ginis_out)<np.mean(gq_out) else 'KALAH'}) | "
         f"vs gu p={w_out_gu.pvalue:.4f} ({'MENANG' if np.mean(ginis_out)<np.mean(gu_out) else 'KALAH'})")
    all_results[run["seed"]] = dict(
        ginis_in=ginis_in, ginis_out=ginis_out,
        gini_in_mean=float(np.mean(ginis_in)), gini_out_mean=float(np.mean(ginis_out)),
        p_in_gq=float(w_in_gq.pvalue), p_out_gq=float(w_out_gq.pvalue),
        p_in_gu=float(w_in_gu.pvalue), p_out_gu=float(w_out_gu.pvalue),
        menang_out_gq=bool(np.mean(ginis_out) < np.mean(gq_out) and w_out_gq.pvalue < 0.05),
        menang_out_gu=bool(np.mean(ginis_out) < np.mean(gu_out) and w_out_gu.pvalue < 0.05),
    )

common.save_json(dict(gq_in=gq_in, gu_in=gu_in, gq_out=gq_out, gu_out=gu_out,
                      heldout_seed=HELDOUT_SEED, per_seed=all_results),
                 "generalization_test_results.json")
print()
print("DONE")
