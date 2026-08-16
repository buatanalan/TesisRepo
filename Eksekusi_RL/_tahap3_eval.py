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
from marl_spklu.experiments.ablations import constant_trust

N_EVAL_SEED = 10
DATASET = common.DATASET_KANONIK
K = 2

def load_policy(ckpt_path, N):
    obs_dim = 6 + STATION_FEAT_DIM * N
    critic_obs_dim = CRITIC_STATION_FEAT_DIM * N + 10
    pol = HPPOPolicy(obs_dim, critic_obs_dim, N)
    pol.load_state_dict(torch.load(ckpt_path))
    pol.eval()
    return pol

def eval_policy_gini(policy, n_eval_seed, trust_value=None):
    ginis = []
    for s in range(n_eval_seed):
        if trust_value is not None:
            with constant_trust(value=trust_value):
                sim = common.fresh_sim(DATASET)
                random.seed(s); np.random.seed(s)
                res = evaluate_policy(sim, policy, FormulaForecaster(), k=K)
        else:
            sim = common.fresh_sim(DATASET)
            random.seed(s); np.random.seed(s)
            res = evaluate_policy(sim, policy, FormulaForecaster(), k=K)
        ginis.append(res["gini_served"])
    return ginis

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0

def pick_median_seed_ginis(results, n_eval_seed, trust_value=None, N=None):
    per_seed = {}
    for run in results:
        pol = load_policy(run["ckpt"], N)
        per_seed[run["seed"]] = eval_policy_gini(pol, n_eval_seed, trust_value)
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    median_seed = order[len(order) // 2]
    return per_seed[median_seed], median_seed, per_seed

print("=== Memuat hasil training ketiga lengan ===")
results_A = json.load(open(os.path.join(common.OUTDIR, "02_training_results.json")))
results_A = [r for r in results_A if r["trust_value"] == 0.65]
results_B = json.load(open(os.path.join(common.OUTDIR, "03_training_results_B.json")))
results_Aprime = json.load(open(os.path.join(common.OUTDIR, "03_training_results_Aprime.json")))
N = results_B[0]["N"]

print("=== Evaluasi Lengan A (constant_trust=0.65) ===")
ginis_A, seed_A, all_A = pick_median_seed_ginis(results_A, N_EVAL_SEED, trust_value=0.65, N=N)
print(f"  median_seed={seed_A}, gini_mean={np.mean(ginis_A):.4f} std={np.std(ginis_A):.4f}")

print("=== Evaluasi Lengan B (trust DINAMIS, tanpa ablation) ===")
ginis_B, seed_B, all_B = pick_median_seed_ginis(results_B, N_EVAL_SEED, trust_value=None, N=N)
print(f"  median_seed={seed_B}, gini_mean={np.mean(ginis_B):.4f} std={np.std(ginis_B):.4f}")

print("=== Evaluasi Lengan A' (constant_trust=0.5144, level akhir B) ===")
ginis_Aprime, seed_Aprime, all_Aprime = pick_median_seed_ginis(results_Aprime, N_EVAL_SEED, trust_value=0.5144, N=N)
print(f"  median_seed={seed_Aprime}, gini_mean={np.mean(ginis_Aprime):.4f} std={np.std(ginis_Aprime):.4f}")

print()
print("=== Uji statistik ===")
w_BA = wilcoxon(ginis_B, ginis_A)
w_BAprime = wilcoxon(ginis_B, ginis_Aprime)
d_BA = cohens_d(ginis_B, ginis_A)
d_BAprime = cohens_d(ginis_B, ginis_Aprime)

print(f"B vs A : gini_B={np.mean(ginis_B):.4f} gini_A={np.mean(ginis_A):.4f} "
     f"delta={np.mean(ginis_B)-np.mean(ginis_A):+.4f} p={w_BA.pvalue:.4f} d={d_BA:+.3f} "
     f"({'B SIGNIFIKAN LEBIH BURUK' if w_BA.pvalue<0.05 and np.mean(ginis_B)>np.mean(ginis_A) else 'TIDAK terbukti B lebih buruk'})")
print(f"B vs A': gini_B={np.mean(ginis_B):.4f} gini_A'={np.mean(ginis_Aprime):.4f} "
     f"delta={np.mean(ginis_B)-np.mean(ginis_Aprime):+.4f} p={w_BAprime.pvalue:.4f} d={d_BAprime:+.3f} "
     f"({'B beda SIGNIFIKAN dari A prime (bukti dinamika, bukan level)' if w_BAprime.pvalue<0.05 else 'B TIDAK beda signifikan dari A prime (bisa jadi cuma level trust, bukan dinamika)'})")

summary = dict(
    ginis_A=ginis_A, ginis_B=ginis_B, ginis_Aprime=ginis_Aprime,
    seed_A=seed_A, seed_B=seed_B, seed_Aprime=seed_Aprime,
    gini_mean_A=float(np.mean(ginis_A)), gini_mean_B=float(np.mean(ginis_B)),
    gini_mean_Aprime=float(np.mean(ginis_Aprime)),
    wilcoxon_B_vs_A_p=float(w_BA.pvalue), wilcoxon_B_vs_Aprime_p=float(w_BAprime.pvalue),
    cohens_d_B_vs_A=d_BA, cohens_d_B_vs_Aprime=d_BAprime,
    B_signifikan_lebih_buruk_dari_A=bool(w_BA.pvalue < 0.05 and np.mean(ginis_B) > np.mean(ginis_A)),
    B_beda_signifikan_dari_Aprime=bool(w_BAprime.pvalue < 0.05),
)
common.save_json(summary, "03_eval_pivot_results.json")
print()
print("DONE")
