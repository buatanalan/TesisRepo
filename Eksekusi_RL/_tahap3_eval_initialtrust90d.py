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
from marl_spklu.experiments.ablations import initial_trust

N_EVAL_SEED = 10
K = 2

CONFIG = json.load(open(os.path.join(common.OUTDIR, "03_config_initialtrust90d_beku.json")))
DATASET_90D = CONFIG["dataset_path"]
results = json.load(open(os.path.join(common.OUTDIR, "03_training_results_initialtrust90d.json")))
N = results[0]["N"]

def load_policy(ckpt_path, N):
    obs_dim = 6 + STATION_FEAT_DIM * N
    critic_obs_dim = CRITIC_STATION_FEAT_DIM * N + 10
    pol = HPPOPolicy(obs_dim, critic_obs_dim, N)
    pol.load_state_dict(torch.load(ckpt_path))
    pol.eval()
    return pol

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0

rows_by_it = {}
for it in CONFIG["initial_trust_values"]:
    seed_runs = [r for r in results if r["initial_trust"] == it]
    print(f"=== initial_trust={it} (horizon 90d) ===", flush=True)
    per_seed_gini = {}
    per_seed_trust_final = {}
    for run in seed_runs:
        pol = load_policy(run["ckpt"], N)
        ginis, trust_finals = [], []
        for s in range(N_EVAL_SEED):
            with initial_trust(value=it):
                sim = common.fresh_sim(DATASET_90D)
                random.seed(s); np.random.seed(s)
                res = evaluate_policy(sim, pol, FormulaForecaster(), k=K)
            ginis.append(res["gini_served"])
            tr = np.array([u.trust for u in sim.users])
            trust_finals.append(float(tr.mean()))
        per_seed_gini[run["seed"]] = ginis
        per_seed_trust_final[run["seed"]] = trust_finals
        print(f"  seed={run['seed']}: gini_mean={np.mean(ginis):.4f} trust_final_mean={np.mean(trust_finals):.4f}")

    order = sorted(per_seed_gini.keys(), key=lambda sd: np.mean(per_seed_gini[sd]))
    median_seed = order[len(order) // 2]
    rows_by_it[str(it)] = dict(
        initial_trust=it, median_seed=median_seed,
        gini_mean=float(np.mean(per_seed_gini[median_seed])),
        gini_std=float(np.std(per_seed_gini[median_seed])),
        trust_final_mean=float(np.mean(per_seed_trust_final[median_seed])),
        ginis=per_seed_gini[median_seed],
        all_seed_gini_means={sd: float(np.mean(v)) for sd, v in per_seed_gini.items()},
        all_seed_trust_final={sd: float(np.mean(v)) for sd, v in per_seed_trust_final.items()},
    )

print()
print("=== RINGKASAN sapuan trust-awal x horizon 90d ===")
for it, row in rows_by_it.items():
    print(f"initial_trust={it}: gini={row['gini_mean']:.4f} trust_final={row['trust_final_mean']:.4f} "
         f"(drift={row['trust_final_mean']-float(it):+.4f})")

# Bandingkan tiap titik trust-awal (90d) vs Lengan A pokok (statis 0.65, 30-hari) --
# TIDAK apple-to-apple sempurna (horizon beda), tapi berguna sbg konteks tambahan.
eval_pivot = json.load(open(os.path.join(common.OUTDIR, "03_eval_pivot_results.json")))
ginis_A = eval_pivot["ginis_A"]
print()
print("=== Konteks: dibanding Lengan A pokok (statis 0,65, horizon 30-hari) ===")
for it, row in rows_by_it.items():
    w = wilcoxon(row["ginis"], ginis_A)
    d = cohens_d(row["ginis"], ginis_A)
    print(f"it={it} (90d) vs A(statis 0,65, 30d): delta_gini={row['gini_mean']-np.mean(ginis_A):+.4f} "
         f"p={w.pvalue:.4f} d={d:+.3f}")

common.save_json(rows_by_it, "03_eval_initialtrust90d_results.json")
print("DONE")
