"""Orkestrasi PENUH Tahap 2 & 3 di rezim 4x (perbaikan bug SUBSTRAT -- semua run
sebelumnya ternyata di 1x). Menjalankan SEMUA tahap sekuensial dlm SATU proses supaya
dependensi (Lengan A' butuh trust_final Lengan B) otomatis tertangani tanpa intervensi
manual. Output disimpan dgn sufiks "_4x" -- TIDAK menimpa hasil 1x (dipertahankan sbg
pembanding rezim tambahan, bukan dibuang)."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.experiments.ablations import constant_trust, initial_trust
from marl_spklu.rl.rollout import evaluate_policy
from marl_spklu.rl.policy import HPPOPolicy, STATION_FEAT_DIM, CRITIC_STATION_FEAT_DIM
from marl_spklu.agents.greedy_agent import GreedyAgent
from scipy.stats import wilcoxon

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

# ---------------------------------------------------------------------------
# 0. Dataset 4x (30-hari & 90-hari), seed=42 sama dgn kanonik (hanya beban beda)
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] Generate dataset 4x...", flush=True)
DS_4X_30D = common.generate_load_dataset(4.0, seed=42, n_users=2636, horizon_days=30,
                                         out_path=os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json"))
DS_4X_90D = common.generate_load_dataset(4.0, seed=42, n_users=2636, horizon_days=90,
                                         out_path=os.path.join(common.ROOT, "scenario_dataset_klaster12_4x_90d.json"))
print(f"[{elapsed()}] Dataset 4x siap: {DS_4X_30D}, {DS_4X_90D}", flush=True)

BASE_KW = dict(k=2, rollout_steps=288, lr=1e-4, vf_coef=0.25, ent_coef=0.002, max_step_gap=4)
ANGGARAN = 300
N_EVAL_SEED = 10

def train_one(dataset_path, seed, trust_ctx, tag):
    ctx = trust_ctx if trust_ctx is not None else _nullctx()
    with ctx:
        tr = TorchContinuingTrainer(
            dataset_path, reward_calc=RewardCalculator.seimbang(), seed=seed, verbose=False,
            log_path=os.path.join(common.OUTDIR, f"{tag}_seed{seed}.jsonl"), **BASE_KW)
        policy = tr.train(FormulaForecaster(), n_updates=ANGGARAN)
    ckpt = os.path.join(common.OUTDIR, f"{tag}_seed{seed}.pt")
    torch.save(policy.state_dict(), ckpt)
    return dict(seed=seed, ckpt=ckpt, history=tr.history, obs_dim=tr.obs_dim,
               critic_obs_dim=tr.critic_obs_dim, N=tr.N)

import contextlib
@contextlib.contextmanager
def _nullctx():
    yield

def load_policy(ckpt_path, N):
    obs_dim = 6 + STATION_FEAT_DIM * N
    critic_obs_dim = CRITIC_STATION_FEAT_DIM * N + 10
    pol = HPPOPolicy(obs_dim, critic_obs_dim, N)
    pol.load_state_dict(torch.load(ckpt_path))
    pol.eval()
    return pol

def eval_policy_gini(policy, dataset_path, n_eval_seed, trust_value=None):
    ginis, trust_finals = [], []
    for s in range(n_eval_seed):
        ctx = constant_trust(value=trust_value) if trust_value is not None else _nullctx()
        with ctx:
            sim = common.fresh_sim(dataset_path)
            random.seed(s); np.random.seed(s)
            res = evaluate_policy(sim, policy, FormulaForecaster(), k=2)
        ginis.append(res["gini_served"])
        trust_finals.append(float(np.mean([u.trust for u in sim.users])))
    return ginis, trust_finals

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

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0

# ---------------------------------------------------------------------------
# TAHAP 2 (4x): sapuan trust statis 0.4/0.65/0.9, 3 seed
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === TAHAP 2 (4x): training ===", flush=True)
TRUST_SWEEP = [0.4, 0.65, 0.9]
results_t2 = []
for tv in TRUST_SWEEP:
    for seed in range(3):
        print(f"[{elapsed()}]   trust={tv} seed={seed}", flush=True)
        row = train_one(DS_4X_30D, seed, constant_trust(value=tv), f"04x_t2_t{tv}")
        row["trust_value"] = tv
        results_t2.append(row)
        common.save_json(results_t2, "04x_tahap2_training_results.json")
print(f"[{elapsed()}] TAHAP 2 (4x) training selesai", flush=True)

print(f"[{elapsed()}] === TAHAP 2 (4x): evaluasi ===", flush=True)
eval_t2 = {}
for tv in TRUST_SWEEP:
    seed_runs = [r for r in results_t2 if r["trust_value"] == tv]
    N = seed_runs[0]["N"]
    gq = eval_baseline_gini(DS_4X_30D, tv, lambda: GreedyAgent(mode="queue"), N_EVAL_SEED)
    gu = eval_baseline_gini(DS_4X_30D, tv, lambda: GreedyAgent(mode="utilization"), N_EVAL_SEED)
    per_seed = {}
    for run in seed_runs:
        pol = load_policy(run["ckpt"], N)
        ginis, _ = eval_policy_gini(pol, DS_4X_30D, N_EVAL_SEED, trust_value=tv)
        per_seed[run["seed"]] = ginis
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    median_seed = order[len(order)//2]
    ginis_med = per_seed[median_seed]
    w_gq = wilcoxon(ginis_med, gq); w_gu = wilcoxon(ginis_med, gu)
    spread = float(max(np.mean(v) for v in per_seed.values()) - min(np.mean(v) for v in per_seed.values()))
    eval_t2[str(tv)] = dict(trust_value=tv, gq_mean=float(np.mean(gq)), gu_mean=float(np.mean(gu)),
                            policy_mean=float(np.mean(ginis_med)), spread=spread,
                            p_vs_gq=float(w_gq.pvalue), p_vs_gu=float(w_gu.pvalue),
                            per_seed_means={str(k): float(np.mean(v)) for k,v in per_seed.items()})
    print(f"[{elapsed()}]   trust={tv}: policy={np.mean(ginis_med):.4f} gq={np.mean(gq):.4f} gu={np.mean(gu):.4f} "
         f"p_gq={w_gq.pvalue:.4f} p_gu={w_gu.pvalue:.4f} spread={spread:.4f}", flush=True)
common.save_json(eval_t2, "04x_tahap2_eval_results.json")
print(f"[{elapsed()}] TAHAP 2 (4x) evaluasi selesai", flush=True)

# ---------------------------------------------------------------------------
# TAHAP 3 (4x): Lengan B (dinamis) -> ukur trust_final -> Lengan A'
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === TAHAP 3 (4x): Lengan B (dinamis) ===", flush=True)
results_B = []
for seed in range(3):
    print(f"[{elapsed()}]   Lengan B seed={seed}", flush=True)
    row = train_one(DS_4X_30D, seed, None, "04x_t3_B")
    results_B.append(row)
    common.save_json(results_B, "04x_tahap3_training_results_B.json")

print(f"[{elapsed()}] Ukur trust_final Lengan B...", flush=True)
N = results_B[0]["N"]
trust_finals_B = []
for run in results_B:
    pol = load_policy(run["ckpt"], N)
    _, tf = eval_policy_gini(pol, DS_4X_30D, 3, trust_value=None)
    trust_finals_B.extend(tf)
TRUST_APRIME_4X = float(np.mean(trust_finals_B))
print(f"[{elapsed()}] trust_final Lengan B (4x) = {TRUST_APRIME_4X:.4f} (std={np.std(trust_finals_B):.4f})", flush=True)
common.save_json(dict(trust_final_mean=TRUST_APRIME_4X, trust_final_std=float(np.std(trust_finals_B)),
                      raw=trust_finals_B), "04x_tahap3_trust_final_B.json")

print(f"[{elapsed()}] === TAHAP 3 (4x): Lengan A' (constant_trust={TRUST_APRIME_4X:.4f}) ===", flush=True)
results_Aprime = []
for seed in range(3):
    print(f"[{elapsed()}]   Lengan A' seed={seed}", flush=True)
    row = train_one(DS_4X_30D, seed, constant_trust(value=TRUST_APRIME_4X), "04x_t3_Aprime")
    results_Aprime.append(row)
    common.save_json(results_Aprime, "04x_tahap3_training_results_Aprime.json")

print(f"[{elapsed()}] === TAHAP 3 (4x): evaluasi B vs A(0.65 dari Tahap2-4x) vs A' ===", flush=True)
results_A = [r for r in results_t2 if r["trust_value"] == 0.65]

def pick_median(results, dataset_path, trust_value):
    per_seed = {}
    for run in results:
        pol = load_policy(run["ckpt"], N)
        ginis, _ = eval_policy_gini(pol, dataset_path, N_EVAL_SEED, trust_value=trust_value)
        per_seed[run["seed"]] = ginis
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    return per_seed[order[len(order)//2]]

ginis_A = pick_median(results_A, DS_4X_30D, 0.65)
ginis_B = pick_median(results_B, DS_4X_30D, None)
ginis_Aprime = pick_median(results_Aprime, DS_4X_30D, TRUST_APRIME_4X)

w_BA = wilcoxon(ginis_B, ginis_A); w_BAprime = wilcoxon(ginis_B, ginis_Aprime)
d_BA = cohens_d(ginis_B, ginis_A); d_BAprime = cohens_d(ginis_B, ginis_Aprime)
print(f"[{elapsed()}] B vs A : gini_B={np.mean(ginis_B):.4f} gini_A={np.mean(ginis_A):.4f} "
     f"p={w_BA.pvalue:.4f} d={d_BA:+.3f}", flush=True)
print(f"[{elapsed()}] B vs A': gini_B={np.mean(ginis_B):.4f} gini_A'={np.mean(ginis_Aprime):.4f} "
     f"p={w_BAprime.pvalue:.4f} d={d_BAprime:+.3f}", flush=True)
common.save_json(dict(ginis_A=ginis_A, ginis_B=ginis_B, ginis_Aprime=ginis_Aprime,
                      trust_aprime=TRUST_APRIME_4X,
                      gini_mean_A=float(np.mean(ginis_A)), gini_mean_B=float(np.mean(ginis_B)),
                      gini_mean_Aprime=float(np.mean(ginis_Aprime)),
                      wilcoxon_B_vs_A_p=float(w_BA.pvalue), wilcoxon_B_vs_Aprime_p=float(w_BAprime.pvalue),
                      cohens_d_B_vs_A=d_BA, cohens_d_B_vs_Aprime=d_BAprime),
                 "04x_tahap3_eval_pivot_results.json")

# ---------------------------------------------------------------------------
# PERLUASAN (4x): sapuan trust-awal x horizon 90-hari
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === PERLUASAN (4x): sapuan trust-awal x 90-hari ===", flush=True)
IT_SWEEP = [0.3, 0.5, 0.7]
results_it = []
for it in IT_SWEEP:
    for seed in range(3):
        print(f"[{elapsed()}]   initial_trust={it} seed={seed}", flush=True)
        row = train_one(DS_4X_90D, seed, initial_trust(value=it), f"04x_it{it}_90d")
        row["initial_trust"] = it
        results_it.append(row)
        common.save_json(results_it, "04x_training_results_initialtrust90d.json")

print(f"[{elapsed()}] === PERLUASAN (4x): evaluasi sapuan trust-awal ===", flush=True)
N_it = results_it[0]["N"]
rows_it = {}
for it in IT_SWEEP:
    seed_runs = [r for r in results_it if r["initial_trust"] == it]
    per_seed_gini, per_seed_trust = {}, {}
    for run in seed_runs:
        pol = load_policy(run["ckpt"], N_it)
        ginis, trust_finals = [], []
        for s in range(N_EVAL_SEED):
            with initial_trust(value=it):
                sim = common.fresh_sim(DS_4X_90D)
                random.seed(s); np.random.seed(s)
                res = evaluate_policy(sim, pol, FormulaForecaster(), k=2)
            ginis.append(res["gini_served"])
            trust_finals.append(float(np.mean([u.trust for u in sim.users])))
        per_seed_gini[run["seed"]] = ginis
        per_seed_trust[run["seed"]] = trust_finals
    order = sorted(per_seed_gini.keys(), key=lambda sd: np.mean(per_seed_gini[sd]))
    median_seed = order[len(order)//2]
    rows_it[str(it)] = dict(initial_trust=it, median_seed=median_seed,
                            gini_mean=float(np.mean(per_seed_gini[median_seed])),
                            trust_final_mean=float(np.mean(per_seed_trust[median_seed])),
                            ginis=per_seed_gini[median_seed])
    print(f"[{elapsed()}]   it={it}: gini={rows_it[str(it)]['gini_mean']:.4f} "
         f"trust_final={rows_it[str(it)]['trust_final_mean']:.4f}", flush=True)

g03, g05, g07 = rows_it["0.3"]["ginis"], rows_it["0.5"]["ginis"], rows_it["0.7"]["ginis"]
w_0305 = wilcoxon(g03, g05); w_0307 = wilcoxon(g03, g07); w_0507 = wilcoxon(g05, g07)
d_0305 = cohens_d(g03, g05); d_0307 = cohens_d(g03, g07); d_0507 = cohens_d(g05, g07)
print(f"[{elapsed()}] LOCK-IN CHECK (4x): it=0.3 vs 0.5 p={w_0305.pvalue:.4f} d={d_0305:+.3f} | "
     f"it=0.3 vs 0.7 p={w_0307.pvalue:.4f} d={d_0307:+.3f} | "
     f"it=0.5 vs 0.7 (kontrol) p={w_0507.pvalue:.4f} d={d_0507:+.3f}", flush=True)
common.save_json(dict(rows_it=rows_it, wilcoxon_it03_vs_it05_p=float(w_0305.pvalue),
                      cohens_d_it03_vs_it05=d_0305, wilcoxon_it03_vs_it07_p=float(w_0307.pvalue),
                      cohens_d_it03_vs_it07=d_0307, wilcoxon_it05_vs_it07_p=float(w_0507.pvalue),
                      cohens_d_it05_vs_it07=d_0507),
                 "04x_eval_lockin_results.json")

print(f"[{elapsed()}] === SEMUA SELESAI (rezim 4x) ===", flush=True)
