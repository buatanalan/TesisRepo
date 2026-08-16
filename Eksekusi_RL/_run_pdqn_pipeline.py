"""Migrasi metode utama ke PDQN diskrit (Lin dkk. 2024) -- replikasi pola tangga ablasi
A/B/A' yg sama dgn H-PPO v2 (Tahap 2/3), TAPI dgn PDQN diskrit yg SUDAH established
(tak perlu dibangun dari nol -- lihat memori pivot-ke-pdqn). Dataset & rezim SAMA dgn
hasil H-PPO 4x terakhir (scenario_dataset_klaster12_4x.json, seed=42) supaya
apple-to-apple dgn substrat final yg sudah dibekukan.

Titik A pakai SATU nilai trust statis (0,5 -- netral, sama dgn INIT_TRUST default &
salah satu titik yg diuji paper asli), BUKAN sapuan 3 titik spt H-PPO Tahap 2 --
menjaga cakupan kerja tetap ringan (prinsip: bangun dari temuan yg ada)."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.dqn_trainer import DQNContinuingTrainer
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.experiments.ablations import constant_trust
from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent
from marl_spklu.agents.greedy_agent import GreedyAgent
from scipy.stats import wilcoxon
import contextlib

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

@contextlib.contextmanager
def _nullctx():
    yield

DS_4X_30D = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
assert os.path.exists(DS_4X_30D), f"Dataset 4x tidak ditemukan: {DS_4X_30D}"

TRUST_A = 0.5
N_TRAIN_SEED = 3
N_EVAL_SEED = 10
ANGGARAN = 300          # n_updates (potongan), SAMA istilah dgn H-PPO utk komparabilitas
ROLLOUT_STEPS = 288     # SAMA dgn H-PPO 4x
# PERBAIKAN (ditemukan pasca bug anti-herding): default 100 disinkron ~5x/chunk pd
# throughput 4x (~531 langkah gradien/chunk) -- terlalu sering, target kehilangan fungsi
# stabilisasi (mirip tanpa target tetap). Smoke-test 3000 (80 iter): q_mean stabil
# 0,30-0,40 (vs 100: 2,2 turun-naik liar), Gini 10 iter akhir 0,038-0,053 (LEBIH BAIK
# dari greedy 0,086-0,090, vs 100: masih 0,13-0,26).
TARGET_UPDATE_EVERY = 3000

def train_one(seed, trust_ctx, tag):
    ctx = trust_ctx if trust_ctx is not None else _nullctx()
    with ctx:
        tr = DQNContinuingTrainer(
            DS_4X_30D, rollout_steps=ROLLOUT_STEPS, reward_calc=RewardCalculator.seimbang(),
            seed=seed, verbose=False, target_update_every=TARGET_UPDATE_EVERY,
            log_path=os.path.join(common.OUTDIR, f"{tag}_seed{seed}.jsonl"))
        q_net = tr.train(n_updates=ANGGARAN)
    ckpt = os.path.join(common.OUTDIR, f"{tag}_seed{seed}.pt")
    torch.save(q_net.state_dict(), ckpt)
    return dict(seed=seed, ckpt=ckpt, history=tr.history, obs_dim=tr.obs_dim, N=tr.N,
               n_types=tr.n_types, use_preference=tr.use_preference,
               pref_feature_mode=tr.pref_feature_mode)

def load_qnet(ckpt_path, obs_dim, N, n_types, use_preference, pref_feature_mode):
    from marl_spklu.rl.pdqn_policy import PDQNQNetwork
    q_net = PDQNQNetwork(obs_dim, N, n_types=n_types, use_preference=use_preference,
                         pref_feature_mode=pref_feature_mode)
    q_net.load_state_dict(torch.load(ckpt_path))
    q_net.eval()
    return q_net

def eval_qnet_gini(run, dataset_path, n_eval_seed, trust_value=None):
    q_net = load_qnet(run["ckpt"], run["obs_dim"], run["N"], run["n_types"],
                      run["use_preference"], run["pref_feature_mode"])
    ginis, trust_finals = [], []
    for s in range(n_eval_seed):
        ctx = constant_trust(value=trust_value) if trust_value is not None else _nullctx()
        with ctx:
            sim = common.fresh_sim(dataset_path)
            random.seed(s); np.random.seed(s)
            agent = PDQNInferenceAgent(q_net, forecaster=FormulaForecaster(),
                                       pref_feature_mode=run["pref_feature_mode"])
            agent.bind_to_sim(sim)
            sim.run(max_steps=sim.max_steps, agent=agent)
        served = np.array([sp.total_served for sp in sim.spklus.values()], float)
        ginis.append(common.gini(served))
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

def pick_median(results, dataset_path, trust_value):
    per_seed = {}
    for run in results:
        ginis, _ = eval_qnet_gini(run, dataset_path, N_EVAL_SEED, trust_value=trust_value)
        per_seed[run["seed"]] = ginis
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    return per_seed[order[len(order)//2]], per_seed

# ---------------------------------------------------------------------------
# TITIK A (PDQN, trust statis=0.5)
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === TITIK A (PDQN, constant_trust={TRUST_A}) ===", flush=True)
results_A = []
for seed in range(N_TRAIN_SEED):
    print(f"[{elapsed()}]   seed={seed}", flush=True)
    row = train_one(seed, constant_trust(value=TRUST_A), "pdqn_A")
    results_A.append(row)
    common.save_json(results_A, "pdqn_A_training_results.json")
print(f"[{elapsed()}] Titik A training selesai", flush=True)

print(f"[{elapsed()}] === Evaluasi Titik A vs greedy_queue/greedy_util ===", flush=True)
gq = eval_baseline_gini(DS_4X_30D, TRUST_A, lambda: GreedyAgent(mode="queue"), N_EVAL_SEED)
gu = eval_baseline_gini(DS_4X_30D, TRUST_A, lambda: GreedyAgent(mode="utilization"), N_EVAL_SEED)
ginis_A, per_seed_A = pick_median(results_A, DS_4X_30D, TRUST_A)
spread_A = float(max(np.mean(v) for v in per_seed_A.values()) - min(np.mean(v) for v in per_seed_A.values()))
w_gq = wilcoxon(ginis_A, gq); w_gu = wilcoxon(ginis_A, gu)
print(f"[{elapsed()}] Titik A: policy={np.mean(ginis_A):.4f} gq={np.mean(gq):.4f} gu={np.mean(gu):.4f} "
     f"p_gq={w_gq.pvalue:.4f} p_gu={w_gu.pvalue:.4f} spread={spread_A:.4f}", flush=True)
common.save_json(dict(gq=gq, gu=gu, ginis_A=ginis_A, spread_A=spread_A,
                      p_vs_gq=float(w_gq.pvalue), p_vs_gu=float(w_gu.pvalue),
                      per_seed_means_A={str(k): float(np.mean(v)) for k, v in per_seed_A.items()}),
                 "pdqn_A_eval_results.json")

# ---------------------------------------------------------------------------
# LENGAN B (PDQN, trust dinamis)
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === LENGAN B (PDQN, trust DINAMIS) ===", flush=True)
results_B = []
for seed in range(N_TRAIN_SEED):
    print(f"[{elapsed()}]   seed={seed}", flush=True)
    row = train_one(seed, None, "pdqn_B")
    results_B.append(row)
    common.save_json(results_B, "pdqn_B_training_results.json")
print(f"[{elapsed()}] Lengan B training selesai", flush=True)

print(f"[{elapsed()}] Ukur trust_final Lengan B...", flush=True)
trust_finals_B = []
for run in results_B:
    _, tf = eval_qnet_gini(run, DS_4X_30D, 3, trust_value=None)
    trust_finals_B.extend(tf)
TRUST_APRIME = float(np.mean(trust_finals_B))
print(f"[{elapsed()}] trust_final Lengan B = {TRUST_APRIME:.4f} (std={np.std(trust_finals_B):.4f})", flush=True)
common.save_json(dict(trust_final_mean=TRUST_APRIME, trust_final_std=float(np.std(trust_finals_B)),
                      raw=trust_finals_B), "pdqn_trust_final_B.json")

# ---------------------------------------------------------------------------
# LENGAN A' (PDQN, statis di level akhir B)
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === LENGAN A' (PDQN, constant_trust={TRUST_APRIME:.4f}) ===", flush=True)
results_Aprime = []
for seed in range(N_TRAIN_SEED):
    print(f"[{elapsed()}]   seed={seed}", flush=True)
    row = train_one(seed, constant_trust(value=TRUST_APRIME), "pdqn_Aprime")
    results_Aprime.append(row)
    common.save_json(results_Aprime, "pdqn_Aprime_training_results.json")
print(f"[{elapsed()}] Lengan A' training selesai", flush=True)

# ---------------------------------------------------------------------------
# EVALUASI B vs A vs A'
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === Evaluasi B vs A vs A' ===", flush=True)
ginis_B, _ = pick_median(results_B, DS_4X_30D, None)
ginis_Aprime, _ = pick_median(results_Aprime, DS_4X_30D, TRUST_APRIME)

w_BA = wilcoxon(ginis_B, ginis_A); w_BAprime = wilcoxon(ginis_B, ginis_Aprime)
d_BA = cohens_d(ginis_B, ginis_A); d_BAprime = cohens_d(ginis_B, ginis_Aprime)
print(f"[{elapsed()}] B vs A : gini_B={np.mean(ginis_B):.4f} gini_A={np.mean(ginis_A):.4f} "
     f"p={w_BA.pvalue:.4f} d={d_BA:+.3f}", flush=True)
print(f"[{elapsed()}] B vs A': gini_B={np.mean(ginis_B):.4f} gini_A'={np.mean(ginis_Aprime):.4f} "
     f"p={w_BAprime.pvalue:.4f} d={d_BAprime:+.3f}", flush=True)
common.save_json(dict(ginis_A=ginis_A, ginis_B=ginis_B, ginis_Aprime=ginis_Aprime,
                      trust_aprime=TRUST_APRIME,
                      gini_mean_A=float(np.mean(ginis_A)), gini_mean_B=float(np.mean(ginis_B)),
                      gini_mean_Aprime=float(np.mean(ginis_Aprime)),
                      wilcoxon_B_vs_A_p=float(w_BA.pvalue), wilcoxon_B_vs_Aprime_p=float(w_BAprime.pvalue),
                      cohens_d_B_vs_A=d_BA, cohens_d_B_vs_Aprime=d_BAprime),
                 "pdqn_eval_pivot_results.json")

print(f"[{elapsed()}] === SEMUA SELESAI (PDQN) ===", flush=True)
