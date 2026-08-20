"""Pelatihan+evaluasi lengan MASTER stasiun-sebagai-agen + PPO/GAE/clip (bukan DDPG) --
ablasi tulang-punggung terhadap `_run_master_ddpg_pipeline.py`. Lihat docstring
`marl_spklu/rl/master_station_ppo_policy.py` untuk apa yang ditahan konstan/diganti.

Pola berkas SAMA PERSIS `_run_master_ddpg_pipeline.py` (checkpoint per seed, resumable,
evaluasi Wilcoxon/Cohen's d vs greedy) supaya kedua lengan langsung sebanding.

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_station_ppo_pipeline.py \
        > Eksekusi_RL/outputs/master_station_ppo_pipeline.log 2>&1 &
"""
import sys, os, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.master_station_ppo_policy import (MasterStationPPOTrainer,
                                                      MasterStationPPOPolicy,
                                                      MasterStationPPOInferenceAgent)
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.agents.greedy_agent import GreedyAgent
from scipy.stats import wilcoxon

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

p = argparse.ArgumentParser()
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-eval-seed", type=int, default=10)
p.add_argument("--n-updates", type=int, default=300, help="jumlah chunk rollout (anggaran)")
p.add_argument("--rollout-steps", type=int, default=96, help="langkah simulasi per chunk")
p.add_argument("--k", type=int, default=3, help="langit-langit jumlah stasiun direkomendasikan")
p.add_argument("--dataset", type=str, default="4x",
              help="'4x' (BAKU, regime dibekukan Tahap 1, sepadan dgn seluruh lengan lain) "
                   "| '1x' (DATASET_KANONIK, TAK sepadan) | path berkas .json eksplisit")
args = p.parse_args()

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
if args.dataset == "4x":
    DATASET = _DATASET_4X
    assert os.path.exists(DATASET), f"dataset 4x tak ditemukan: {DATASET}"
elif args.dataset == "1x":
    DATASET = common.DATASET_KANONIK
    print(f"[{elapsed()}] !! PERINGATAN: regime 1x dipilih eksplisit -- hasil TAK SEPADAN "
         f"dgn lengan lain (regime 4x). Hanya utk debug/kalibrasi.", flush=True)
else:
    DATASET = args.dataset
    assert os.path.exists(DATASET), f"dataset tak ditemukan: {DATASET}"

print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"k={args.k}", flush=True)


def train_one(seed, tag):
    tr = MasterStationPPOTrainer(DATASET, rollout_steps=args.rollout_steps, seed=seed,
                                 verbose=False, k=args.k)
    policy = tr.train(FormulaForecaster(), n_updates=args.n_updates)
    ckpt = os.path.join(common.OUTDIR, f"{tag}_seed{seed}.pt")
    torch.save(policy.state_dict(), ckpt)
    return dict(seed=seed, ckpt=ckpt, history=tr.history, N=tr.N,
               n_grad_updates=len(tr.history))


def load_policy(ckpt_path, n_spklu):
    policy = MasterStationPPOPolicy(n_spklu)
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.eval()
    return policy


def eval_policy_gini(ckpt_path, n_spklu, dataset_path, n_eval_seed, k):
    policy = load_policy(ckpt_path, n_spklu)
    ginis, trust_finals = [], []
    for s in range(n_eval_seed):
        sim = common.fresh_sim(dataset_path)
        random.seed(s); np.random.seed(s)
        agent = MasterStationPPOInferenceAgent(policy, sim, k=k)
        sim.run(max_steps=sim.max_steps, agent=agent)
        served = np.array([sp.total_served for sp in sim.spklus.values()], float)
        ginis.append(common.gini(served))
        trust_finals.append(float(np.mean([u.trust for u in sim.users])))
    return ginis, trust_finals


def eval_baseline_gini(dataset_path, agent_factory, n_eval_seed):
    ginis = []
    for s in range(n_eval_seed):
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


def pick_median(results, n_spklu, dataset_path, n_eval_seed, k):
    per_seed = {}
    for run in results:
        ginis, _ = eval_policy_gini(run["ckpt"], n_spklu, dataset_path, n_eval_seed, k)
        per_seed[run["seed"]] = ginis
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    return per_seed[order[len(order) // 2]], per_seed


# ---------------------------------------------------------------------------
# PELATIHAN (resumable per seed)
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === PELATIHAN MasterStationPPO ({args.n_train_seed} seed) ===",
     flush=True)
results = []
results_path = "master_station_ppo_training_results.json"
existing = {}
try:
    import json
    with open(os.path.join(common.OUTDIR, results_path), encoding="utf-8") as f:
        for row in json.load(f):
            existing[row["seed"]] = row
    print(f"[{elapsed()}] {len(existing)} seed sudah selesai sebelumnya (resume)", flush=True)
except FileNotFoundError:
    pass

for seed in range(args.n_train_seed):
    if seed in existing:
        print(f"[{elapsed()}]   seed={seed} -- SKIP (checkpoint sudah ada)", flush=True)
        results.append(existing[seed])
        continue
    print(f"[{elapsed()}]   seed={seed} -- mulai training", flush=True)
    row = train_one(seed, "master_station_ppo")
    print(f"[{elapsed()}]   seed={seed} -- SELESAI (chunk={row['n_grad_updates']})", flush=True)
    results.append(row)
    common.save_json(results, results_path)
print(f"[{elapsed()}] Pelatihan selesai ({len(results)} seed)", flush=True)

N = results[0]["N"]

# ---------------------------------------------------------------------------
# EVALUASI vs greedy_queue/greedy_util
# ---------------------------------------------------------------------------
print(f"[{elapsed()}] === Evaluasi vs greedy_queue/greedy_util ({args.n_eval_seed} seed) ===",
     flush=True)
gq = eval_baseline_gini(DATASET, lambda: GreedyAgent(mode="queue"), args.n_eval_seed)
gu = eval_baseline_gini(DATASET, lambda: GreedyAgent(mode="utilization"), args.n_eval_seed)
ginis_policy, per_seed = pick_median(results, N, DATASET, args.n_eval_seed, args.k)
spread = float(max(np.mean(v) for v in per_seed.values())
              - min(np.mean(v) for v in per_seed.values()))
w_gq = wilcoxon(ginis_policy, gq)
w_gu = wilcoxon(ginis_policy, gu)
d_gq = cohens_d(ginis_policy, gq)
d_gu = cohens_d(ginis_policy, gu)

print(f"[{elapsed()}] MasterStationPPO={np.mean(ginis_policy):.4f}  "
     f"greedy_queue={np.mean(gq):.4f}  greedy_util={np.mean(gu):.4f}", flush=True)
print(f"[{elapsed()}] p_vs_gq={w_gq.pvalue:.4f} d={d_gq:+.3f}  "
     f"p_vs_gu={w_gu.pvalue:.4f} d={d_gu:+.3f}  spread_antar_seed={spread:.4f}", flush=True)

common.save_json(dict(
    gini_policy=ginis_policy, gini_greedy_queue=gq, gini_greedy_util=gu,
    mean_policy=float(np.mean(ginis_policy)), mean_gq=float(np.mean(gq)),
    mean_gu=float(np.mean(gu)), p_vs_gq=float(w_gq.pvalue), p_vs_gu=float(w_gu.pvalue),
    cohens_d_vs_gq=d_gq, cohens_d_vs_gu=d_gu, spread_antar_seed=spread,
    per_seed_means={str(k): float(np.mean(v)) for k, v in per_seed.items()},
    config=dict(n_train_seed=args.n_train_seed, n_eval_seed=args.n_eval_seed,
               n_updates=args.n_updates, rollout_steps=args.rollout_steps, k=args.k)),
    "master_station_ppo_eval_results.json")

print(f"[{elapsed()}] === SEMUA SELESAI (MasterStationPPO) ===", flush=True)
