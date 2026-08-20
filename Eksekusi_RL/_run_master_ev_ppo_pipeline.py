"""Pelatihan+evaluasi lengan MASTER perspektif-EV, tulang punggung PPOTrainer standar
(marl_spklu/rl/master_ev_ppo_policy.py) -- pengganti `_run_master_ev_pipeline.py`
(trainer custom kritik-per-timestep, terbukti tak stabil 300 chunk penuh).

Dataset SELALU regime 4x. Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_ev_ppo_pipeline.py \
        > Eksekusi_RL/outputs/master_ev_ppo_pipeline.log 2>&1 &
"""
import sys, os, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOTrainer, MasterEVPPOPolicy,
                                                MasterEVPPOPrefPolicy, MasterEVPPOInferenceAgent)
from marl_spklu.agents.greedy_agent import GreedyAgent
from scipy.stats import wilcoxon

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

p = argparse.ArgumentParser()
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-eval-seed", type=int, default=10)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--k", type=int, default=3)
p.add_argument("--n-critics", type=int, default=1)
p.add_argument("--pref", action="store_true",
              help="tambahkan modul preferensi PDQN (MasterEVPPOPrefPolicy) -- pengujian "
                   "ULANG hipotesis 'P gagal krn identitas-ambigu di perspektif stasiun', "
                   "kini di unit-agen-EV + kritik V(s) stabil")
p.add_argument("--pref-feature-mode", action="store_true",
              help="riwayat preferensi sbg vektor fitur (bukan one-hot identitas) -- "
                   "hanya berlaku bila --pref diberikan")
p.add_argument("--dataset", type=str, default="4x",
              help="'4x' (BAKU, 30 hari) | '1x' (DATASET_KANONIK, TAK sepadan) | path eksplisit "
                   "(mis. scenario_dataset_klaster12_4x_90d.json -- WAJIB sertakan --horizon 90d)")
p.add_argument("--horizon", type=str, default="30d",
              help="penanda tag checkpoint/hasil -- WAJIB diubah ('90d') saat --dataset menunjuk "
                   "dataset 90-hari, supaya tak menimpa diam-diam checkpoint 30-hari yg sudah ada "
                   "(tag 'master_ev_ppo*' polos khusus utk horizon baku 30d)")
args = p.parse_args()

assert not (args.pref_feature_mode and not args.pref), "--pref-feature-mode butuh --pref"
POLICY_CLS = MasterEVPPOPrefPolicy if args.pref else MasterEVPPOPolicy
POLICY_KW = dict(pref_feature_mode=args.pref_feature_mode) if args.pref else dict()

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
if args.dataset == "4x":
    DATASET = _DATASET_4X
    assert os.path.exists(DATASET), f"dataset 4x tak ditemukan: {DATASET}"
elif args.dataset == "1x":
    DATASET = common.DATASET_KANONIK
    print(f"[{elapsed()}] !! PERINGATAN: regime 1x -- hasil TAK SEPADAN.", flush=True)
else:
    DATASET = args.dataset
    if not os.path.isabs(DATASET) and not os.path.exists(DATASET):
        DATASET = os.path.join(common.ROOT, DATASET)
    assert os.path.exists(DATASET), f"dataset tak ditemukan: {DATASET}"
    assert args.horizon != "30d", (
        "--dataset kustom diberikan tapi --horizon masih baku ('30d') -- kemungkinan besar "
        "kekeliruan (checkpoint akan bertabrakan diam-diam dgn hasil 30d yg sudah ada). "
        "Sertakan --horizon eksplisit, mis. --horizon 90d.")

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
_suffix = ("_pref_feat" if args.pref_feature_mode else ("_pref" if args.pref else "")) + _horizon_suffix
TAG_ARM = "master_ev_ppo" + _suffix
print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} (perspektif-EV, PPOTrainer standar, V(s) atensi tunggal, "
     f"pref={args.pref} pref_feature_mode={args.pref_feature_mode})", flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"k={args.k} n_critics={args.n_critics}", flush=True)


def train_one(seed, tag):
    from marl_spklu.rl.forecaster import FormulaForecaster
    tr = MasterEVPPOTrainer(DATASET, rollout_steps=args.rollout_steps, seed=seed, verbose=False,
                            k=args.k, n_critics=args.n_critics,
                            policy_cls=POLICY_CLS, policy_kw=POLICY_KW)
    policy = tr.train(FormulaForecaster(), n_updates=args.n_updates)
    ckpt = os.path.join(common.OUTDIR, f"{tag}_actor_seed{seed}.pt")
    torch.save(policy.state_dict(), ckpt)
    return dict(seed=seed, ckpt=ckpt, history=tr.history)


def load_policy(ckpt_path, n_spklu):
    pol = POLICY_CLS(n_spklu, n_critics=args.n_critics, **POLICY_KW)
    pol.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    pol.eval()
    return pol


def eval_policy_gini(ckpt_path, dataset_path, n_eval_seed, k):
    from marl_spklu.rl.forecaster import FormulaForecaster
    sim0 = common.fresh_sim(dataset_path)
    pol = load_policy(ckpt_path, len(sim0.spklus))
    ginis = []
    for s in range(n_eval_seed):
        sim = common.fresh_sim(dataset_path)
        random.seed(s); np.random.seed(s)
        agent = MasterEVPPOInferenceAgent(pol, sim, FormulaForecaster(), k=k)
        sim.run(max_steps=sim.max_steps, agent=agent)
        served = np.array([sp.total_served for sp in sim.spklus.values()], float)
        ginis.append(common.gini(served))
    return ginis


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


def pick_median(results, dataset_path, n_eval_seed, k):
    per_seed = {}
    for run in results:
        per_seed[run["seed"]] = eval_policy_gini(run["ckpt"], dataset_path, n_eval_seed, k)
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    return per_seed[order[len(order) // 2]], per_seed


print(f"[{elapsed()}] === PELATIHAN MasterEV-PPO ({args.n_train_seed} seed) ===", flush=True)
results = []
results_path = f"{TAG_ARM}_training_results.json"
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
    row = train_one(seed, TAG_ARM)
    print(f"[{elapsed()}]   seed={seed} -- SELESAI", flush=True)
    results.append(row)
    common.save_json(results, results_path)
print(f"[{elapsed()}] Pelatihan selesai ({len(results)} seed)", flush=True)

print(f"[{elapsed()}] === Evaluasi vs greedy_queue/greedy_util ({args.n_eval_seed} seed) ===", flush=True)
gq = eval_baseline_gini(DATASET, lambda: GreedyAgent(mode="queue"), args.n_eval_seed)
gu = eval_baseline_gini(DATASET, lambda: GreedyAgent(mode="utilization"), args.n_eval_seed)
ginis_policy, per_seed = pick_median(results, DATASET, args.n_eval_seed, args.k)
spread = float(max(np.mean(v) for v in per_seed.values())
              - min(np.mean(v) for v in per_seed.values()))
w_gq = wilcoxon(ginis_policy, gq)
w_gu = wilcoxon(ginis_policy, gu)
d_gq = cohens_d(ginis_policy, gq)
d_gu = cohens_d(ginis_policy, gu)

print(f"[{elapsed()}] MasterEV-PPO={np.mean(ginis_policy):.4f}  greedy_queue={np.mean(gq):.4f}  "
     f"greedy_util={np.mean(gu):.4f}", flush=True)
print(f"[{elapsed()}] p_vs_gq={w_gq.pvalue:.4f} d={d_gq:+.3f}  "
     f"p_vs_gu={w_gu.pvalue:.4f} d={d_gu:+.3f}  spread_antar_seed={spread:.4f}", flush=True)

common.save_json(dict(
    gini_policy=ginis_policy, gini_greedy_queue=gq, gini_greedy_util=gu,
    mean_policy=float(np.mean(ginis_policy)), mean_gq=float(np.mean(gq)), mean_gu=float(np.mean(gu)),
    p_vs_gq=float(w_gq.pvalue), p_vs_gu=float(w_gu.pvalue),
    cohens_d_vs_gq=d_gq, cohens_d_vs_gu=d_gu, spread_antar_seed=spread,
    per_seed_means={str(k): float(np.mean(v)) for k, v in per_seed.items()},
    config=dict(n_train_seed=args.n_train_seed, n_eval_seed=args.n_eval_seed,
               n_updates=args.n_updates, rollout_steps=args.rollout_steps, k=args.k,
               n_critics=args.n_critics, pref=args.pref,
               pref_feature_mode=args.pref_feature_mode, horizon=args.horizon,
               dataset=DATASET)),
    f"{TAG_ARM}_eval_results.json")

print(f"[{elapsed()}] === SEMUA SELESAI (MasterEV-PPO) ===", flush=True)
