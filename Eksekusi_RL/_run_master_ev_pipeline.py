"""Pelatihan+evaluasi lengan MASTER perspektif-EV (marl_spklu/rl/master_ev_*.py) --
dirancang dijalankan di server (background, log flush=True, checkpoint per seed --
resumable). Mirror `_run_master_ddpg_pipeline.py` tapi lengan berbeda arsitektur
total (PPO on-policy, kritik dibatch per-timestep, bukan DDPG off-policy).

Dataset SELALU regime 4x (dibekukan Tahap 1) -- BUKAN DATASET_KANONIK (1x, lihat
memori "bug rezim 1x vs 4x").

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_ev_pipeline.py \
        > Eksekusi_RL/outputs/master_ev_pipeline.log 2>&1 &
"""
import sys, os, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.master_ev_trainer import MasterEVTrainer, MasterEVInferenceAgent
from marl_spklu.rl.master_ev_policy import MasterEVActor
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER_EV
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
p.add_argument("--delay-minutes", type=float, default=15.0, help="d Delayed Access Strategy")
p.add_argument("--k", type=int, default=3, help="langit-langit jumlah stasiun direkomendasikan")
p.add_argument("--dataset", type=str, default="4x",
              help="'4x' (BAKU, regime dibekukan Tahap 1) | '1x' (DATASET_KANONIK, TAK sepadan) "
                   "| path berkas .json eksplisit")
p.add_argument("--forecaster", type=str, default="formula", choices=["formula", "vwf"],
              help="'formula' (BAKU historis -- FormulaForecaster kasar, TAK PERNAH diuji "
                   "dgn VWF sebelum ini) | 'vwf' (VirtualWaitForecaster -- basis SAMA dgn "
                   "eta_norm yg dilihat aktor; menguji apakah kegagalan kritik-kolektif "
                   "murni arsitektural atau tercampur celah forecaster, lihat diskusi "
                   "'apakah ada pengaruh dari vwf'). >0 WAJIB tag terpisah, lihat di bawah.")
args = p.parse_args()

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
if args.dataset == "4x":
    DATASET = _DATASET_4X
    assert os.path.exists(DATASET), f"dataset 4x tak ditemukan: {DATASET}"
elif args.dataset == "1x":
    DATASET = common.DATASET_KANONIK
    print(f"[{elapsed()}] !! PERINGATAN: regime 1x dipilih eksplisit -- hasil TAK SEPADAN "
         f"dgn uji_konsolidasi_30d.json (regime 4x).", flush=True)
else:
    DATASET = args.dataset
    assert os.path.exists(DATASET), f"dataset tak ditemukan: {DATASET}"

_fc_suffix = "" if args.forecaster == "formula" else f"_{args.forecaster}"
TAG_ARM = "master_ev" + _fc_suffix
print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} (perspektif-EV, kritik per-timestep, "
     f"forecaster={args.forecaster})", flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"delay_minutes={args.delay_minutes} k={args.k}", flush=True)


def make_forecaster():
    from marl_spklu.rl.forecaster import FormulaForecaster, VirtualWaitForecaster
    return VirtualWaitForecaster() if args.forecaster == "vwf" else FormulaForecaster()


def train_one(seed, tag):
    tr = MasterEVTrainer(DATASET, rollout_steps=args.rollout_steps, seed=seed, verbose=False,
                         delay_minutes=args.delay_minutes, k=args.k)
    actor, critic = tr.train(n_updates=args.n_updates, forecaster=make_forecaster())
    ckpt_actor = os.path.join(common.OUTDIR, f"{tag}_actor_seed{seed}.pt")
    ckpt_critic = os.path.join(common.OUTDIR, f"{tag}_critic_seed{seed}.pt")
    torch.save(actor.state_dict(), ckpt_actor)
    torch.save(critic.state_dict(), ckpt_critic)
    return dict(seed=seed, ckpt_actor=ckpt_actor, ckpt_critic=ckpt_critic,
               history=tr.history, delay_steps=tr.delay_steps, n_critics=tr.n_critics)


def load_actor(ckpt_path):
    actor = MasterEVActor(STATION_FEAT_DIM_MASTER_EV)
    actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    actor.eval()
    return actor


def eval_actor_gini(ckpt_path, dataset_path, n_eval_seed, k):
    actor = load_actor(ckpt_path)
    ginis, trust_finals = [], []
    for s in range(n_eval_seed):
        sim = common.fresh_sim(dataset_path)
        random.seed(s); np.random.seed(s)
        agent = MasterEVInferenceAgent(actor, forecaster=make_forecaster(), k=k)
        agent.bind_to_sim(sim)
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


def pick_median(results, dataset_path, n_eval_seed, k):
    per_seed = {}
    for run in results:
        ginis, _ = eval_actor_gini(run["ckpt_actor"], dataset_path, n_eval_seed, k)
        per_seed[run["seed"]] = ginis
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    return per_seed[order[len(order) // 2]], per_seed


print(f"[{elapsed()}] === PELATIHAN MasterEV ({args.n_train_seed} seed) ===", flush=True)
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

print(f"[{elapsed()}] MasterEV={np.mean(ginis_policy):.4f}  greedy_queue={np.mean(gq):.4f}  "
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
               n_updates=args.n_updates, rollout_steps=args.rollout_steps,
               delay_minutes=args.delay_minutes, k=args.k)),
    f"{TAG_ARM}_eval_results.json")

print(f"[{elapsed()}] === SEMUA SELESAI (MasterEV) ===", flush=True)
