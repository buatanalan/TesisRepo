"""Pelatihan Master-PPO (`master_pure_ppo_policy.py`/`master_pure_ppo_trainer.py`,
2026-08-28) -- MASTER murni dgn tulang punggung PPO menggantikan DDPG. SATU tahap
saja (bukan 3 spt versi DDPG -- lih. docstring `master_pure_ppo_trainer.py` knp
DGR genuine-spesialis tak berlaku utk V(s), diganti gap-ratio berbasis return).

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_pure_ppo_pipeline.py \
        > Eksekusi_RL/outputs/master_pure_ppo_pipeline.log 2>&1 &
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import common
from marl_spklu.rl.master_pure_ppo_trainer import MasterPurePPOTrainer

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

p = argparse.ArgumentParser()
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--dataset", type=str, default="4x")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--overwrite", action="store_true")
args = p.parse_args()

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
DATASET = _DATASET_4X if args.dataset == "4x" else os.path.join(common.ROOT, args.dataset)
if args.dataset != "4x":
    assert args.horizon != "30d", "--dataset kustom butuh --horizon eksplisit"

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
TAG_ARM = f"master_pure_ppo{_horizon_suffix}"

print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM}", flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"n_train_seed={args.n_train_seed}", flush=True)


def train_one(seed):
    tr = MasterPurePPOTrainer(dataset_path=DATASET, rollout_steps=args.rollout_steps,
                              seed=seed, verbose=False)
    actor, critic = tr.train(n_updates=args.n_updates)
    actor_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    critic_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_critic_seed{seed}.pt")
    torch.save(actor.state_dict(), actor_ckpt)
    torch.save(critic.state_dict(), critic_ckpt)
    return dict(seed=seed, actor_ckpt=actor_ckpt, critic_ckpt=critic_ckpt, history=tr.history)


print(f"[{elapsed()}] === PELATIHAN Master-PPO ({args.n_train_seed} seed) ===", flush=True)
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
    row = train_one(seed)
    print(f"[{elapsed()}]   seed={seed} -- SELESAI", flush=True)
    results.append(row)
    common.save_json(results, results_path)
print(f"[{elapsed()}] Pelatihan selesai ({len(results)} seed)", flush=True)

_eval_out_path = os.path.join(common.OUTDIR, f"{TAG_ARM}_eval_results.json")
if os.path.exists(_eval_out_path) and not args.overwrite:
    import json as _json
    _existing_eval = _json.load(open(_eval_out_path, encoding="utf-8"))
    _existing_nu = (_existing_eval.get("config") or {}).get("n_updates")
    if _existing_nu is not None and _existing_nu > args.n_updates:
        raise SystemExit(
            f"[{elapsed()}] MENOLAK menimpa {_eval_out_path}: n_updates run ini "
            f"({args.n_updates}) LEBIH KECIL dari yang sudah tersimpan ({_existing_nu}). "
            f"ulangi HANYA perintah ini dgn --overwrite bila memang bermaksud menimpa.")

common.save_json(dict(config=dict(n_train_seed=args.n_train_seed, n_updates=args.n_updates,
                                  rollout_steps=args.rollout_steps, horizon=args.horizon,
                                  dataset=DATASET)),
                 f"{TAG_ARM}_eval_results.json")
print(f"[{elapsed()}] === SEMUA SELESAI ({TAG_ARM}) ===", flush=True)
