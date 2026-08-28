"""Pelatihan Master-PPO (`master_pure_ppo_policy.py`/`master_pure_ppo_trainer.py`,
2026-08-28) -- MASTER murni, tulang punggung PPO menggantikan DDPG. TIGA tahap
(diubah dari SATU setelah diskusi -- lih. riwayat commit), sama pola versi DDPG:

  1. `--mode pretrain_specialist --stream-select 0` (wait) -> hasilkan r_star (JSON)
  2. `--mode pretrain_specialist --stream-select 1` (gini) -> hasilkan r_star (JSON)
  3. `--mode dgr` (baca r_star #1 & #2 otomatis by tag, jadi acuan TETAP gap-ratio)

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_pure_ppo_pipeline.py \
        --mode pretrain_specialist --stream-select 0 \
        > Eksekusi_RL/outputs/master_pure_ppo_pipeline.log 2>&1 &
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import common
from marl_spklu.rl.master_pure_ppo_trainer import MasterPurePPOTrainer

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

p = argparse.ArgumentParser()
p.add_argument("--mode", type=str, required=True, choices=["pretrain_specialist", "dgr"])
p.add_argument("--stream-select", type=int, default=None, choices=[0, 1],
              help="WAJIB bila --mode pretrain_specialist. 0=wait, 1=gini.")
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--dataset", type=str, default="4x")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--specialist0-tag", type=str, default=None)
p.add_argument("--specialist1-tag", type=str, default=None)
p.add_argument("--specialist-seed", type=int, default=0,
              help="Seed spesialis mana yg r_star-nya dipakai sbg acuan (baku seed=0).")
p.add_argument("--overwrite", action="store_true")
args = p.parse_args()

if args.mode == "pretrain_specialist":
    assert args.stream_select is not None, "--mode pretrain_specialist WAJIB --stream-select 0|1"

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
DATASET = _DATASET_4X if args.dataset == "4x" else os.path.join(common.ROOT, args.dataset)
if args.dataset != "4x":
    assert args.horizon != "30d", "--dataset kustom butuh --horizon eksplisit"

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"

if args.mode == "pretrain_specialist":
    STREAM_NAME = {0: "wait", 1: "gini"}[args.stream_select]
    TAG_ARM = f"master_pure_ppo_specialist{args.stream_select}_{STREAM_NAME}{_horizon_suffix}"
else:
    TAG_ARM = f"master_pure_ppo_dgr{_horizon_suffix}"

print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} mode={args.mode} stream_select={args.stream_select}",
     flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"n_train_seed={args.n_train_seed}", flush=True)


def _specialist_tag(stream: int, explicit: str):
    if explicit:
        return explicit
    name = {0: "wait", 1: "gini"}[stream]
    return f"master_pure_ppo_specialist{stream}_{name}{_horizon_suffix}"


def _load_r_star(tag: str, seed: int) -> float:
    path = os.path.join(common.OUTDIR, f"{tag}_r_star_seed{seed}.json")
    assert os.path.exists(path), f"r_star spesialis tak ditemukan: {path}"
    return json.load(open(path, encoding="utf-8"))["r_star"]


def train_one(seed):
    kw = dict(dataset_path=DATASET, mode=args.mode, rollout_steps=args.rollout_steps,
             seed=seed, verbose=False)
    if args.mode == "pretrain_specialist":
        kw["stream_select"] = args.stream_select
    else:
        tag0 = _specialist_tag(0, args.specialist0_tag)
        tag1 = _specialist_tag(1, args.specialist1_tag)
        r0 = _load_r_star(tag0, args.specialist_seed)
        r1 = _load_r_star(tag1, args.specialist_seed)
        kw["specialist_r_star"] = [r0, r1]
        print(f"[{elapsed()}]   r_star dimuat: wait={r0:.4f} gini={r1:.4f}", flush=True)

    tr = MasterPurePPOTrainer(**kw)
    result = tr.train(n_updates=args.n_updates)
    if args.mode == "pretrain_specialist":
        actor, critic, r_star = result
        common.save_json({"r_star": r_star}, f"{TAG_ARM}_r_star_seed{seed}.json")
    else:
        actor, critic = result
        r_star = None
    actor_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    critic_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_critic_seed{seed}.pt")
    torch.save(actor.state_dict(), actor_ckpt)
    torch.save(critic.state_dict(), critic_ckpt)
    return dict(seed=seed, actor_ckpt=actor_ckpt, critic_ckpt=critic_ckpt,
               r_star=r_star, history=tr.history)


print(f"[{elapsed()}] === PELATIHAN Master-PPO ({args.n_train_seed} seed) ===", flush=True)
results = []
results_path = f"{TAG_ARM}_training_results.json"
existing = {}
try:
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
    print(f"[{elapsed()}]   seed={seed} -- SELESAI"
         + (f" (r_star={row['r_star']:.4f})" if row["r_star"] is not None else ""), flush=True)
    results.append(row)
    common.save_json(results, results_path)
print(f"[{elapsed()}] Pelatihan selesai ({len(results)} seed)", flush=True)

_eval_out_path = os.path.join(common.OUTDIR, f"{TAG_ARM}_eval_results.json")
if os.path.exists(_eval_out_path) and not args.overwrite:
    _existing_eval = json.load(open(_eval_out_path, encoding="utf-8"))
    _existing_nu = (_existing_eval.get("config") or {}).get("n_updates")
    if _existing_nu is not None and _existing_nu > args.n_updates:
        raise SystemExit(
            f"[{elapsed()}] MENOLAK menimpa {_eval_out_path}: n_updates run ini "
            f"({args.n_updates}) LEBIH KECIL dari yang sudah tersimpan ({_existing_nu}). "
            f"ulangi HANYA perintah ini dgn --overwrite bila memang bermaksud menimpa.")

common.save_json(dict(config=dict(mode=args.mode, stream_select=args.stream_select,
                                  n_train_seed=args.n_train_seed, n_updates=args.n_updates,
                                  rollout_steps=args.rollout_steps, horizon=args.horizon,
                                  dataset=DATASET)),
                 f"{TAG_ARM}_eval_results.json")
print(f"[{elapsed()}] === SEMUA SELESAI ({TAG_ARM}) ===", flush=True)
