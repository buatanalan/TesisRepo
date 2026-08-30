"""Pelatihan MASTER **benar-benar murni** (`master_pure_policy.py`/`master_pure_
trainer.py`, 2026-08-28) -- diverifikasi langsung thd `2102.07359v1.pdf`, kelas BARU
tak menumpuk pada `master_ddpg_*` lama. Rujukan lengkap deviasi & keputusan desain:
`Eksekusi_RL/ARSITEKTUR_MASTER_REFERENSI.md`.

Dua mode, WAJIB dijalankan berurutan:
  1. `--mode pretrain_specialist --stream-select 0` (wait/CWT-analog)
  2. `--mode pretrain_specialist --stream-select 1` (gini/pemerataan, pengganti CP)
  3. `--mode dgr` (butuh checkpoint #1 & #2 sudah ada, di-load otomatis by tag)

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_pure_pipeline.py \
        --mode pretrain_specialist --stream-select 0 \
        > Eksekusi_RL/outputs/master_pure_pipeline.log 2>&1 &
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.master_pure_trainer import MasterPureTrainer
from marl_spklu.rl.master_pure_policy import MasterPureActor, MasterPureCritic
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER
from marl_spklu.rl.rewards import RewardCalculator

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

p = argparse.ArgumentParser()
p.add_argument("--mode", type=str, required=True, choices=["pretrain_specialist", "dgr"])
p.add_argument("--stream-select", type=int, default=None, choices=[0, 1],
              help="WAJIB bila --mode pretrain_specialist. 0=wait(CWT-analog), "
                   "1=gini/pemerataan(pengganti CP, keputusan 2026-08-28).")
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--updates-per-chunk", type=int, default=20,
              help="Langkah gradien MAKS per chunk (baku 20, dibatasi min(n_new, ini)). "
                   "2026-08-31: dgn baku ini DDPG hanya menerima ~1/3 langkah gradien "
                   "PPO meski memproses jumlah transisi lingkungan yg SAMA (54,5rb vs "
                   "54,9rb, rasio 1,01x) -- krn PPO mengulang 10 epoch per batch sedangkan "
                   "DDPG dibatasi 20 langkah/chunk terlepas dari ukuran buffer replay. "
                   "Pakai --updates-per-chunk 62 utk anggaran gradien setara PPO (rasio "
                   "terukur 3,11x, dihitung via _hitung_langkah_gradien.py).")
p.add_argument("--dataset", type=str, default="4x",
              help="'4x' -> scenario_dataset_klaster12_4x.json (baku 30d). Path lain "
                   "utk dataset custom (mis. 90d) -- WAJIB dibarengi --horizon eksplisit.")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--specialist0-tag", type=str, default=None,
              help="Tag dasar checkpoint spesialis stream 0 (baku: master_pure_specialist0"
                   "_{horizon}). Hanya berlaku --mode dgr.")
p.add_argument("--specialist1-tag", type=str, default=None,
              help="Sama utk stream 1. Hanya berlaku --mode dgr.")
p.add_argument("--specialist-seed", type=int, default=0,
              help="Seed spesialis mana yg dipakai sbg Q*/b* beku (baku seed=0).")
p.add_argument("--overwrite", action="store_true",
              help="Timpa eval_results.json lama meski n_updates lebih kecil (lih. "
                   "pengaman insiden 2026-08-23 di _run_master_ev_ppo_pipeline.py).")
p.add_argument("--wait-reward-clip", type=float, default=None,
              help="Klip opsional pd `improvement` wait_reward (satuan wait_scale) -- "
                   "sama mekanisme diuji di _run_master_pure_ppo_pipeline.py, kini diuji "
                   "jg utk DDPG murni (2026-08-29). None=perilaku lama (tak diklip).")
p.add_argument("--wait-fail-threshold", type=float, default=None,
              help="Ambang gagal (menit, replika CWT paper MASTER) -- penalti TETAP saat "
                   "wait_actual > ambang. None=nonaktif.")
p.add_argument("--wait-fail-penalty", type=float, default=-1.0,
              help="Penalti TETAP (satuan wait_scale) saat wait_actual > --wait-fail-threshold.")
args = p.parse_args()

if args.mode == "pretrain_specialist":
    assert args.stream_select is not None, "--mode pretrain_specialist WAJIB --stream-select 0|1"
else:
    assert args.specialist0_tag or True  # boleh pakai baku, dibangun di bawah

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
DATASET = _DATASET_4X if args.dataset == "4x" else os.path.join(common.ROOT, args.dataset)
if args.dataset != "4x":
    assert args.horizon != "30d", (
        "--dataset kustom diberikan tapi --horizon masih baku ('30d') -- kemungkinan "
        "kekeliruan. Sertakan --horizon eksplisit, mis. --horizon 90d.")

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
_clip_suffix = "" if args.wait_reward_clip is None else f"_clip{args.wait_reward_clip:g}"
_fail_suffix = ("" if args.wait_fail_threshold is None
               else f"_cwtfail{args.wait_fail_threshold:g}pen{args.wait_fail_penalty:g}")
_clip_suffix = _clip_suffix + _fail_suffix + (
    "" if args.updates_per_chunk == 20 else f"_upc{args.updates_per_chunk}")

if args.mode == "pretrain_specialist":
    STREAM_NAME = {0: "wait", 1: "gini"}[args.stream_select]
    TAG_ARM = f"master_pure_specialist{args.stream_select}_{STREAM_NAME}{_horizon_suffix}{_clip_suffix}"
else:
    TAG_ARM = f"master_pure_dgr{_horizon_suffix}{_clip_suffix}"

print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} mode={args.mode} "
     f"stream_select={args.stream_select}", flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"n_train_seed={args.n_train_seed}", flush=True)


def _specialist_tag(stream: int, explicit: str):
    if explicit:
        return explicit
    name = {0: "wait", 1: "gini"}[stream]
    return f"master_pure_specialist{stream}_{name}{_horizon_suffix}{_clip_suffix}"


def _load_specialist(tag: str, seed: int):
    actor = MasterPureActor(STATION_FEAT_DIM_MASTER)
    critic = MasterPureCritic(STATION_FEAT_DIM_MASTER, n_critics=1)
    a_path = os.path.join(common.OUTDIR, f"{tag}_actor_seed{seed}.pt")
    c_path = os.path.join(common.OUTDIR, f"{tag}_critic_seed{seed}.pt")
    assert os.path.exists(a_path), f"checkpoint spesialis tak ditemukan: {a_path}"
    assert os.path.exists(c_path), f"checkpoint spesialis tak ditemukan: {c_path}"
    actor.load_state_dict(torch.load(a_path, map_location="cpu"))
    critic.load_state_dict(torch.load(c_path, map_location="cpu"))
    actor.eval(); critic.eval()
    return actor, critic


def train_one(seed):
    kw = dict(dataset_path=DATASET, mode=args.mode, rollout_steps=args.rollout_steps,
             seed=seed, verbose=False, updates_per_chunk=args.updates_per_chunk)
    if args.wait_reward_clip is not None or args.wait_fail_threshold is not None:
        kw["reward_calc"] = RewardCalculator(wait_reward_clip=args.wait_reward_clip,
                                             wait_fail_threshold=args.wait_fail_threshold,
                                             wait_fail_penalty=args.wait_fail_penalty)
    if args.mode == "pretrain_specialist":
        kw["stream_select"] = args.stream_select
    else:
        tag0 = _specialist_tag(0, args.specialist0_tag)
        tag1 = _specialist_tag(1, args.specialist1_tag)
        specialists = [_load_specialist(tag0, args.specialist_seed),
                      _load_specialist(tag1, args.specialist_seed)]
        kw["specialists"] = specialists
    tr = MasterPureTrainer(**kw)
    actor, critic = tr.train(n_updates=args.n_updates)
    actor_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    critic_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_critic_seed{seed}.pt")
    torch.save(actor.state_dict(), actor_ckpt)
    torch.save(critic.state_dict(), critic_ckpt)
    return dict(seed=seed, actor_ckpt=actor_ckpt, critic_ckpt=critic_ckpt, history=tr.history)


print(f"[{elapsed()}] === PELATIHAN MasterPure ({args.n_train_seed} seed) ===", flush=True)
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
            f"Training & checkpoint SUDAH tersimpan dgn aman di atas -- ulangi HANYA "
            f"perintah ini dgn --overwrite bila memang bermaksud menimpa.")

common.save_json(dict(
    config=dict(mode=args.mode, stream_select=args.stream_select, n_train_seed=args.n_train_seed,
               n_updates=args.n_updates, rollout_steps=args.rollout_steps, horizon=args.horizon,
               dataset=DATASET),
), f"{TAG_ARM}_eval_results.json")
print(f"[{elapsed()}] === SEMUA SELESAI ({TAG_ARM}) ===", flush=True)
