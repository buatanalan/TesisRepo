"""Pelatihan Master-PPO MODE ALIRAN-MURNI (`pure_streams=True`, 2026-09-12) --
menggantikan skema reward LAMA 2-aliran (wait+prox / gini+flock) dgn 3 aliran murni
(wait / gini / acceptance) di `MasterPurePPOTrainer`, MENYAMAKAN skema reward yg
dipakai lengan Hybrid-PPO (seluruh hasil ablasi utama Bab V) -- lihat
`_run_master_pure_hybrid_ppo_pipeline.py` utk rujukan pola CLI & konstruksi
`RewardCalculator` yg DISALIN PERSIS di sini (bukan angka baru).

SCRIPT BARU, TERPISAH dari `_run_master_pure_ppo_pipeline.py` (skema LAMA, dasar
Tabel VI.1/VI.2 -- TETAP dipertahankan apa adanya utk kompatibilitas mundur, TIDAK
diedit). Checkpoint keluaran skrip ini diberi penanda `pure3` di tag supaya TIDAK
menimpa checkpoint lama.

Tiga tahap, sama pola versi 2-aliran:
  1. `--mode pretrain_specialist --stream-select 0` (wait)  -> r_star
  2. `--mode pretrain_specialist --stream-select 1` (gini)  -> r_star
  3. `--mode pretrain_specialist --stream-select 2` (acceptance) -> r_star
  4. `--mode dgr` (r_star #1,#2,#3 dimuat otomatis by tag)

WAJIB `--alpha-accept != 0` (aliran acceptance kosong tanpa itu, sama pengaman
`_run_master_pure_hybrid_ppo_pipeline.py`).

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_pure_ppo_pure3_pipeline.py \
        --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0 \
        > Eksekusi_RL/outputs/master_pure_ppo_pure3_pipeline.log 2>&1 &
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import common
from marl_spklu.rl.master_pure_ppo_trainer import MasterPurePPOTrainer
from marl_spklu.rl.master_pure_ppo_policy import MasterPurePPOActorV2
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.rollout import STREAM_INDIVIDUAL, STREAM_GLOBAL

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

p = argparse.ArgumentParser()
p.add_argument("--mode", type=str, required=True, choices=["pretrain_specialist", "dgr"])
p.add_argument("--stream-select", type=int, default=None, choices=[0, 1, 2],
              help="WAJIB bila --mode pretrain_specialist. 0=wait, 1=gini, 2=acceptance.")
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--dataset", type=str, default="4x")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--specialist0-tag", type=str, default=None)
p.add_argument("--specialist1-tag", type=str, default=None)
p.add_argument("--specialist2-tag", type=str, default=None)
p.add_argument("--specialist-seed", type=int, default=0,
              help="Seed spesialis mana yg r_star-nya dipakai sbg acuan (baku seed=0).")
p.add_argument("--overwrite", action="store_true")
p.add_argument("--wait-reward-clip", type=float, default=None,
              help="Klip opsional pd `improvement` wait_reward -- sama mekanisme lengan "
                   "lain. None=perilaku lama (tak diklip).")
p.add_argument("--wait-fail-threshold", type=float, default=None,
              help="Ambang gagal (menit, replika CWT paper MASTER). None=nonaktif.")
p.add_argument("--wait-fail-penalty", type=float, default=-1.0,
              help="Penalti TETAP (satuan wait_scale) saat wait_actual > --wait-fail-threshold.")
p.add_argument("--reward-preset", type=str, default="raw", choices=["raw", "seimbang4x"],
              help="'raw' (BAKU -- RewardCalculator() mentah) | 'seimbang4x' (preset "
                   "TERKALIBRASI, use_delta_gini=True) -- SAMA pilihan & semantik dgn "
                   "`_run_master_pure_hybrid_ppo_pipeline.py`, DISALIN PERSIS (bukan "
                   "angka baru).")
p.add_argument("--alpha-accept", type=float, default=1.0,
              help="Bobot suku kepatuhan SIMETRIS. WAJIB != 0 di mode aliran-murni "
                   "(aliran acceptance kosong tanpa itu) -- nilainya sendiri TAK "
                   "berpengaruh krn penskala seragam satu-suku lenyap di normalisasi "
                   "advantage & gap-ratio (lih. catatan STREAM_PURE_* di rollout.py).")
p.add_argument("--accept-stream", type=str, default="global", choices=["global", "individual"],
              help="Tak berlaku scr struktural di --pure-streams (acceptance selalu "
                   "aliran sendiri) -- disediakan HANYA supaya signature CLI identik "
                   "dgn lengan Hybrid, diabaikan diam-diam.")
p.add_argument("--beta-denom", type=str, default=None, choices=["r_star", "ret_std"],
              help="Penyebut gap-ratio DGR. Baku 'ret_std' di mode aliran-murni (WAJIB -- "
                   "lih. catatan di `master_pure_ppo_trainer.py::_compute_beta`, sama "
                   "alasan `MasterHybridPPOTrainer`).")
args = p.parse_args()

if args.mode == "pretrain_specialist":
    assert args.stream_select is not None, "--mode pretrain_specialist WAJIB --stream-select 0|1|2"
assert args.alpha_accept != 0.0, (
    "--alpha-accept WAJIB != 0 di pipeline pure3 (aliran acceptance akan kosong tanpa itu)")
if args.beta_denom is None:
    args.beta_denom = "ret_std"

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
DATASET = _DATASET_4X if args.dataset == "4x" else os.path.join(common.ROOT, args.dataset)
if args.dataset != "4x":
    assert args.horizon != "30d", "--dataset kustom butuh --horizon eksplisit"

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
_clip_suffix = "" if args.wait_reward_clip is None else f"_clip{args.wait_reward_clip:g}"
_fail_suffix = ("" if args.wait_fail_threshold is None
               else f"_cwtfail{args.wait_fail_threshold:g}pen{args.wait_fail_penalty:g}")
_rw_suffix = "" if args.reward_preset == "raw" else f"_{args.reward_preset}"
_acc_suffix = f"_acc{args.alpha_accept:g}"
_clip_suffix = _clip_suffix + _fail_suffix + _rw_suffix + _acc_suffix

# `_svh` (2026-09-12): penanda arsitektur aktor `StationVectorHead` (BUKAN MLP polos
# lama) -- WAJIB beda dari tag pure3 lama supaya checkpoint arsitektur-beda tak
# saling menimpa (bentuk state_dict genuinely berbeda).
if args.mode == "pretrain_specialist":
    STREAM_NAME = {0: "wait", 1: "gini", 2: "accept"}[args.stream_select]
    TAG_ARM = f"master_pure_ppo_pure3_svh_specialist{args.stream_select}_{STREAM_NAME}{_horizon_suffix}{_clip_suffix}"
else:
    TAG_ARM = f"master_pure_ppo_pure3_svh_dgr{_horizon_suffix}{_clip_suffix}"

print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} mode={args.mode} stream_select={args.stream_select}",
     flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"n_train_seed={args.n_train_seed}", flush=True)
print(f"[{elapsed()}] DGR: penyebut gap-ratio = {args.beta_denom}", flush=True)


def _specialist_tag(stream: int, explicit: str):
    if explicit:
        return explicit
    name = {0: "wait", 1: "gini", 2: "accept"}[stream]
    return f"master_pure_ppo_pure3_svh_specialist{stream}_{name}{_horizon_suffix}{_clip_suffix}"


def _load_r_star(tag: str, seed: int) -> float:
    path = os.path.join(common.OUTDIR, f"{tag}_r_star_seed{seed}.json")
    assert os.path.exists(path), f"r_star spesialis tak ditemukan: {path}"
    return json.load(open(path, encoding="utf-8"))["r_star"]


def _build_reward_calc():
    # Pola KONSTRUKSI PERSIS SAMA `_run_master_pure_hybrid_ppo_pipeline.py::train_one`
    # -- 'raw' = RewardCalculator() default (alpha_wait=1.0 alpha_gini=0.5), 'seimbang4x'
    # = preset terkalibrasi (use_delta_gini=True). DISALIN, bukan direka ulang.
    _rc_kw = dict(wait_reward_clip=args.wait_reward_clip,
                  wait_fail_threshold=args.wait_fail_threshold,
                  wait_fail_penalty=args.wait_fail_penalty,
                  alpha_accept=args.alpha_accept)
    return (RewardCalculator.seimbang4x(**_rc_kw) if args.reward_preset == "seimbang4x"
           else RewardCalculator(**_rc_kw))


def train_one(seed):
    kw = dict(dataset_path=DATASET, mode=args.mode, rollout_steps=args.rollout_steps,
             seed=seed, verbose=False, pure_streams=True, beta_denom=args.beta_denom,
             accept_stream=STREAM_GLOBAL, reward_calc=_build_reward_calc(),
             actor_cls=MasterPurePPOActorV2)
    if args.mode == "pretrain_specialist":
        kw["stream_select"] = args.stream_select
    else:
        tag0 = _specialist_tag(0, args.specialist0_tag)
        tag1 = _specialist_tag(1, args.specialist1_tag)
        tag2 = _specialist_tag(2, args.specialist2_tag)
        r0 = _load_r_star(tag0, args.specialist_seed)
        r1 = _load_r_star(tag1, args.specialist_seed)
        r2 = _load_r_star(tag2, args.specialist_seed)
        kw["specialist_r_star"] = [r0, r1, r2]
        print(f"[{elapsed()}]   r_star dimuat: wait={r0:.4f} gini={r1:.4f} accept={r2:.4f}",
             flush=True)

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


print(f"[{elapsed()}] === PELATIHAN Master-PPO PURE3 ({args.n_train_seed} seed) ===", flush=True)
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
                                  dataset=DATASET, pure_streams=True,
                                  beta_denom=args.beta_denom,
                                  reward_preset=args.reward_preset,
                                  alpha_accept=args.alpha_accept)),
                 f"{TAG_ARM}_eval_results.json")
print(f"[{elapsed()}] === SEMUA SELESAI ({TAG_ARM}) ===", flush=True)
