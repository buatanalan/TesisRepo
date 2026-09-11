"""Pelatihan Master-DDPG (`master_pure_trainer.py::MasterPureTrainer`) MODE
ALIRAN-MURNI (`pure_streams=True`, 2026-09-12) -- menggantikan skema reward LAMA
2-aliran (wait/CWT-analog + gini/pemerataan) dgn 3 aliran murni (wait / gini /
acceptance), MENYAMAKAN skema reward yg dipakai lengan Hybrid-PPO (seluruh hasil
ablasi utama Bab V). Lihat `_run_master_pure_hybrid_ppo_pipeline.py` utk rujukan
pola CLI & konstruksi `RewardCalculator` yg DISALIN PERSIS di sini.

SCRIPT BARU, TERPISAH dari `_run_master_pure_pipeline.py` (skema LAMA, DDPG, dasar
Tabel VI.1/VI.2 -- TETAP dipertahankan apa adanya, TIDAK diedit). Checkpoint keluaran
skrip ini diberi penanda `pure3` di tag supaya TIDAK menimpa checkpoint lama.

CATATAN gap-ratio Pers.13 (2026-09-12): trainer DDPG menghitung gap thd Q SPESIALIS
beku (BUKAN return teragregasi spt versi PPO) -- `beta_denom="ret_std"` di sini
diADAPTASI menjadi "penyebut = simpangan-baku Q*_k spesialis dalam batch" (lih.
catatan lengkap `MasterPureTrainer._compute_beta_dgr`), properti kebal-skala yg SAMA
dipertahankan meski mekanismenya tak identik literal Pers.13 return-based.

Empat tahap, sama pola versi 2-aliran:
  1. `--mode pretrain_specialist --stream-select 0` (wait)       -> checkpoint spesialis
  2. `--mode pretrain_specialist --stream-select 1` (gini)       -> checkpoint spesialis
  3. `--mode pretrain_specialist --stream-select 2` (acceptance) -> checkpoint spesialis
  4. `--mode dgr` (checkpoint #1,#2,#3 dimuat otomatis by tag)

WAJIB `--alpha-accept != 0` (aliran acceptance kosong tanpa itu).

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_ddpg_pure3_pipeline.py \
        --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0 \
        > Eksekusi_RL/outputs/master_ddpg_pure3_pipeline.log 2>&1 &
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import common
from marl_spklu.rl.master_pure_trainer import MasterPureTrainer
from marl_spklu.rl.master_pure_policy import MasterPureActorV2, MasterPureCritic
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.rollout import STREAM_GLOBAL

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
p.add_argument("--updates-per-chunk", type=int, default=20)
p.add_argument("--dataset", type=str, default="4x")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--specialist0-tag", type=str, default=None)
p.add_argument("--specialist1-tag", type=str, default=None)
p.add_argument("--specialist2-tag", type=str, default=None)
p.add_argument("--specialist-seed", type=int, default=0,
              help="Seed spesialis mana yg dipakai sbg Q*/b* beku (baku seed=0).")
p.add_argument("--overwrite", action="store_true")
p.add_argument("--wait-reward-clip", type=float, default=None,
              help="Klip opsional pd `improvement` wait_reward. None=perilaku lama.")
p.add_argument("--wait-fail-threshold", type=float, default=None,
              help="Ambang gagal (menit, replika CWT paper MASTER). None=nonaktif.")
p.add_argument("--wait-fail-penalty", type=float, default=-1.0,
              help="Penalti TETAP (satuan wait_scale) saat wait_actual > --wait-fail-threshold.")
p.add_argument("--reward-preset", type=str, default="raw", choices=["raw", "seimbang4x"],
              help="'raw' (BAKU -- RewardCalculator() mentah) | 'seimbang4x' (preset "
                   "TERKALIBRASI, use_delta_gini=True) -- SAMA pilihan & semantik dgn "
                   "`_run_master_pure_hybrid_ppo_pipeline.py`, DISALIN PERSIS.")
p.add_argument("--alpha-accept", type=float, default=1.0,
              help="Bobot suku kepatuhan SIMETRIS. WAJIB != 0 di mode aliran-murni.")
p.add_argument("--beta-denom", type=str, default=None, choices=["r_star", "ret_std"],
              help="Penyebut gap-ratio DGR (lih. catatan adaptasi di docstring modul). "
                   "Baku 'ret_std' di mode aliran-murni (WAJIB).")
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
    assert args.horizon != "30d", (
        "--dataset kustom diberikan tapi --horizon masih baku -- sertakan --horizon eksplisit.")

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
_clip_suffix = "" if args.wait_reward_clip is None else f"_clip{args.wait_reward_clip:g}"
_fail_suffix = ("" if args.wait_fail_threshold is None
               else f"_cwtfail{args.wait_fail_threshold:g}pen{args.wait_fail_penalty:g}")
_rw_suffix = "" if args.reward_preset == "raw" else f"_{args.reward_preset}"
_acc_suffix = f"_acc{args.alpha_accept:g}"
_upc_suffix = "" if args.updates_per_chunk == 20 else f"_upc{args.updates_per_chunk}"
_clip_suffix = _clip_suffix + _fail_suffix + _rw_suffix + _acc_suffix + _upc_suffix

# `_svh` (2026-09-12): penanda arsitektur aktor `StationVectorHead` (BUKAN MLP polos
# lama) -- WAJIB beda dari tag pure3 lama supaya checkpoint arsitektur-beda tak
# saling menimpa (bentuk state_dict genuinely berbeda).
if args.mode == "pretrain_specialist":
    STREAM_NAME = {0: "wait", 1: "gini", 2: "accept"}[args.stream_select]
    TAG_ARM = f"master_pure_ddpg_pure3_svh_specialist{args.stream_select}_{STREAM_NAME}{_horizon_suffix}{_clip_suffix}"
else:
    TAG_ARM = f"master_pure_ddpg_pure3_svh_dgr{_horizon_suffix}{_clip_suffix}"

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
    return f"master_pure_ddpg_pure3_svh_specialist{stream}_{name}{_horizon_suffix}{_clip_suffix}"


def _load_specialist(tag: str, seed: int):
    # `MasterPureActorV2` (StationVectorHead) -- spesialis pipeline SVH baru ini SELALU
    # arsitektur V2, konsisten dgn aktor `train_one` di bawah (actor_cls=MasterPureActorV2).
    actor = MasterPureActorV2(STATION_FEAT_DIM_MASTER)
    critic = MasterPureCritic(STATION_FEAT_DIM_MASTER, n_critics=1)
    a_path = os.path.join(common.OUTDIR, f"{tag}_actor_seed{seed}.pt")
    c_path = os.path.join(common.OUTDIR, f"{tag}_critic_seed{seed}.pt")
    assert os.path.exists(a_path), f"checkpoint spesialis tak ditemukan: {a_path}"
    assert os.path.exists(c_path), f"checkpoint spesialis tak ditemukan: {c_path}"
    actor.load_state_dict(torch.load(a_path, map_location="cpu"))
    critic.load_state_dict(torch.load(c_path, map_location="cpu"))
    actor.eval(); critic.eval()
    return actor, critic


def _build_reward_calc():
    # Pola KONSTRUKSI PERSIS SAMA `_run_master_pure_hybrid_ppo_pipeline.py::train_one`.
    _rc_kw = dict(wait_reward_clip=args.wait_reward_clip,
                  wait_fail_threshold=args.wait_fail_threshold,
                  wait_fail_penalty=args.wait_fail_penalty,
                  alpha_accept=args.alpha_accept)
    return (RewardCalculator.seimbang4x(**_rc_kw) if args.reward_preset == "seimbang4x"
           else RewardCalculator(**_rc_kw))


def _actor_cls_svh(n_spklu, **kw):
    # Trainer memanggil `actor_cls(self.N, **actor_kwargs)` (pola LAMA utk aktor
    # Hybrid yg argumen pertamanya `n_spklu`) -- `MasterPureActorV2` argumen
    # pertamanya `station_feat_dim` (SAMA pola `MasterPureActor` asli), jadi `n_spklu`
    # DIABAIKAN sengaja di sini (bukan bug), pola wrapper minimal-invasif drpd
    # mengubah signature trainer generik yg juga dipakai lengan Hybrid lain.
    return MasterPureActorV2(STATION_FEAT_DIM_MASTER, **kw)


def train_one(seed):
    kw = dict(dataset_path=DATASET, mode=args.mode, rollout_steps=args.rollout_steps,
             seed=seed, verbose=False, updates_per_chunk=args.updates_per_chunk,
             pure_streams=True, beta_denom=args.beta_denom, accept_stream=STREAM_GLOBAL,
             reward_calc=_build_reward_calc(), actor_cls=_actor_cls_svh)
    if args.mode == "pretrain_specialist":
        kw["stream_select"] = args.stream_select
    else:
        tag0 = _specialist_tag(0, args.specialist0_tag)
        tag1 = _specialist_tag(1, args.specialist1_tag)
        tag2 = _specialist_tag(2, args.specialist2_tag)
        specialists = [_load_specialist(tag0, args.specialist_seed),
                      _load_specialist(tag1, args.specialist_seed),
                      _load_specialist(tag2, args.specialist_seed)]
        kw["specialists"] = specialists
    tr = MasterPureTrainer(**kw)
    actor, critic = tr.train(n_updates=args.n_updates)
    actor_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    critic_ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_critic_seed{seed}.pt")
    torch.save(actor.state_dict(), actor_ckpt)
    torch.save(critic.state_dict(), critic_ckpt)
    return dict(seed=seed, actor_ckpt=actor_ckpt, critic_ckpt=critic_ckpt, history=tr.history)


print(f"[{elapsed()}] === PELATIHAN MasterDDPG PURE3 ({args.n_train_seed} seed) ===", flush=True)
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
    print(f"[{elapsed()}]   seed={seed} -- SELESAI", flush=True)
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
            f"Training & checkpoint SUDAH tersimpan dgn aman di atas -- ulangi HANYA "
            f"perintah ini dgn --overwrite bila memang bermaksud menimpa.")

common.save_json(dict(
    config=dict(mode=args.mode, stream_select=args.stream_select, n_train_seed=args.n_train_seed,
               n_updates=args.n_updates, rollout_steps=args.rollout_steps, horizon=args.horizon,
               dataset=DATASET, pure_streams=True, beta_denom=args.beta_denom,
               reward_preset=args.reward_preset, alpha_accept=args.alpha_accept),
), f"{TAG_ARM}_eval_results.json")
print(f"[{elapsed()}] === SEMUA SELESAI ({TAG_ARM}) ===", flush=True)
