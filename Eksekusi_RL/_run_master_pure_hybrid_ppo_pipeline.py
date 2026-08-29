"""Pelatihan Master-Hybrid PPO (2026-08-29) -- sama arsitektur aktor Hybrid-DDPG
(modul P late-inject + station attention) tapi kepala akhir softmax/kategorikal
(`MasterHybridPPOActor`, `master_pure_hybrid_trainer.py::MasterHybridPPOTrainer`).

Tiga tahap sama pola `_run_master_pure_ppo_pipeline.py`:
  1. `--mode pretrain_specialist --stream-select 0` (wait) -> r_star
  2. `--mode pretrain_specialist --stream-select 1` (gini) -> r_star
  3. `--mode dgr` (r_star #1&#2 dimuat otomatis by tag)

Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py \
        --mode pretrain_specialist --stream-select 0 \
        > Eksekusi_RL/outputs/master_pure_hybrid_ppo_pipeline.log 2>&1 &
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import common
from marl_spklu.rl.master_pure_hybrid_trainer import MasterHybridPPOTrainer
from marl_spklu.rl.rewards import RewardCalculator

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

ACTOR_KW_BASE = dict(vec_dim=8, bid_hidden=16, pref_d_lstm=8, pref_d_attn=8, station_attn_dim=8)

p = argparse.ArgumentParser()
p.add_argument("--mode", type=str, required=True, choices=["pretrain_specialist", "dgr"])
p.add_argument("--stream-select", type=int, default=None, choices=[0, 1])
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--dataset", type=str, default="4x")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--specialist0-tag", type=str, default=None)
p.add_argument("--specialist1-tag", type=str, default=None)
p.add_argument("--specialist-seed", type=int, default=0)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--wait-reward-clip", type=float, default=None,
              help="Klip opsional pd `improvement` wait_reward (satuan wait_scale) -- "
                   "sama mekanisme diuji di lengan PPO/DDPG murni, kini diuji jg utk "
                   "Hybrid-PPO (2026-08-29). None=perilaku lama (tak diklip).")
p.add_argument("--wait-fail-threshold", type=float, default=None,
              help="Ambang gagal (menit, replika CWT paper MASTER) -- penalti TETAP saat "
                   "wait_actual > ambang. None=nonaktif.")
p.add_argument("--wait-fail-penalty", type=float, default=-1.0,
              help="Penalti TETAP (satuan wait_scale) saat wait_actual > --wait-fail-threshold.")
p.add_argument("--pref-feature-mode", action="store_true",
              help="Riwayat preferensi sbg PASANGAN VEKTOR FITUR stasiun ([jarak, est_wait, "
                   "antrean, konektor, utilisasi] utk direkomendasikan ++ dipilih, 10 dim) "
                   "-- bukan one-hot identitas paper PDQN. Sama mode dipakai Kandidat A.")
p.add_argument("--ev-obs", action="store_true",
              help="Observasi (K x 10) sesuai spesifikasi o_i Bab IV: 7 fitur stasiun "
                   "+ 3 fitur PEMOHON yg disiarkan (jarak relatif, SoC, kapasitas "
                   "baterai) via `build_joint_obs_master_ev`. BAKU MATI -> tetap 7 fitur "
                   "§3.1 murni (Pers.11, stasiun buta thd pemohon). Tanpa flag ini agen "
                   "hanya mengenali pemohon lewat riwayat (pref_hist), TIDAK lewat "
                   "keadaan fisiknya saat ini -- shg tak bisa menalar keterjangkauan "
                   "maupun range-anxiety utk permintaan yang sedang dilayani.")
p.add_argument("--pref-gate-init", type=float, default=0.0,
              help="Nilai AWAL gerbang preferensi (baku 0.0 = GTrXL zero-init, perilaku "
                   "lama). MASALAH: pada gerbang PERSIS 0 gradien ke pref_lstm/pref_attn "
                   "adalah PERSIS NOL, jadi modul P tak bisa mulai belajar (deadlock "
                   "ayam-telur). Nilai kecil bukan-nol (mis. 0.1) memutusnya.")
p.add_argument("--pref-pair-outcome", action="store_true",
              help="Tempelkan blok HASIL [complied, realized_gap_norm] di belakang pasangan "
                   "fitur (10 -> 12 dim, 2026-08-29). `realized_gap` = besaran yg SAMA "
                   "dipakai User.update_trust, jadi `pref_lstm` menduga preferensi DAN "
                   "kepercayaan sekaligus -- `hist_lstm` terpisah jadi mubazir, bukan "
                   "bersaing. WAJIB --pref-feature-mode.")
p.add_argument("--reward-preset", type=str, default="raw", choices=["raw", "seimbang4x"],
              help="'raw' (BAKU -- RewardCalculator() mentah, alpha_wait=1.0 alpha_gini=0.5, "
                   "TAK PERNAH dikalibrasi utk rezim 4x) | 'seimbang4x' (RewardCalculator."
                   "seimbang4x() -- preset TERKALIBRASI, alpha_wait=0.0046 alpha_gini=2.6019, "
                   "use_delta_gini=True). Diuji 2026-08-29: satu2nya komponen Kandidat A yg "
                   "belum pernah ditransplantasi ke lengan MASTER manapun, diduga faktor "
                   "paling berpengaruh thd gap gini/entropy/herding vs Kandidat A. Digabung "
                   "dgn --wait-fail-* via override kwargs (kompatibel).")
args = p.parse_args()

if args.mode == "pretrain_specialist":
    assert args.stream_select is not None, "--mode pretrain_specialist WAJIB --stream-select 0|1"

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
DATASET = _DATASET_4X if args.dataset == "4x" else os.path.join(common.ROOT, args.dataset)
if args.dataset != "4x":
    assert args.horizon != "30d", "--dataset kustom butuh --horizon eksplisit"

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
_clip_suffix = "" if args.wait_reward_clip is None else f"_clip{args.wait_reward_clip:g}"
_fail_suffix = ("" if args.wait_fail_threshold is None
               else f"_cwtfail{args.wait_fail_threshold:g}pen{args.wait_fail_penalty:g}")
_rw_suffix = "" if args.reward_preset == "raw" else f"_{args.reward_preset}"
assert not (args.pref_pair_outcome and not args.pref_feature_mode), (
    "--pref-pair-outcome WAJIB disertai --pref-feature-mode (blok hasil tak bermakna "
    "di mode one-hot identitas)")
from marl_spklu.rl.master_paper_obs import (STATION_FEAT_DIM_MASTER,
                                            STATION_FEAT_DIM_MASTER_EV)
ACTOR_KW = dict(ACTOR_KW_BASE, pref_feature_mode=args.pref_feature_mode,
                pref_pair_outcome=args.pref_pair_outcome,
                pref_gate_init=args.pref_gate_init,
                station_feat_dim=(STATION_FEAT_DIM_MASTER_EV if args.ev_obs
                                  else STATION_FEAT_DIM_MASTER))
_pref_suffix = (("_preffeat" if args.pref_feature_mode else "")
                + ("_pairout" if args.pref_pair_outcome else "")
                + ("" if args.pref_gate_init == 0.0 else f"_pg{args.pref_gate_init:g}")
                + ("_evobs" if args.ev_obs else ""))
_clip_suffix = _clip_suffix + _fail_suffix + _rw_suffix + _pref_suffix
if args.mode == "pretrain_specialist":
    STREAM_NAME = {0: "wait", 1: "gini"}[args.stream_select]
    TAG_ARM = f"master_hybrid_ppo_specialist{args.stream_select}_{STREAM_NAME}{_horizon_suffix}{_clip_suffix}"
else:
    TAG_ARM = f"master_hybrid_ppo_dgr{_horizon_suffix}{_clip_suffix}"

print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} mode={args.mode} stream_select={args.stream_select}",
     flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"n_train_seed={args.n_train_seed}", flush=True)


def _specialist_tag(stream: int, explicit: str):
    if explicit:
        return explicit
    name = {0: "wait", 1: "gini"}[stream]
    return f"master_hybrid_ppo_specialist{stream}_{name}{_horizon_suffix}{_clip_suffix}"


def _load_r_star(tag: str, seed: int) -> float:
    path = os.path.join(common.OUTDIR, f"{tag}_r_star_seed{seed}.json")
    assert os.path.exists(path), f"r_star spesialis tak ditemukan: {path}"
    return json.load(open(path, encoding="utf-8"))["r_star"]


def train_one(seed):
    kw = dict(dataset_path=DATASET, mode=args.mode, rollout_steps=args.rollout_steps,
             seed=seed, verbose=False, actor_kwargs=ACTOR_KW)
    if (args.wait_reward_clip is not None or args.wait_fail_threshold is not None
            or args.reward_preset != "raw"):
        _rc_kw = dict(wait_reward_clip=args.wait_reward_clip,
                      wait_fail_threshold=args.wait_fail_threshold,
                      wait_fail_penalty=args.wait_fail_penalty)
        kw["reward_calc"] = (RewardCalculator.seimbang4x(**_rc_kw)
                             if args.reward_preset == "seimbang4x" else RewardCalculator(**_rc_kw))
    if args.mode == "pretrain_specialist":
        kw["stream_select"] = args.stream_select
    else:
        tag0 = _specialist_tag(0, args.specialist0_tag)
        tag1 = _specialist_tag(1, args.specialist1_tag)
        r0 = _load_r_star(tag0, args.specialist_seed)
        r1 = _load_r_star(tag1, args.specialist_seed)
        kw["specialist_r_star"] = [r0, r1]
        print(f"[{elapsed()}]   r_star dimuat: wait={r0:.4f} gini={r1:.4f}", flush=True)

    tr = MasterHybridPPOTrainer(**kw)
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


print(f"[{elapsed()}] === PELATIHAN Master-Hybrid PPO ({args.n_train_seed} seed) ===", flush=True)
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
                                  dataset=DATASET, actor_kw=ACTOR_KW)),
                 f"{TAG_ARM}_eval_results.json")
print(f"[{elapsed()}] === SEMUA SELESAI ({TAG_ARM}) ===", flush=True)
