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
import sys, os, time, json, argparse, re as _re
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
p.add_argument("--stream-select", type=int, default=None, choices=[0, 1, 2],
              help="0=wait, 1=gini. 2=acceptance HANYA sah bila --pure-streams.")
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--dataset", type=str, default="4x")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--specialist0-tag", type=str, default=None)
p.add_argument("--specialist1-tag", type=str, default=None)
p.add_argument("--initial-trust", type=float, default=None,
              help="Trust AWAL semua pengguna (baku None -> INIT_TRUST=0.5 bawaan). "
                   "Trust TETAP DINAMIS sesudahnya -- ini menguji sensitivitas thd titik "
                   "awal, BUKAN membekukan dinamika performatif. Uji generalisasi "
                   "LINGKUNGAN. Harus di (0,1). Dipakai saat LATIH maupun UJI.")
p.add_argument("--gamma", type=float, default=0.99,
              help="Faktor diskon PPO/GAE (baku 0.99). Uji sensitivitas ALGORITMA "
                   "(bukan lingkungan). CATATAN: efeknya teredam oleh `max_step_gap=4` "
                   "yang memutus rantai bootstrap antar-transisi berjauhan waktu, jadi "
                   "horizon efektif sudah pendek terlepas dari gamma.")
p.add_argument("--gamma-est-wait", type=float, default=None,
              help="Sensitivitas P_rec pengguna thd estimasi waktu tunggu (baku None -> "
                   "GAMMA_DEFAULT=0.05590271 bawaan `user.py`). BUKAN `--gamma` di atas -- "
                   "itu diskon PPO/GAE (algoritma), ini parameter LINGKUNGAN/perilaku "
                   "pengguna. `user.py::GAMMA_SWEEP` = (0.02795135, 0.05590271, "
                   "0.11180542) = x0.5/x1/x2 titik sapuan baku. Dipakai saat LATIH "
                   "maupun UJI.")
p.add_argument("--beta-denom", type=str, default=None, choices=["r_star", "ret_std"],
              help="Penyebut gap-ratio DGR. 'r_star' = |r_star| (perilaku lama). "
                   "'ret_std' = simpangan baku return -- WAJIB bila ada aliran yang "
                   "reratanya mendekati nol (mis. delta-gini, yang reratanya ~0 secara "
                   "STRUKTURAL krn deret teleskopik). Baku: 'ret_std' bila --pure-streams, "
                   "'r_star' bila tidak (menjaga hasil lama tetap sah).")
p.add_argument("--specialist2-tag", type=str, default=None,
              help="Hanya dipakai bila --pure-streams (aliran ke-3 = acceptance).")
p.add_argument("--pure-streams", action="store_true",
              help="MODE ALIRAN-MURNI: satu suku per aliran -- 0=wait(+CFR), 1=gini, "
                   "2=acceptance. `prox` & `flock` DIBUANG. Menghapus kebutuhan "
                   "kalibrasi `alpha` sepenuhnya: dgn satu suku per aliran, bobotnya "
                   "merosot jadi penskala seragam yang lenyap BAIK di normalisasi "
                   "advantage per-aliran MAUPUN di gap-ratio DGR -- seluruh "
                   "penyeimbangan objektif ditangani `beta` yang dinamis. "
                   "Butuh 3 tahap spesialis (--stream-select 0,1,2) sebelum --mode dgr. "
                   "WAJIB --alpha-accept != 0 (aliran 2 kosong tanpa itu).")
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
p.add_argument("--pref-hist-k", type=int, default=None,
              help="Override panjang jendela riwayat P (baku None -> ikut PDQN_HIST_K=10 "
                   "global, TAK berubah). Mis. 5 -- ablasi apakah jendela lebih pendek "
                   "membantu optimasi `pref_lstm` (bukan soal padding, itu sudah ditangani "
                   "benar via pack_padded_sequence -- ini murni soal panjang konteks).")
p.add_argument("--alpha-accept", type=float, default=0.0,
              help="Bobot suku kepatuhan SIMETRIS (+alpha patuh / -alpha tolak), BAKU 0.0 "
                   "= MATI (perilaku lama). TEMUAN yg mendasarinya: `wait_reward` hanya "
                   "aktif bila patuh & TAK ADA hukuman bila ditolak, `decision_reward`(Prox) "
                   "sama saja patuh/tidak -- artinya KEPATUHAN BUKAN sasaran reward sama "
                   "sekali, shg tak ada jalur bagi Modul P utk mengubah 'paham preferensi' "
                   "jadi 'kurangi penolakan demi pemerataan'.")
p.add_argument("--accept-stream", type=str, default="global", choices=["global", "individual"],
              help="Aliran tujuan suku kepatuhan. BAKU 'global' (bersama gini+flock) -- "
                   "'individual' (bersama wait+prox) adalah konfigurasi yg SUDAH "
                   "terdiagnosis bermasalah pd Kandidat A (2026-08-21): acceptance +-1 "
                   "SEGERA menumpuk dgn wait yg kecil & tertunda -> r_bar aliran meledak "
                   "10-30x. Sediakan hanya utk pembanding bila ingin mereplikasi diagnosis.")
p.add_argument("--critic-pref", action="store_true",
              help="Pakai `MasterHybridPPOCritic` (menerima pref_hist, param P TERPISAH "
                   "dari aktor) menggantikan `MasterPurePPOCritic` (SELALU buta P) -- uji "
                   "hipotesis kritik jadi sumber variansi advantage tambahan khusus utk "
                   "keputusan berbasis P. WAJIB --pref-feature-mode (P tak berarti kalau "
                   "aktor sendiri tak memakainya).")
p.add_argument("--critic-pref-gate-init", type=float, default=0.1,
              help="Gerbang P KRITIK, TERPISAH dari --pref-gate-init aktor (baku 0.1, "
                   "BUKAN 0.0 -- kalau ikut default aktor 0.0, kritik terjebak deadlock "
                   "gradien sama spt bug lama di aktor). Sengaja terpisah supaya varian "
                   "'Attn-saja + kritik-ber-P' bisa menguji kritik BER-P AKTIF sementara "
                   "gerbang P AKTOR tetap 0 (P aktor sengaja inert).")
p.add_argument("--no-station-attn", action="store_true",
              help="Matikan `SmallStationAttention` sepenuhnya (2026-08-30, ABLASI) -- "
                   "utk mengisolasi kontribusi Modul P TANPA atensi antar-stasiun (varian "
                   "'P saja'), memisahkannya dari kontribusi station attention (varian "
                   "'attention saja' = BAKU, tanpa --pref-feature-mode).")
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
    assert args.stream_select is not None, "--mode pretrain_specialist WAJIB --stream-select"
assert not (args.stream_select == 2 and not args.pure_streams), (
    "--stream-select 2 (acceptance) hanya ada di mode --pure-streams")
assert not (args.pure_streams and args.alpha_accept == 0.0), (
    "--pure-streams WAJIB --alpha-accept != 0 -- aliran 2 (acceptance) akan kosong "
    "sepenuhnya tanpa itu. Nilainya sendiri tak berpengaruh (penskala seragam satu-suku "
    "lenyap di normalisasi advantage & gap-ratio); pakai 1.0 saja.")
assert not (args.pure_streams and args.accept_stream != "global"), (
    "--accept-stream tak berlaku di --pure-streams (acceptance selalu punya aliran sendiri)")
# Baku bergantung mode -- DINYATAKAN, bukan diam-diam (lihat cetakan di bawah).
if args.beta_denom is None:
    args.beta_denom = "ret_std" if args.pure_streams else "r_star"

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
DATASET = _DATASET_4X if args.dataset == "4x" else os.path.join(common.ROOT, args.dataset)
if args.dataset != "4x":
    assert args.horizon != "30d", "--dataset kustom butuh --horizon eksplisit"

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
# BUG DITEMUKAN & DIPERBAIKI 2026-08-31: tag SEBELUMNYA tak menyertakan nama dataset
# sama sekali -- `--dataset scenario_dataset_klaster12_6x_90d.json --horizon 90d`
# menghasilkan TAG_ARM PERSIS SAMA dgn baku `..._4x_90d.json --horizon 90d`
# (keduanya cuma "_90d"), sehingga run beban 6x akan MENIMPA checkpoint/
# training_results.json rezim 4x yg sudah ada TANPA peringatan apa pun. Rezim beban
# adalah properti SUBSTRAT (Tahap 1, `common.py::SUBSTRAT["rezim_operasi_load_
# multiplier"]=4.0, dibekukan), bukan sekadar path berkas -- wajib tercermin di tag
# spt initial_trust/gamma. Dataset "4x" (kata kunci BAKU) dan file eksplisit yg
# namanya memuat "_4x" TIDAK diberi akhiran (kompatibel mundur dgn seluruh tag lama);
# rezim lain (mis. "_6x_") WAJIB diberi akhiran.
_m_beban = _re.search(r"_(\d+(?:\.\d+)?)x(?:_|\.|$)", os.path.basename(args.dataset))
_beban_suffix = ("" if args.dataset == "4x" or (_m_beban and _m_beban.group(1) == "4")
                else (f"_load{_m_beban.group(1)}x" if _m_beban else "_dsCUSTOM"))
if _beban_suffix:
    print(f"[{elapsed()}] Dataset non-baku terdeteksi -> akhiran tag '{_beban_suffix}' "
         f"ditambahkan (mencegah tabrakan dgn hasil rezim 4x).", flush=True)
_clip_suffix = "" if args.wait_reward_clip is None else f"_clip{args.wait_reward_clip:g}"
_fail_suffix = ("" if args.wait_fail_threshold is None
               else f"_cwtfail{args.wait_fail_threshold:g}pen{args.wait_fail_penalty:g}")
_rw_suffix = "" if args.reward_preset == "raw" else f"_{args.reward_preset}"
assert not (args.pref_pair_outcome and not args.pref_feature_mode), (
    "--pref-pair-outcome WAJIB disertai --pref-feature-mode (blok hasil tak bermakna "
    "di mode one-hot identitas)")
assert not (args.critic_pref and not args.pref_feature_mode), (
    "--critic-pref WAJIB disertai --pref-feature-mode (P kritik tak berarti kalau "
    "aktor sendiri tak memakainya)")
from marl_spklu.rl.master_paper_obs import (STATION_FEAT_DIM_MASTER,
                                            STATION_FEAT_DIM_MASTER_EV)
ACTOR_KW = dict(ACTOR_KW_BASE, pref_feature_mode=args.pref_feature_mode,
                pref_pair_outcome=args.pref_pair_outcome,
                pref_gate_init=args.pref_gate_init,
                use_station_attn=not args.no_station_attn,
                pref_hist_k=args.pref_hist_k,
                station_feat_dim=(STATION_FEAT_DIM_MASTER_EV if args.ev_obs
                                  else STATION_FEAT_DIM_MASTER))
_pref_suffix = (("_preffeat" if args.pref_feature_mode else "")
                + ("_pairout" if args.pref_pair_outcome else "")
                + ("" if args.pref_gate_init == 0.0 else f"_pg{args.pref_gate_init:g}")
                + ("_evobs" if args.ev_obs else "")
                + ("_noattn" if args.no_station_attn else "")
                + ("" if args.pref_hist_k is None else f"_histK{args.pref_hist_k}")
                + ("_critpref" if args.critic_pref else "")
                + ("_pure3" if args.pure_streams else "")
                + ("" if args.initial_trust is None else f"_it{args.initial_trust:g}")
                + ("" if args.gamma == 0.99 else f"_g{args.gamma:g}")
                + ("" if args.gamma_est_wait is None else f"_gw{args.gamma_est_wait:g}")
                + ("" if args.alpha_accept == 0.0 or args.pure_streams else
                   f"_acc{args.alpha_accept:g}" +
                   ("" if args.accept_stream == "global" else "S1")))
_clip_suffix = _beban_suffix + _clip_suffix + _fail_suffix + _rw_suffix + _pref_suffix
if args.mode == "pretrain_specialist":
    STREAM_NAME = {0: "wait", 1: "gini", 2: "accept"}[args.stream_select]
    TAG_ARM = f"master_hybrid_ppo_specialist{args.stream_select}_{STREAM_NAME}{_horizon_suffix}{_clip_suffix}"
else:
    TAG_ARM = f"master_hybrid_ppo_dgr{_horizon_suffix}{_clip_suffix}"

print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} mode={args.mode} stream_select={args.stream_select}",
     flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"n_train_seed={args.n_train_seed}", flush=True)
if args.mode == "dgr":
    print(f"[{elapsed()}] DGR: penyebut gap-ratio = {args.beta_denom}"
         + ("  (std return -- aman utk aliran ber-rerata ~0)" if args.beta_denom == "ret_std"
            else "  (|r_star| -- perilaku lama)"), flush=True)


def _specialist_tag(stream: int, explicit: str):
    if explicit:
        return explicit
    # Mode aliran-murni menambah aliran ke-3 (acceptance). Nama aliran 0/1 SENGAJA
    # dipertahankan ("wait"/"gini") supaya tag mode lama tetap cocok apa adanya.
    name = {0: "wait", 1: "gini", 2: "accept"}[stream]
    return f"master_hybrid_ppo_specialist{stream}_{name}{_horizon_suffix}{_clip_suffix}"


def _load_r_star(tag: str, seed: int) -> float:
    path = os.path.join(common.OUTDIR, f"{tag}_r_star_seed{seed}.json")
    assert os.path.exists(path), f"r_star spesialis tak ditemukan: {path}"
    return json.load(open(path, encoding="utf-8"))["r_star"]


def train_one(seed):
    from marl_spklu.rl.rollout import STREAM_INDIVIDUAL, STREAM_GLOBAL
    kw = dict(dataset_path=DATASET, mode=args.mode, rollout_steps=args.rollout_steps,
             seed=seed, verbose=False, actor_kwargs=ACTOR_KW, critic_pref=args.critic_pref,
             critic_pref_gate_init=args.critic_pref_gate_init,
             accept_stream=(STREAM_GLOBAL if args.accept_stream == "global"
                            else STREAM_INDIVIDUAL),
             pure_streams=args.pure_streams, beta_denom=args.beta_denom,
             gamma=args.gamma)
    if (args.wait_reward_clip is not None or args.wait_fail_threshold is not None
            or args.reward_preset != "raw" or args.alpha_accept != 0.0):
        _rc_kw = dict(wait_reward_clip=args.wait_reward_clip,
                      wait_fail_threshold=args.wait_fail_threshold,
                      wait_fail_penalty=args.wait_fail_penalty,
                      alpha_accept=args.alpha_accept)
        kw["reward_calc"] = (RewardCalculator.seimbang4x(**_rc_kw)
                             if args.reward_preset == "seimbang4x" else RewardCalculator(**_rc_kw))
    if args.mode == "pretrain_specialist":
        kw["stream_select"] = args.stream_select
    else:
        _explicit = [args.specialist0_tag, args.specialist1_tag, args.specialist2_tag]
        _nama = ["wait", "gini", "accept"]
        _n = 3 if args.pure_streams else 2
        r_list = [_load_r_star(_specialist_tag(s, _explicit[s]), args.specialist_seed)
                  for s in range(_n)]
        kw["specialist_r_star"] = r_list
        print(f"[{elapsed()}]   r_star dimuat: "
             + " ".join(f"{_nama[s]}={r_list[s]:.4f}" for s in range(_n)), flush=True)

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

# `initial_trust` menambal `User.__init__`, jadi HARUS aktif setiap kali simulator
# dibuat -- bukan hanya di awal. Trainer memanggil `_fresh_sim()` saat konstruksi DAN
# di tiap batas horizon (`_carry_forward`), sehingga konteks dibentangkan menutupi
# SELURUH loop pelatihan, bukan dipasang per-pemanggilan.
import contextlib
_env_stack = contextlib.ExitStack()
if args.initial_trust is not None:
    from marl_spklu.experiments.ablations import initial_trust as _initial_trust
    _env_stack.enter_context(_initial_trust(args.initial_trust))
    print(f"[{elapsed()}] Lingkungan: initial_trust={args.initial_trust:g} "
         f"(trust tetap DINAMIS sesudahnya)", flush=True)
if args.gamma_est_wait is not None:
    from marl_spklu.experiments.ablations import gamma_est_wait as _gamma_est_wait
    _env_stack.enter_context(_gamma_est_wait(args.gamma_est_wait))
    print(f"[{elapsed()}] Lingkungan: gamma_est_wait={args.gamma_est_wait:g} "
         f"(sensitivitas P_rec thd estimasi waktu tunggu, BUKAN diskon PPO/GAE)",
         flush=True)
if args.gamma != 0.99:
    print(f"[{elapsed()}] Algoritma: gamma={args.gamma:g} (baku 0.99)", flush=True)

with _env_stack:
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
