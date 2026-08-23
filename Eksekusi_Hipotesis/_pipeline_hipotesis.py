"""Pipeline pengujian hipotesis H1-H6 (`draft tesis/Hipotesis_Penelitian.md`).

Turunan `Eksekusi_RL/_run_master_ev_ppo_pipeline.py` dengan EMPAT perbedaan pokok:

1. `--initial-trust` / `--constant-trust` / `--constant-trust-shadow` -- pengaturan
   tingkat kepercayaan yang TIDAK ADA di pipeline lama, sehingga seluruh eksperimen
   sebelumnya berjalan di satu tingkat kepercayaan bawaan saja. Ini penghambat E0:
   tanpa ini H2, E2, E4, E7 dan separuh syarat pembatalan hipotesis mustahil dijalankan.

2. `--tag` EKSPLISIT, bukan diturunkan dari tumpukan akhiran pengaturan. Tag lama
   (`master_ev_ppo_pref_feat_nohist_acc1_vwf_K3`) diturunkan otomatis, sehingga dua
   konfigurasi berbeda bisa menghasilkan nama sama dan saling menimpa diam-diam --
   persis insiden 21-23 Agustus 2026 yang membuat model eksperimen utama hilang.
   Di sini tag ditulis manusia, dan tanggal dibubuhkan otomatis.

3. Evaluasi mencatat EMPAT metrik (gini/penerimaan/tunggu/kepercayaan) plus sebaran
   kepercayaan akhir, bukan hanya Gini. E4 (kurva H2) membutuhkan sebaran itu, dan
   pipeline lama tidak pernah menyimpannya.

4. Diagnosis akurasi-janji DIHAPUS -- masalah itu sudah dikeluarkan dari cakupan
   (lihat "Batas Cakupan" di Hipotesis_Penelitian.md). Menghemat satu putaran evaluasi
   penuh per lengan.

Contoh:
    python _pipeline_hipotesis.py --tag h6b_utama \\
        --pref --pref-feature-mode --no-hist \\
        --alpha-accept 1.0 --n-critics 3 --forecaster vwf \\
        --initial-trust 0.5 --n-train-seed 5 --n-eval-seed 10 --n-updates 300
"""
import sys, os, time, json, random, argparse, contextlib, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Eksekusi_RL"))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import common
from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOTrainer, MasterEVPPOPolicy,
                                                MasterEVPPOPrefPolicy, MasterEVPPOInferenceAgent)
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.experiments.ablations import (initial_trust, constant_trust,
                                              constant_trust_shadow)
from scipy.stats import wilcoxon

# Hasil folder ini TERPISAH dari Eksekusi_RL/outputs -- `common.save_json` memakai
# `common.OUTDIR`, jadi ditimpa di sini supaya tak ada satu pun berkas eksplorasi lama
# yang bisa tertimpa oleh eksperimen hipotesis, atau sebaliknya.
OUTDIR = os.path.join(_HERE, "outputs")
os.makedirs(OUTDIR, exist_ok=True)
common.OUTDIR = OUTDIR

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"


p = argparse.ArgumentParser()
p.add_argument("--tag", type=str, required=True,
               help="nama lengan, ditulis manusia (mis. 'h6b_utama'). Tingkat kepercayaan "
                    "dan tanggal dibubuhkan otomatis -> 'h6b_utama__it05_20260824'.")
p.add_argument("--n-train-seed", type=int, default=5)
p.add_argument("--n-eval-seed", type=int, default=10)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--k", type=int, default=3)
p.add_argument("--n-critics", type=int, default=1)
p.add_argument("--beta-mode", type=str, default="fixed", choices=["fixed", "gap_ratio"])
p.add_argument("--beta-sigma", type=float, default=0.1)
p.add_argument("--pref", action="store_true")
p.add_argument("--pref-feature-mode", action="store_true")
p.add_argument("--no-hist", action="store_true")
p.add_argument("--pref-hist-k", type=int, default=None)
p.add_argument("--dataset", type=str, default="4x")
p.add_argument("--horizon", type=str, default="30d")
p.add_argument("--alpha-trust", type=float, default=0.0)
p.add_argument("--alpha-accept", type=float, default=0.0)
p.add_argument("--alpha-equity", type=float, default=0.0)
p.add_argument("--reward-preset", type=str, default="seimbang4x", choices=["raw", "seimbang4x"])
p.add_argument("--forecaster", type=str, default="vwf", choices=["formula", "vwf"])

# --- pengaturan kepercayaan: INTI dari E0 -------------------------------------
p.add_argument("--initial-trust", type=float, default=None,
               help="Kepercayaan AWAL semua pengguna (0<v<1). Dinamikanya TETAP JALAN "
                    "sesudahnya. Untuk H2/E2/E4: pakai 0.3 / 0.5 / 0.7.")
p.add_argument("--constant-trust", type=float, default=None,
               help="Kepercayaan DIBEKUKAN sepanjang simulasi (update_trust jadi no-op). "
                    "Untuk melatih lengan 'statis' di E7.")
p.add_argument("--constant-trust-shadow", type=float, default=None,
               help="Kepercayaan tetap DIPERBARUI, tapi nilai yang DIPAKAI MENGAMBIL "
                    "KEPUTUSAN dibekukan. Dipakai saat MENGEVALUASI lengan statis di E7 "
                    "-- memperlihatkan apa yang AKAN terjadi pada kepercayaan seandainya "
                    "lingkaran umpan baliknya tidak diputus.")
p.add_argument("--overwrite", action="store_true",
               help="Izinkan menimpa hasil yang n_updates-nya lebih besar. Tanpa ini "
                    "pipeline berhenti sebelum menulis (model tetap tersimpan).")
args = p.parse_args()

assert not (args.pref_feature_mode and not args.pref), "--pref-feature-mode butuh --pref"
_trust_flags = [args.initial_trust, args.constant_trust, args.constant_trust_shadow]
assert sum(v is not None for v in _trust_flags) <= 1, (
    "Pilih SATU saja di antara --initial-trust / --constant-trust / --constant-trust-shadow.")
for _v, _n in zip(_trust_flags, ("--initial-trust", "--constant-trust", "--constant-trust-shadow")):
    assert _v is None or 0.0 < _v < 1.0, f"{_n} harus di antara 0 dan 1 (eksklusif), dapat {_v}"

POLICY_CLS = MasterEVPPOPrefPolicy if args.pref else MasterEVPPOPolicy
POLICY_KW = dict(pref_feature_mode=args.pref_feature_mode) if args.pref else dict()
if args.no_hist:
    POLICY_KW["use_hist"] = False

# --- dataset ------------------------------------------------------------------
if args.dataset == "4x":
    DATASET = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
elif args.dataset == "1x":
    DATASET = common.DATASET_KANONIK
    print(f"[{elapsed()}] !! regime 1x -- hanya sah untuk E7 (dua tingkat keramaian)", flush=True)
else:
    DATASET = args.dataset
    if not os.path.isabs(DATASET) and not os.path.exists(DATASET):
        DATASET = os.path.join(common.ROOT, DATASET)
    assert args.horizon != "30d", (
        "--dataset kustom diberikan tapi --horizon masih '30d'. Ini jebakan yang sudah "
        "pernah terjadi: dataset 90 hari akan menimpa diam-diam model 30 hari. "
        "Sertakan --horizon 90d.")
assert os.path.exists(DATASET), f"dataset tak ditemukan: {DATASET}"


# --- penamaan: eksplisit + bertanggal, TIDAK diturunkan dari pengaturan --------
def _trust_tag():
    for v, pre in ((args.initial_trust, "it"), (args.constant_trust, "ct"),
                   (args.constant_trust_shadow, "cs")):
        if v is not None:
            return f"__{pre}{v:g}".replace(".", "")
    return "__itbaku"


STAMP = datetime.date.today().strftime("%Y%m%d")
TAG = f"{args.tag}{_trust_tag()}" + ("" if args.horizon == "30d" else f"_{args.horizon}")
TAG_STAMPED = f"{TAG}_{STAMP}"


def trust_ctx():
    """Konteks kepercayaan. HARUS membungkus pelatihan MAUPUN evaluasi: `initial_trust`
    dan `constant_trust` menambal `User.__init__`, yang hanya berpengaruh bila terpasang
    SEBELUM `Simulator.load_from_dataset()` -- yaitu sebelum tiap `common.fresh_sim()`."""
    if args.initial_trust is not None:
        return initial_trust(value=args.initial_trust)
    if args.constant_trust is not None:
        return constant_trust(value=args.constant_trust)
    if args.constant_trust_shadow is not None:
        return constant_trust_shadow(value=args.constant_trust_shadow)
    return contextlib.nullcontext()


def make_forecaster():
    from marl_spklu.rl.forecaster import FormulaForecaster, VirtualWaitForecaster
    return VirtualWaitForecaster() if args.forecaster == "vwf" else FormulaForecaster()


def make_reward_calc():
    kw = dict(alpha_trust=args.alpha_trust, alpha_accept=args.alpha_accept,
              alpha_equity=args.alpha_equity)
    return (RewardCalculator.seimbang4x(**kw) if args.reward_preset == "seimbang4x"
            else RewardCalculator(**kw))


def train_one(seed):
    tr = MasterEVPPOTrainer(DATASET, rollout_steps=args.rollout_steps, seed=seed, verbose=False,
                            reward_calc=make_reward_calc(), k=args.k, n_critics=args.n_critics,
                            policy_cls=POLICY_CLS, policy_kw=POLICY_KW,
                            pref_hist_k=args.pref_hist_k, beta_mode=args.beta_mode,
                            beta_sigma=args.beta_sigma)
    policy = tr.train(make_forecaster(), n_updates=args.n_updates)
    ckpt = os.path.join(OUTDIR, f"{TAG_STAMPED}_actor_seed{seed}.pt")
    torch.save(policy.state_dict(), ckpt)
    return dict(seed=seed, ckpt=ckpt, history=tr.history)


def load_policy(ckpt_path, n_spklu):
    pol = POLICY_CLS(n_spklu, n_critics=args.n_critics, **POLICY_KW)
    pol.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    pol.eval()
    return pol


def _metrik(sim):
    """Empat metrik inti + sebaran kepercayaan akhir. Pipeline lama hanya mencatat Gini;
    sebaran kepercayaan adalah yang dibutuhkan E4 untuk menguji H2a (apakah kondisi akhir
    bergantung kondisi awal), dan rata-rata saja tidak cukup untuk itu."""
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    return dict(
        gini=common.gini(sv), served=int(sv.sum()),
        acc=float(c.mean()) if c.size else 0.0,
        wait=float(w.mean()) if w.size else 0.0,
        trust=float(tr.mean()), trust_sd=float(tr.std()),
        trust_p10=float(np.percentile(tr, 10)), trust_p50=float(np.percentile(tr, 50)),
        trust_p90=float(np.percentile(tr, 90)),
        trust_frac_bawah_03=float((tr < 0.3).mean()),
    )


def eval_policy(ckpt_path, n_eval_seed):
    sim0 = common.fresh_sim(DATASET)
    pol = load_policy(ckpt_path, len(sim0.spklus))
    out = []
    for s in range(n_eval_seed):
        sim = common.fresh_sim(DATASET)
        random.seed(s); np.random.seed(s)
        agent = MasterEVPPOInferenceAgent(pol, sim, make_forecaster(), k=args.k,
                                          pref_hist_k=args.pref_hist_k)
        sim.run(max_steps=sim.max_steps, agent=agent)
        out.append(_metrik(sim))
    return out


def eval_baseline(agent_factory, n_eval_seed):
    out = []
    for s in range(n_eval_seed):
        sim = common.fresh_sim(DATASET)
        random.seed(s); np.random.seed(s)
        sim.run(max_steps=sim.max_steps, agent=agent_factory())
        out.append(_metrik(sim))
    return out


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def kolom(rows, key):
    return [r[key] for r in rows]


# =============================================================================
with trust_ctx():
    print(f"[{elapsed()}] tag={TAG_STAMPED}", flush=True)
    print(f"[{elapsed()}] dataset={os.path.basename(DATASET)} horizon={args.horizon} "
          f"pref={args.pref} n_critics={args.n_critics} alpha_accept={args.alpha_accept} "
          f"forecaster={args.forecaster}", flush=True)
    print(f"[{elapsed()}] kepercayaan: initial={args.initial_trust} constant={args.constant_trust} "
          f"shadow={args.constant_trust_shadow}", flush=True)
    print(f"[{elapsed()}] anggaran: {args.n_train_seed} seed x {args.n_updates} pembaruan", flush=True)

    # --- pelatihan (bisa dilanjut bila terputus) ------------------------------
    print(f"[{elapsed()}] === PELATIHAN ===", flush=True)
    results, existing = [], {}
    results_name = f"{TAG_STAMPED}_training_results.json"
    try:
        with open(os.path.join(OUTDIR, results_name), encoding="utf-8") as f:
            for row in json.load(f):
                existing[row["seed"]] = row
        print(f"[{elapsed()}] {len(existing)} seed sudah ada -- dilanjutkan", flush=True)
    except FileNotFoundError:
        pass

    for seed in range(args.n_train_seed):
        if seed in existing:
            print(f"[{elapsed()}]   seed={seed} LEWATI (sudah ada)", flush=True)
            results.append(existing[seed]); continue
        print(f"[{elapsed()}]   seed={seed} mulai", flush=True)
        results.append(train_one(seed))
        common.save_json(results, results_name)
        print(f"[{elapsed()}]   seed={seed} selesai", flush=True)

    # --- evaluasi -------------------------------------------------------------
    print(f"[{elapsed()}] === EVALUASI ({args.n_eval_seed} seed lingkungan) ===", flush=True)
    per_seed = {r["seed"]: eval_policy(r["ckpt"], args.n_eval_seed) for r in results}
    urut = sorted(per_seed, key=lambda sd: np.mean(kolom(per_seed[sd], "gini")))
    seed_median = urut[len(urut) // 2]
    rows = per_seed[seed_median]

    gq = eval_baseline(lambda: GreedyAgent(mode="queue"), args.n_eval_seed)
    gu = eval_baseline(lambda: GreedyAgent(mode="utilization"), args.n_eval_seed)

    g_pol, g_gq, g_gu = kolom(rows, "gini"), kolom(gq, "gini"), kolom(gu, "gini")
    w_gq, w_gu = wilcoxon(g_pol, g_gq), wilcoxon(g_pol, g_gu)
    spread = float(max(np.mean(kolom(v, "gini")) for v in per_seed.values())
                   - min(np.mean(kolom(v, "gini")) for v in per_seed.values()))

    def ringkas(rs):
        return {k: float(np.mean(kolom(rs, k))) for k in rs[0]}

    R, RQ, RU = ringkas(rows), ringkas(gq), ringkas(gu)
    print(f"[{elapsed()}] {'':14s} {'gini':>8s} {'terima':>8s} {'tunggu':>8s} {'percaya':>8s}", flush=True)
    for nama, r in (("kebijakan", R), ("greedy_queue", RQ), ("greedy_util", RU)):
        print(f"[{elapsed()}] {nama:14s} {r['gini']:8.4f} {r['acc']:8.3f} "
              f"{r['wait']:8.1f} {r['trust']:8.3f}", flush=True)
    print(f"[{elapsed()}] gini vs greedy_queue: p={w_gq.pvalue:.4f} d={cohens_d(g_pol, g_gq):+.3f}", flush=True)
    print(f"[{elapsed()}] gini vs greedy_util : p={w_gu.pvalue:.4f} d={cohens_d(g_pol, g_gu):+.3f}", flush=True)
    print(f"[{elapsed()}] sebaran antar-seed  : {spread:.4f}  (seed median={seed_median})", flush=True)

    # --- pengaman penimpaan ---------------------------------------------------
    out_name = f"{TAG_STAMPED}_eval.json"
    out_path = os.path.join(OUTDIR, out_name)
    if os.path.exists(out_path) and not args.overwrite:
        prev = json.load(open(out_path, encoding="utf-8")).get("config", {}).get("n_updates")
        if prev is not None and prev > args.n_updates:
            raise SystemExit(
                f"[{elapsed()}] MENOLAK menimpa {out_name}: run ini {args.n_updates} "
                f"pembaruan, yang tersimpan {prev}. Kemungkinan uji asap menimpa hasil "
                f"serius. Model SUDAH tersimpan aman. Ulangi dengan --overwrite bila memang "
                f"bermaksud menimpa.")

    common.save_json(dict(
        tag=TAG, tag_stamped=TAG_STAMPED, seed_median=seed_median,
        ringkas=dict(kebijakan=R, greedy_queue=RQ, greedy_util=RU),
        gini_policy=g_pol, gini_greedy_queue=g_gq, gini_greedy_util=g_gu,
        p_vs_gq=float(w_gq.pvalue), p_vs_gu=float(w_gu.pvalue),
        cohens_d_vs_gq=cohens_d(g_pol, g_gq), cohens_d_vs_gu=cohens_d(g_pol, g_gu),
        spread_antar_seed=spread,
        per_eval_seed=rows,
        per_train_seed={str(k): ringkas(v) for k, v in per_seed.items()},
        config=dict(vars(args), dataset=DATASET, tanggal=STAMP)),
        out_name)
    print(f"[{elapsed()}] === SELESAI -> outputs/{out_name} ===", flush=True)
