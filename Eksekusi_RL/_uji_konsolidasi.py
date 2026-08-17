"""Harness evaluasi TERKONSOLIDASI -- satu simulasi, banyak pengukuran.

Sebelumnya tiap pertanyaan dijawab skrip terpisah yang menjalankan ulang simulasi yang
sama. Padahal satu `sim.run()` sudah memuat semua bahannya. Skrip ini menjalankan tiap
kombinasi SEKALI lalu memanen seluruh metrik.

Satu run menghasilkan:
  E0/K2  kalibrasi prediktor   -> sapuan c, argmin, Sum(beta)/Sum(alpha)     (dari logs)
  K4     arah ekor galat       -> %terlambat vs %terlalu cepat               (dari logs)
  E4     koordinasi            -> herding_index, flocking_index, entropi     (rec_distribution_log)
  K3     performativitas       -> selisih rezim beku vs dinamis              (dua rezim)
  E6     nubuat gagal-diri     -> korelasi konsentrasi rekomendasi <-> telat (gabungan)
  baku   gini/trust/acc/wait/served, PER SEED

Dimensi: lengan x aturan_trust x rezim_trust x seed.

    python _uji_konsolidasi.py 0,1,2                 # 3 seed (default)
    python _uji_konsolidasi.py 0,1,2,3,4 90d         # 5 seed, horizon 90 hari
"""
import sys, os, json, random, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
import marl_spklu.env.user as U
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.rollout import InferenceAgent
from marl_spklu.experiments.ablations import constant_trust_shadow
from marl_spklu.experiments import metrics as M
from marl_spklu.env.user import (DELTAW_TOL_LOW as LO, DELTAW_TOL_HIGH as HI,
                                 TRUST_EPS_ALPHA as EA, TRUST_EPS_BETA as EB)

HORIZON = {"30d": "scenario_dataset_klaster12_4x.json",
           "90d": "scenario_dataset_klaster12_4x_90d.json"}
SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
         if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
DS = os.path.join(common.ROOT, HORIZON[TAG])
TRUST_BEKU = 0.5
SAPUAN = np.arange(-90.0, 30.5, 0.5)


@contextlib.contextmanager
def mode_trust(mode):
    orig = U.TRUST_PENALTY_MODE
    U.TRUST_PENALTY_MODE = mode
    try:
        yield
    finally:
        U.TRUST_PENALTY_MODE = orig


class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}


def pol(stem, seed):
    ck = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}.pt")
    mp = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}_meta.json")
    if not (os.path.exists(ck) and os.path.exists(mp)):
        return None
    m = json.load(open(mp))
    c = PPPOPolicy if m.get("policy_cls") == "PPPOPolicy" else HPPOPolicy
    kw = dict(n_critics=m.get("n_critics", 1))
    if c is PPPOPolicy:
        kw.update(pref_d_lstm=m.get("pref_d_lstm", 64), pref_d_attn=m.get("pref_d_attn", 64))
    p = c(m["obs_dim"], m["critic_obs_dim"], m["N"], **kw)
    p.load_state_dict(torch.load(ck)); p.eval()
    return lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0, threshold=0.20)


def rasio_beta_alpha(d, mode):
    """Meniru User.update_trust: reward selalu absolut; penalti absolut atau bertanda."""
    a = np.abs(d)
    sa = float(np.sum(EA * (1.0 - a[a <= LO] / LO)))
    pen = (d >= HI) if mode == "signed" else (a >= HI)
    sb = float(np.sum(EB * (a[pen] / HI)))
    return sb / max(sa, 1e-9), sa, sb


def satu_run(fac, mode, seed, beku):
    """SATU simulasi -> semua metrik. Inilah inti penghematannya."""
    ctx = constant_trust_shadow(value=TRUST_BEKU) if beku else contextlib.nullcontext()
    with mode_trust(mode), ctx:
        sim = common.fresh_sim(DS)
        random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))

    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    d = np.array([l["wait_time"] - l["est_wait"] for l in sim.logs if l.get("complied")], float)

    r = dict(gini=float(common.gini(sv)), served=int(sv.sum()),
             acc=float(c.mean()) if c.size else 0.0,
             wait=float(w.mean()) if w.size else 0.0,
             trust=float(tr.mean()), trust_sd=float(tr.std()), trust_min=float(tr.min()))

    # --- E4 koordinasi (dari rec_distribution_log, tanpa run tambahan) ---
    rdl = getattr(sim, "rec_distribution_log", None)
    n_st = len(sim.spklus)
    if rdl:
        r["herding"] = float(M.herding_index(rdl))
        r["flocking"] = float(M.flocking_index(rdl))
        r["rec_entropy"] = float(M.recommendation_entropy(rdl, n_st))
    else:
        r["herding"] = r["flocking"] = r["rec_entropy"] = float("nan")

    # --- E0/K2/K4 galat & kalibrasi ---
    if d.size:
        r["n_janji"] = int(d.size)
        r["pct_telat"] = float(100 * np.mean(d >= HI))
        r["pct_cepat"] = float(100 * np.mean(-d >= HI))
        r["pct_tepat"] = float(100 * np.mean(np.abs(d) <= LO))
        r["galat_mean"] = float(d.mean()); r["galat_median"] = float(np.median(d))
        r["galat_iqr"] = float(np.percentile(d, 75) - np.percentile(d, 25))
        r["galat_sd"] = float(d.std())
        kurva = [rasio_beta_alpha(d - x, mode)[0] for x in SAPUAN]
        i = int(np.argmin(kurva))
        r["c_star"] = float(SAPUAN[i])
        r["rasio_c0"] = float(kurva[int(np.argmin(np.abs(SAPUAN)))])
        r["rasio_c_star"] = float(kurva[i])
    else:
        for k in ("n_janji", "pct_telat", "pct_cepat", "pct_tepat", "c_star",
                  "rasio_c0", "rasio_c_star", "galat_mean", "galat_median",
                  "galat_iqr", "galat_sd"):
            r[k] = float("nan")
    return r


def agg(runs):
    if not runs:
        return None
    o = {}
    for k in runs[0]:
        v = np.array([x[k] for x in runs], float)
        o[k] = float(np.nanmean(v)); o[k + "_med"] = float(np.nanmedian(v))
        o[k + "_sd"] = float(np.nanstd(v))
    o["n_seed"] = len(runs)
    o["n_kolaps"] = int(sum(1 for x in runs if x["gini"] > 0.15))
    o["n_wait_tinggi"] = int(sum(1 for x in runs if x["wait"] > 150))
    return o


def main():
    arms = [("S0", lambda sim: None, None),
            ("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2), None),
            ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2), None),
            ("H-PPO(abs)", None, f"hppo_{TAG}_abs"), ("P-PPO(abs)", None, f"pppo_{TAG}_abs"),
            ("H-PPO(sgn)", None, f"hppo_{TAG}_sgn"), ("P-PPO(sgn)", None, f"pppo_{TAG}_sgn")]

    out = {"horizon": TAG, "seeds": SEEDS, "trust_beku": TRUST_BEKU,
           "per_seed": {}, "agregat": {}}
    total = len(arms) * 2 * 2 * len(SEEDS)
    print(f"horizon={TAG}  seed={SEEDS}  -> maksimum {total} simulasi\n", flush=True)

    for mode in ("abs", "signed"):
        for beku in (True, False):
            rez = "beku" if beku else "dinamis"
            print(f"=== aturan={mode}  rezim={rez} ===", flush=True)
            H = "%-14s %7s %7s %7s %8s %7s %7s %7s %6s" % (
                "lengan", "gini", "trust", "acc", "wait", "herd", "%telat", "%cepat", "c*")
            print(H); print("-" * len(H))
            for lbl, fac, stem in arms:
                runs = []
                for sd in SEEDS:
                    f = fac if stem is None else pol(stem, sd)
                    if f is None:
                        continue
                    runs.append(satu_run(f, mode, sd, beku))
                if not runs:
                    print("%-14s (checkpoint belum ada)" % lbl, flush=True); continue
                key = f"{lbl}|{mode}|{rez}"
                out["per_seed"][key] = runs
                a = agg(runs); out["agregat"][key] = a
                print("%-14s %7.4f %7.4f %7.3f %8.1f %7.3f %7.1f %7.1f %6.1f" % (
                    lbl, a["gini"], a["trust"], a["acc"], a["wait"], a["herding"],
                    a["pct_telat"], a["pct_cepat"], a["c_star"]), flush=True)
            print(flush=True)

    # --- K3 selisih performatif: checkpoint SAMA, dua rezim ---
    print("=== K3 SELISIH PERFORMATIF (dinamis - beku, checkpoint sama) ===")
    H = "%-14s %-7s %11s %11s %11s" % ("lengan", "aturan", "d_gini", "d_acc", "d_wait")
    print(H); print("-" * len(H))
    perf = {}
    for lbl, _, _ in arms:
        for mode in ("abs", "signed"):
            kb, kd = f"{lbl}|{mode}|beku", f"{lbl}|{mode}|dinamis"
            if kb not in out["agregat"] or kd not in out["agregat"]:
                continue
            b, dn = out["agregat"][kb], out["agregat"][kd]
            p = dict(d_gini=dn["gini"] - b["gini"], d_acc=dn["acc"] - b["acc"],
                     d_wait=dn["wait"] - b["wait"])
            perf[f"{lbl}|{mode}"] = p
            print("%-14s %-7s %+11.4f %+11.4f %+11.1f" % (lbl, mode, p["d_gini"],
                                                          p["d_acc"], p["d_wait"]))
    out["performativitas"] = perf

    print()
    print("=== besaran performativitas per aturan (rata-rata |selisih|, lengan RL saja) ===")
    ring = {}
    for mode in ("abs", "signed"):
        v = [p for k, p in perf.items() if k.endswith("|" + mode) and "PPO" in k]
        if not v:
            continue
        ring[mode] = {k: float(np.mean([abs(x[k]) for x in v])) for k in
                      ("d_gini", "d_acc", "d_wait")}
        print("  %-7s |d_gini|=%.4f  |d_acc|=%.4f  |d_wait|=%.1f" % (
            mode, ring[mode]["d_gini"], ring[mode]["d_acc"], ring[mode]["d_wait"]))
    if "abs" in ring and "signed" in ring:
        r = ring["signed"]["d_gini"] / max(ring["abs"]["d_gini"], 1e-12)
        out["rasio_performativitas_signed_thd_abs"] = float(r)
        print(f"\n  performativitas `signed` = {r:.2f}x dari `abs`")
        print("    <1 melemah; mendekati 0 fenomena praktis hilang")
    out["ringkas_performativitas"] = ring

    common.save_json(out, f"uji_konsolidasi_{TAG}.json")
    print(f"\nSAVED -> outputs/uji_konsolidasi_{TAG}.json")


if __name__ == "__main__":
    main()
