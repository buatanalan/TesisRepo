"""Harness evaluasi TERKONSOLIDASI -- satu simulasi, banyak pengukuran.

Sebelumnya tiap pertanyaan dijawab skrip terpisah yang menjalankan ulang simulasi yang
sama. Padahal satu `sim.run()` sudah memuat semua bahannya.

Satu run menghasilkan:
  E0/K2  kalibrasi prediktor   -> sapuan c, argmin, Sum(beta)/Sum(alpha)
  K4     arah ekor galat       -> %terlambat vs %terlalu cepat
  E4     koordinasi            -> herding_index, flocking_index, entropi rekomendasi
  K3     performativitas       -> selisih rezim beku vs dinamis (checkpoint SAMA)
  baku   gini/trust/acc/wait/served, PER SEED

DIPERLUAS 2026-08-18 -- rerata saja menyembunyikan terlalu banyak:
  * PERSENTIL waktu tunggu, DIPISAH patuh vs menolak. Waktu tunggu pengguna yang MENOLAK
    rekomendasi bukan konsekuensi rekomendasi itu; mencampurnya adalah salah atribusi.
  * DERET HARIAN (Gini kumulatif & harian, wait, acceptance, trust) -- menggantikan skrip
    lintasan terpisah yang menjalankan ulang simulasi utk tiap titik waktu.
  * RINGKASAN PENGGUNA -- apakah beban pemerataan ditanggung merata, atau ditimpakan ke
    sebagian kecil pengguna. Gini stasiun membaik sementara ketimpangan antar-PENGGUNA
    memburuk adalah hasil yang berbeda maknanya.
  * STATISTIK PER STASIUN -- utilisasi & antrean sepanjang waktu, bukan hanya served akhir.

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
from marl_spklu.rl.registry import bangun_kebijakan
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
PERSENTIL = [10, 25, 50, 75, 90, 95, 99]
AMBANG_WAIT = [30, 60, 120, 240]


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
    # Registri tunggal (marl_spklu/rl/registry.py) -- kelas dibaca dari meta; memuat dgn
    # kelas keliru akan gagal pada bentuk bobot.
    p = bangun_kebijakan(m, state_dict=torch.load(ck))
    return lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0, threshold=0.20)


def sebaran(x, pref, ambang=None):
    """Ringkasan distribusional -- rerata saja tak cukup utk besaran berekor."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"{pref}_n": 0}
    o = {f"{pref}_n": int(x.size), f"{pref}_mean": float(x.mean()),
         f"{pref}_sd": float(x.std()), f"{pref}_maks": float(x.max()),
         f"{pref}_min": float(x.min())}
    for p in PERSENTIL:
        o[f"{pref}_p{p}"] = float(np.percentile(x, p))
    # Rasio mean/median: penanda kemiringan. >1,5 berarti rerata digerakkan ekor.
    med = o[f"{pref}_p50"]
    o[f"{pref}_skew_mm"] = float(o[f"{pref}_mean"] / med) if abs(med) > 1e-9 else float("nan")
    for a in (ambang or []):
        o[f"{pref}_frac_gt{a}"] = float(np.mean(x > a))
    return o


def rasio_beta_alpha(d, mode):
    a = np.abs(d)
    sa = float(np.sum(EA * (1.0 - a[a <= LO] / LO)))
    pen = (d >= HI) if mode == "signed" else (a >= HI)
    sb = float(np.sum(EB * (a[pen] / HI)))
    return sb / max(sa, 1e-9)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def satu_run(fac, mode, seed, beku):
    ctx = constant_trust_shadow(value=TRUST_BEKU) if beku else contextlib.nullcontext()
    with mode_trust(mode), ctx:
        sim = common.fresh_sim(DS, rekam_deret=True)
        random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))

    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    d = np.array([l["wait_time"] - l["est_wait"] for l in sim.logs if l.get("complied")], float)

    r = dict(gini=gini(sv), served=int(sv.sum()),
             acc=float(c.mean()) if c.size else 0.0,
             wait=float(w.mean()) if w.size else 0.0,
             trust=float(tr.mean()), trust_sd=float(tr.std()), trust_min=float(tr.min()))

    # --- sebaran: wait DIPISAH patuh vs menolak, galat, trust ---
    r.update(sebaran(w, "w", AMBANG_WAIT))
    r.update(sebaran(w[c] if c.size else [], "wpatuh", AMBANG_WAIT))
    r.update(sebaran(w[~c] if c.size else [], "wtolak", AMBANG_WAIT))
    r.update(sebaran(tr, "tr"))
    r.update(sebaran(d, "galat"))

    # --- koordinasi ---
    rdl = getattr(sim, "rec_distribution_log", None)
    if rdl:
        r["herding"] = float(M.herding_index(rdl))
        r["flocking"] = float(M.flocking_index(rdl))
        r["rec_entropy"] = float(M.recommendation_entropy(rdl, len(sim.spklus)))
    else:
        r["herding"] = r["flocking"] = r["rec_entropy"] = float("nan")

    # --- kalibrasi prediktor & arah ekor ---
    if d.size:
        r["pct_telat"] = float(100 * np.mean(d >= HI))
        r["pct_cepat"] = float(100 * np.mean(-d >= HI))
        r["pct_tepat"] = float(100 * np.mean(np.abs(d) <= LO))
        kurva = [rasio_beta_alpha(d - x, mode) for x in SAPUAN]
        i = int(np.argmin(kurva))
        r["c_star"] = float(SAPUAN[i])
        r["rasio_c0"] = float(kurva[int(np.argmin(np.abs(SAPUAN)))])
        r["rasio_c_star"] = float(kurva[i])
    else:
        for k in ("pct_telat", "pct_cepat", "pct_tepat", "c_star", "rasio_c0",
                  "rasio_c_star"):
            r[k] = float("nan")

    # --- ketimpangan ANTAR-PENGGUNA (berbeda dari ketimpangan antar-stasiun) ---
    pg = [x for x in sim.ringkas_pengguna() if x["n_trip"] > 0]
    if pg:
        wm = np.array([x["wait_mean"] for x in pg], float)
        dm = np.array([x["dorong_mean"] for x in pg], float)
        dm = dm[~np.isnan(dm)]
        r["pengguna_aktif"] = len(pg)
        r["gini_wait_pengguna"] = gini(wm)          # <-- beban tunggu merata antar-orang?
        r["gini_trip_pengguna"] = gini(np.array([x["n_trip"] for x in pg], float))
        r["dorong_mean"] = float(dm.mean()) if dm.size else float("nan")
        r["dorong_p90"] = float(np.percentile(dm, 90)) if dm.size else float("nan")
        r["entropi_spklu_pengguna"] = float(np.mean([x["entropi_spklu"] for x in pg]))
        r["frac_trust_rendah"] = float(np.mean(tr < 0.30))
    r.update(ringkas_pengaruh_janji(sim))
    r["_harian"] = sim.daily_log
    r["_stasiun"] = ringkas_stasiun(sim)
    return r


def ringkas_pengaruh_janji(sim):
    """Apakah JANJI waktu tunggu menggeser keputusan, dan pengguna UNTUNG atau RUGI.

    Kontrafaktualnya `sid_pref` -- stasiun yang akan dipilih pengguna tanpa rekomendasi
    (argmax utilitas pribadi). Dihitung utk semua agen, jadi greedy dan S0 ikut terukur.

    Dipisah PATUH vs MENOLAK: pertanyaannya bukan "apakah pengguna untung rata-rata",
    melainkan "apakah pengguna yang MENURUTI rekomendasi diuntungkan". Mencampur keduanya
    mengaburkan justru bagian yang dinilai.
    """
    D = [e for e in sim.decision_log if e.get("sid_pref")]
    if not D:
        return {}
    B = [e for e in D if e.get("wait_aktual") is not None]
    out = {"jn_n": len(D), "jn_n_hasil": len(B),
           "jn_frac_menyimpang": float(np.mean([not e["pilih_sama_pref"] for e in D]))}

    def _blok(sub, pref):
        if not sub:
            return {}
        o = {}
        ea = np.array([e["untung_janji"] for e in sub
                       if e.get("untung_janji") is not None
                       and np.isfinite(e["untung_janji"])], float)
        ep = np.array([e["untung_expost"] for e in sub
                       if e.get("untung_expost") is not None], float)
        gp = np.array([e["geser_p"] for e in sub if np.isfinite(e.get("geser_p", np.nan))],
                      float)
        if ea.size:
            o[f"{pref}_exante_mean"] = float(ea.mean())
            o[f"{pref}_exante_med"] = float(np.median(ea))
            o[f"{pref}_exante_frac_untung"] = float(np.mean(ea > 0))
            o[f"{pref}_exante_frac_rugi"] = float(np.mean(ea < 0))
        if ep.size:
            o[f"{pref}_expost_mean"] = float(ep.mean())
            o[f"{pref}_expost_med"] = float(np.median(ep))
            o[f"{pref}_expost_frac_untung"] = float(np.mean(ep > 0))
            o[f"{pref}_expost_frac_rugi"] = float(np.mean(ep < 0))
            o[f"{pref}_expost_p10"] = float(np.percentile(ep, 10))
            o[f"{pref}_expost_p90"] = float(np.percentile(ep, 90))
        if gp.size:
            o[f"{pref}_geser_mean"] = float(gp.mean())
            o[f"{pref}_geser_med"] = float(np.median(gp))
        o[f"{pref}_n"] = len(sub)
        return o

    out.update(_blok(B, "jn"))                                    # seluruh keputusan
    out.update(_blok([e for e in B if e["patuh"]], "jnpatuh"))     # yang menuruti
    out.update(_blok([e for e in B if not e["patuh"]], "jntolak")) # yang menolak
    return out


def ringkas_stasiun(sim):
    """Utilisasi & antrean SEPANJANG WAKTU per stasiun -- bukan hanya served akhir."""
    per = {}
    for row in sim.station_log:
        per.setdefault(row["spklu"], {"util": [], "queue": []})
        per[row["spklu"]]["util"].append(row["utilisasi"])
        per[row["spklu"]]["queue"].append(row["queue"])
    out = {}
    for sid, v in per.items():
        u = np.array(v["util"], float); q = np.array(v["queue"], float)
        out[sid] = {"util_mean": float(u.mean()), "util_sd": float(u.std()),
                    "util_p90": float(np.percentile(u, 90)),
                    "queue_mean": float(q.mean()), "queue_p90": float(np.percentile(q, 90)),
                    "frac_jam_kosong": float(np.mean(u <= 1e-9)),
                    "frac_jam_antre": float(np.mean(q > 0))}
    return out


def agg(runs):
    if not runs:
        return None
    skalar = [k for k in runs[0] if not k.startswith("_")]
    o = {}
    for k in skalar:
        v = np.array([x[k] for x in runs], float)
        o[k] = float(np.nanmean(v)); o[k + "_med"] = float(np.nanmedian(v))
        o[k + "_sd"] = float(np.nanstd(v))
    o["n_seed"] = len(runs)
    # Dua kriteria TERPISAH -- sebelumnya tercampur, membuat lengan ber-Gini 0,036
    # tertandai "terdegradasi" hanya karena waktu tunggunya tinggi.
    o["n_kolaps"] = int(sum(1 for x in runs if x["gini"] > 0.15))
    o["n_wait_tinggi"] = int(sum(1 for x in runs if x["wait"] > 150))
    return o


def agg_harian(runs):
    """Rata-ratakan deret harian lintas seed (panjang deret sama krn horizon sama)."""
    if not runs or not runs[0].get("_harian"):
        return None
    n = min(len(x["_harian"]) for x in runs)
    kunci = [k for k in runs[0]["_harian"][0] if k != "time"]
    out = []
    for i in range(n):
        row = {"hari": runs[0]["_harian"][i]["hari"]}
        for k in kunci:
            if k == "hari":
                continue
            v = np.array([x["_harian"][i][k] for x in runs], float)
            row[k] = float(np.nanmean(v)); row[k + "_sd"] = float(np.nanstd(v))
        out.append(row)
    return out


def main():
    arms = [("S0", lambda sim: None, None),
            ("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2), None),
            ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2), None),
            ("H-PPO(abs)", None, f"hppo_{TAG}_abs"), ("P-PPO(abs)", None, f"pppo_{TAG}_abs"),
            ("H-PPO(sgn)", None, f"hppo_{TAG}_sgn"), ("P-PPO(sgn)", None, f"pppo_{TAG}_sgn"),
            # Lengan MASTER: baseline tanpa encoder riwayat per-pengguna (2026-08-18)
            ("MASTER(abs)", None, f"master_{TAG}_abs"), ("MASTER(sgn)", None, f"master_{TAG}_sgn"),
            ("MASTER-eq(abs)", None, f"mastereq_{TAG}_abs"),
            ("MASTER-eq(sgn)", None, f"mastereq_{TAG}_sgn"),
            ("MASTER+P(abs)", None, f"masterp_{TAG}_abs"),
            ("MASTER+P(sgn)", None, f"masterp_{TAG}_sgn"),
            # T2 (2026-08-19): stasiun-sebagai-agen + bidding. Pasangan pembanding yang
            # BENAR adalah lengan MASTER di atas (sama-sama tanpa riwayat) -- selisihnya
            # murni peran agen + bentuk aksi.
            ("MASTER-bid(abs)", None, f"masterbid_{TAG}_abs"),
            ("MASTER-bid(sgn)", None, f"masterbid_{TAG}_sgn")]

    out = {"horizon": TAG, "seeds": SEEDS, "trust_beku": TRUST_BEKU,
           "per_seed": {}, "agregat": {}, "harian": {}, "stasiun": {}}
    print(f"horizon={TAG}  seed={SEEDS}  (perekaman deret AKTIF)\n", flush=True)

    for mode in ("abs", "signed"):
        for beku in (True, False):
            rez = "beku" if beku else "dinamis"
            print(f"=== aturan={mode}  rezim={rez} ===", flush=True)
            H = "%-14s %7s %7s %7s %8s %8s %7s %7s %8s %8s" % (
                "lengan", "gini", "trust", "acc", "wait~", "wait_p90", "herd", "gW_usr",
                "simpang", "untung%")
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
                out["per_seed"][key] = [{k: v for k, v in r.items() if not k.startswith("_")}
                                        for r in runs]
                a = agg(runs); out["agregat"][key] = a
                out["harian"][key] = agg_harian(runs)
                out["stasiun"][key] = runs[0]["_stasiun"]     # seed pertama saja
                print("%-14s %7.4f %7.4f %7.3f %8.1f %8.1f %7.3f %7.3f %8.3f %7.1f%%" % (
                    lbl, a["gini"], a["trust"], a["acc"], a.get("w_p50", float("nan")),
                    a.get("w_p90", float("nan")), a["herding"],
                    a.get("gini_wait_pengguna", float("nan")),
                    a.get("jn_frac_menyimpang", float("nan")),
                    100 * a.get("jnpatuh_expost_frac_untung", float("nan"))), flush=True)
            print(flush=True)

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

    ring = {}
    for mode in ("abs", "signed"):
        v = [p for k, p in perf.items() if k.endswith("|" + mode) and "PPO" in k]
        if v:
            ring[mode] = {k: float(np.mean([abs(x[k]) for x in v]))
                          for k in ("d_gini", "d_acc", "d_wait")}
    out["ringkas_performativitas"] = ring
    if "abs" in ring and "signed" in ring:
        r = ring["signed"]["d_gini"] / max(ring["abs"]["d_gini"], 1e-12)
        out["rasio_performativitas_signed_thd_abs"] = float(r)
        print(f"\nperformativitas `signed` = {r:.2f}x dari `abs`")

    common.save_json(out, f"uji_konsolidasi_{TAG}.json")
    print(f"\nSAVED -> outputs/uji_konsolidasi_{TAG}.json")


if __name__ == "__main__":
    main()
