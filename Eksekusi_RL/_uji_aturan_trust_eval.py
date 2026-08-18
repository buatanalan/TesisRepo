"""Evaluasi faktorial aturan-trust (abs vs signed), teragregasi lintas seed.

Setiap aturan dievaluasi dgn baseline-nya SENDIRI (S0/greedy dijalankan ulang di bawah
aturan itu). Membandingkan RL aturan-signed thd greedy aturan-abs akan mencampur dua
faktor sekaligus, jadi baseline tak boleh dipakai ulang lintas mode.

Kebijakan RL dievaluasi pada aturan yang MELATIHNYA (matched). Sel silang (dilatih abs,
diuji signed dst) juga dihitung -- itu yang memisahkan "kebijakannya memang lebih baik"
dari "lingkungan pengujiannya lebih longgar", krn checkpoint-nya identik.

Pemakaian (argumen SAMA dgn _uji_aturan_trust.py):
    python _uji_aturan_trust_eval.py 0,1,2 30d
    python _uji_aturan_trust_eval.py 0 90d

Keluaran: outputs/uji_aturan_trust_<horizon>.json  (pola `uji_*.json` -> ikut terlacak git)
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
import marl_spklu.env.user as U
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.rl.registry import bangun_kebijakan
from marl_spklu.rl.rollout import InferenceAgent
from _uji_aturan_trust import mode_trust, DATASET, TAG_HORIZON, SEEDS, nama_lengan

DS = os.path.join(common.ROOT, DATASET)


class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}


def pol(stem, seed):
    """Muat checkpoint; None bila belum ada (lengan itu dilewati, bukan menggagalkan run)."""
    ck = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}.pt")
    mp = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}_meta.json")
    if not (os.path.exists(ck) and os.path.exists(mp)):
        return None
    m = json.load(open(mp))
    # Registri tunggal (marl_spklu/rl/registry.py) -- bentuk biner lama di sini memuat
    # SELURUH lengan MASTER sbg HPPOPolicy dan gagal pada bentuk bobot.
    p = bangun_kebijakan(m, state_dict=torch.load(ck))
    return lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0, threshold=0.20)


def ev(fac, mode, seed):
    with mode_trust(mode):
        sim = common.fresh_sim(DS)
        random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    d = np.array([l["wait_time"] - l["est_wait"] for l in sim.logs if l.get("complied")], float)
    return dict(gini=float(common.gini(sv)), acc=float(c.mean()) if c.size else 0.0,
                wait=float(w.mean()) if w.size else 0.0, served=int(sv.sum()),
                trust=float(tr.mean()), tr_min=float(tr.min()),
                turun_pct=float(100 * (0.5 - tr.mean()) / 0.5),
                pct_telat=float(100 * np.mean(d >= U.DELTAW_TOL_HIGH)) if d.size else 0.0,
                pct_cepat=float(100 * np.mean(-d >= U.DELTAW_TOL_HIGH)) if d.size else 0.0)


def agg(runs):
    """Mean DAN median dilaporkan keduanya: distribusi hasil berekor (run terdegradasi),
    sehingga mean saja menyesatkan dan median saja menyembunyikan kegagalan."""
    if not runs:
        return None
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs], float)
        out[k] = float(v.mean()); out[k + "_med"] = float(np.median(v))
        out[k + "_sd"] = float(v.std())
    out["n_seed"] = len(runs)
    # Run terdegradasi: Gini tinggi ATAU wait meledak. Dilaporkan, bukan dibuang.
    out["n_degradasi"] = int(sum(1 for r in runs if r["gini"] > 0.15 or r["wait"] > 150))
    return out


def baris(lbl, a):
    return ("%-20s %7.4f %7.4f | %7.4f %7.4f | %6.3f %8.2f | %4d %5d" % (
        lbl, a["gini"], a["gini_med"], a["trust"], a["trust_med"],
        a["acc"], a["wait"], a["n_seed"], a["n_degradasi"]))


HEAD = ("%-20s %7s %7s | %7s %7s | %6s %8s | %4s %5s" % (
    "lengan", "gini", "gini~", "trust", "trust~", "acc", "wait", "seed", "degr"))


def main():
    print(f"horizon={TAG_HORIZON} ({DATASET})  seed={SEEDS}", flush=True)
    out = {}

    for mode in ("abs", "signed"):
        arms = [("S0 natural", lambda sim: None, None),
                ("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2), None),
                ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2), None),
                (f"H-PPO[{mode}]", None, nama_lengan("hppo", mode)),
                (f"P-PPO[{mode}]", None, nama_lengan("pppo", mode))]
        print(f"\n=== aturan trust: {mode} | horizon {TAG_HORIZON} ===", flush=True)
        print(HEAD); print("-" * len(HEAD))
        for lbl, fac, stem in arms:
            runs = []
            for sd in SEEDS:
                f = fac if stem is None else pol(stem, sd)
                if f is None:
                    continue
                runs.append(ev(f, mode, sd))
            a = agg(runs)
            if a is None:
                print("%-20s  (checkpoint belum ada, dilewati)" % lbl, flush=True); continue
            out[f"{lbl}|{mode}"] = a
            print(baris(lbl, a), flush=True)

    # Sel silang: checkpoint IDENTIK, hanya aturan pengujian yang berbeda. Ini rancangan
    # terbersih utk mengisolasi efek aturan -- tak ada perbedaan pelatihan sama sekali.
    print(f"\n=== sel silang (checkpoint sama, aturan uji berbeda) ===", flush=True)
    print(HEAD); print("-" * len(HEAD))
    for metode in ("hppo", "pppo"):
        for tr_mode in ("abs", "signed"):
            stem = nama_lengan(metode, tr_mode)
            te_mode = "signed" if tr_mode == "abs" else "abs"
            runs = [ev(p, te_mode, sd) for sd in SEEDS
                    if (p := pol(stem, sd)) is not None]
            a = agg(runs)
            if a is None:
                continue
            out[f"{stem}|latih={tr_mode}|uji={te_mode}"] = a
            print(baris(f"{stem}->{te_mode}", a), flush=True)

    nama = f"uji_aturan_trust_{TAG_HORIZON}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}")


if __name__ == "__main__":
    main()
