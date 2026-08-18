"""Distribusi waktu tunggu -- bukan hanya rerata.

Rerata menyembunyikan dua hal yang menentukan tafsir:

  1. KEMIRINGAN. Bila sebaran berekor, rerata digerakkan sedikit kasus ekstrem dan tidak
     menggambarkan pengalaman pengguna tipikal. Median + persentil yang menggambarkannya.

  2. SIAPA yang menunggu. `wait` dihitung atas SELURUH trip, termasuk pengguna yang MENOLAK
     rekomendasi lalu pergi ke stasiun pilihannya sendiri. Waktu tunggu mereka BUKAN
     konsekuensi dari rekomendasi yang diberikan. Memisahkan patuh vs menolak menentukan
     apakah "RL membuat orang menunggu lama" itu benar atau salah atribusi.

    python _uji_distribusi_wait.py 0,1,2 signed     # aturan trust utk evaluasi
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

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
         if len(sys.argv) > 1 else [0, 1, 2])
MODE = sys.argv[2] if len(sys.argv) > 2 else "signed"
PERSENTIL = [10, 25, 50, 75, 90, 95, 99]
AMBANG = [30, 60, 120, 240]      # menit


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


def kumpul(fac, seed):
    with mode_trust(MODE):
        sim = common.fresh_sim(DS)
        random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    return w, c


def ringkas(w):
    if w.size == 0:
        return None
    o = dict(n=int(w.size), mean=float(w.mean()), sd=float(w.std()), maks=float(w.max()))
    for p in PERSENTIL:
        o[f"p{p}"] = float(np.percentile(w, p))
    for a in AMBANG:
        o[f"frac_gt{a}"] = float(np.mean(w > a))
    o["rasio_mean_median"] = float(o["mean"] / max(o["p50"], 1e-9))
    return o


def main():
    arms = [("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2), None),
            ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2), None),
            ("H-PPO(abs)", None, "hppo_30d_abs"), ("P-PPO(abs)", None, "pppo_30d_abs"),
            ("H-PPO(sgn)", None, "hppo_30d_sgn"), ("P-PPO(sgn)", None, "pppo_30d_sgn"),
            ("S0", lambda sim: None, None)]

    out = {"aturan": MODE, "seeds": SEEDS, "lengan": {}}
    print(f"aturan trust={MODE}  seed={SEEDS}\n", flush=True)

    for lbl, fac, stem in arms:
        W, C = [], []
        for sd in SEEDS:
            f = fac if stem is None else pol(stem, sd)
            if f is None:
                continue
            w, c = kumpul(f, sd)
            W.append(w); C.append(c)
        if not W:
            print("%-14s (checkpoint belum ada)" % lbl, flush=True); continue
        w = np.concatenate(W); c = np.concatenate(C)
        out["lengan"][lbl] = dict(semua=ringkas(w), patuh=ringkas(w[c]),
                                  menolak=ringkas(w[~c]),
                                  frac_patuh=float(c.mean()) if c.size else 0.0)
        print("selesai: %s" % lbl, flush=True)

    def tabel(judul, kunci):
        print(f"\n=== {judul} ===")
        H = "%-14s %7s %7s %7s %7s %7s %7s %7s %8s %7s" % (
            "lengan", "n", "mean", "p50", "p75", "p90", "p95", "p99", "maks", "m/med")
        print(H); print("-" * len(H))
        for lbl, r in out["lengan"].items():
            v = r[kunci]
            if v is None:
                print("%-14s (kosong)" % lbl); continue
            print("%-14s %7d %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %8.0f %7.1f" % (
                lbl, v["n"], v["mean"], v["p50"], v["p75"], v["p90"], v["p95"],
                v["p99"], v["maks"], v["rasio_mean_median"]))

    tabel("SELURUH trip", "semua")
    tabel("hanya yang MEMATUHI rekomendasi", "patuh")
    tabel("hanya yang MENOLAK rekomendasi", "menolak")

    print("\n=== proporsi trip di atas ambang (seluruh trip) ===")
    H = "%-14s" % "lengan" + "".join("%10s" % f">{a}mnt" for a in AMBANG)
    print(H); print("-" * len(H))
    for lbl, r in out["lengan"].items():
        v = r["semua"]
        print("%-14s" % lbl + "".join("%9.1f%%" % (100 * v[f"frac_gt{a}"]) for a in AMBANG))

    common.save_json(out, f"uji_distribusi_wait_{MODE}.json")
    print(f"\nSAVED -> outputs/uji_distribusi_wait_{MODE}.json")


if __name__ == "__main__":
    main()
