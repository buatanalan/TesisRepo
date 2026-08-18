"""K3 -- Mengukur PERFORMATIVITAS: seberapa besar kebijakan mengubah lingkungannya sendiri.

DEFINISI (Perdomo dkk. 2020): kebijakan performatif bila penerapannya MENGUBAH distribusi
tempat ia dievaluasi. Ukurannya adalah SELISIH PERFORMATIF:

    selisih = kinerja(pi, trust DINAMIS) - kinerja(pi, trust BEKU)

Nol berarti kebijakan tidak mengubah lingkungannya -- tidak ada performativitas.

RANCANGAN (ini yang membedakannya dari perbandingan Tahap2-vs-Tahap3 sebelumnya):
kebijakan yang SAMA PERSIS dievaluasi di dua rezim. Perbandingan Tahap2-vs-Tahap3 memakai
checkpoint BERBEDA, sehingga selisihnya mencampur efek rezim dengan efek pelatihan --
tidak dapat dipakai mengukur performativitas.

`constant_trust_shadow(0.5)` membekukan trust yang MENGGERAKKAN KEPUTUSAN
(`User.trust_effective`), sementara `update_trust` tetap berjalan sebagai bayangan
diagnostik. Jadi rezim beku bukan berarti trust mati -- hanya umpan baliknya diputus.

Dijalankan pada KEDUA aturan trust, karena pertanyaan pokoknya: apakah performativitas
masih terukur setelah `signed` menjadi model utama?

    python _uji_performativitas.py                    # hppo_t3 & pppo_t3, seed 0,1,2
    python _uji_performativitas.py hppo_30d_abs 0,1,2
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

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
TRUST_BEKU = 0.5


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


def ev(fac, mode, seed, beku):
    """beku=True -> trust_effective dipatok TRUST_BEKU (umpan balik diputus)."""
    ctx = constant_trust_shadow(value=TRUST_BEKU) if beku else contextlib.nullcontext()
    with mode_trust(mode), ctx:
        sim = common.fresh_sim(DS)
        random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)   # bayangan saat beku
    return dict(gini=float(common.gini(sv)), acc=float(c.mean()) if c.size else 0.0,
                wait=float(w.mean()) if w.size else 0.0,
                trust=float(tr.mean()), served=int(sv.sum()))


def main():
    stems = ([sys.argv[1]] if len(sys.argv) > 1 else ["hppo_t3", "pppo_t3"])
    seeds = ([int(s) for s in sys.argv[2].replace(" ", "").split(",")]
             if len(sys.argv) > 2 else [0, 1, 2])
    out = {"trust_beku": TRUST_BEKU, "seeds": seeds, "hasil": {}}

    print(f"seed={seeds}  trust beku={TRUST_BEKU}  (checkpoint sama di kedua rezim)\n")
    H = "%-16s %-7s %9s %9s %11s | %9s %9s %11s" % (
        "lengan", "aturan", "gini_beku", "gini_din", "SELISIH", "acc_beku", "acc_din", "SELISIH")
    print(H); print("-" * len(H))

    for stem in stems:
        for mode in ("abs", "signed"):
            rb, rd = [], []
            for sd in seeds:
                f = pol(stem, sd)
                if f is None:
                    continue
                rb.append(ev(f, mode, sd, beku=True))
                rd.append(ev(f, mode, sd, beku=False))
            if not rb:
                print("%-16s %-7s (checkpoint belum ada)" % (stem, mode)); continue
            agg = lambda rs, k: float(np.mean([r[k] for r in rs]))
            gb, gd = agg(rb, "gini"), agg(rd, "gini")
            ab, ad = agg(rb, "acc"), agg(rd, "acc")
            wb, wd = agg(rb, "wait"), agg(rd, "wait")
            out["hasil"][f"{stem}|{mode}"] = dict(
                gini_beku=gb, gini_dinamis=gd, d_gini=gd - gb,
                acc_beku=ab, acc_dinamis=ad, d_acc=ad - ab,
                wait_beku=wb, wait_dinamis=wd, d_wait=wd - wb,
                trust_dinamis=agg(rd, "trust"), n_seed=len(rb))
            print("%-16s %-7s %9.4f %9.4f %+11.4f | %9.3f %9.3f %+11.3f" % (
                stem, mode, gb, gd, gd - gb, ab, ad, ad - ab))

    # Besaran performativitas = seberapa jauh rezim dinamis menyimpang dari rezim beku.
    print()
    print("=== besaran performativitas per aturan (rata-rata |selisih| lintas lengan) ===")
    print("%-8s %12s %12s %12s" % ("aturan", "|d_gini|", "|d_acc|", "|d_wait|"))
    print("-" * 48)
    ring = {}
    for mode in ("abs", "signed"):
        v = [r for k, r in out["hasil"].items() if k.endswith("|" + mode)]
        if not v:
            continue
        ring[mode] = dict(
            d_gini=float(np.mean([abs(r["d_gini"]) for r in v])),
            d_acc=float(np.mean([abs(r["d_acc"]) for r in v])),
            d_wait=float(np.mean([abs(r["d_wait"]) for r in v])))
        print("%-8s %12.4f %12.4f %12.1f" % (
            mode, ring[mode]["d_gini"], ring[mode]["d_acc"], ring[mode]["d_wait"]))
    out["ringkas"] = ring

    if "abs" in ring and "signed" in ring:
        r = ring["signed"]["d_gini"] / max(ring["abs"]["d_gini"], 1e-12)
        out["rasio_signed_thd_abs_gini"] = float(r)
        print()
        print(f"performativitas di `signed` = {r:.2f}x dari `abs` (pada Gini)")
        print("  <1 berarti MELEMAH; mendekati 0 berarti fenomena praktis hilang")

    common.save_json(out, "uji_performativitas.json")
    print("\nSAVED -> outputs/uji_performativitas.json")


if __name__ == "__main__":
    main()
