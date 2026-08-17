"""G1 -- Apakah keunggulan P-PPO benar-benar berasal dari INFORMASI preferensi?

LATAR: `pref_gate` pada checkpoint yang berlaku bernilai kecil dan tandanya tak konsisten
(-0,039 s/d -0,156; 5 negatif, 1 positif). Argumen KB.3 lama ("gate konvergen ke 0,213 ->
modul terpakai") karenanya gugur. Namun P-PPO tetap unggul jelas (0,0365 vs 0,0638). Maka
keunggulannya mungkin BUKAN dari informasi preferensi.

TIGA KONDISI, checkpoint IDENTIK, evaluasi saja (tanpa latih ulang) sehingga tak ada
perbedaan pelatihan yang mengotori perbandingan:

  A0  kontrol   -- apa adanya
  A1  gate=0    -- modul DIHAPUS dari jalur maju (setara fungsional H-PPO)
  A2  riwayat   -- modul & kapasitasnya TETAP, tetapi diberi riwayat MILIK PENGGUNA LAIN
      ditukar      sehingga informasinya rusak sementara distribusi masukan tetap sama

Pemisahan A1/A2 itu intinya: A1 menghapus modul, A2 hanya menghapus informasinya.

  A1 & A2 sama-sama memburuk -> modul memakai informasi preferensi (K2/K3 berdiri)
  A1 memburuk, A2 tidak      -> yang berkontribusi KAPASITAS, bukan preferensi
  keduanya tak berubah       -> modul tak berkontribusi (cabang H3' dokumen v2)

    python _uji_ablasi_preferensi.py 0,1,2 signed
"""
import sys, os, json, random, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
import marl_spklu.env.user as U
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.rollout import InferenceAgent, RLRolloutAgent

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
         if len(sys.argv) > 1 else [0, 1, 2])
MODE = sys.argv[2] if len(sys.argv) > 2 else "signed"


@contextlib.contextmanager
def mode_trust(mode):
    orig = U.TRUST_PENALTY_MODE
    U.TRUST_PENALTY_MODE = mode
    try:
        yield
    finally:
        U.TRUST_PENALTY_MODE = orig


@contextlib.contextmanager
def riwayat_ditukar(seed):
    """A2: `_build_pref_hist` mengembalikan riwayat pengguna LAIN yang dipilih acak.

    Memakai RNG TERPISAH (bukan `random` global) supaya aliran acak simulasi tidak
    tergeser -- kalau tergeser, selisih hasil bisa berasal dari lintasan simulasi yang
    berbeda, bukan dari ablasi.
    """
    rng = random.Random(10_000 + seed)
    asli = RLRolloutAgent._build_pref_hist

    def ditukar(self, user):
        kunci = [k for k in self._pref_hist.keys() if k != user.user_id]
        if not kunci:
            return asli(self, user)
        donor = rng.choice(kunci)

        class _Palsu:            # hanya perlu atribut user_id
            user_id = donor
        return asli(self, _Palsu())

    RLRolloutAgent._build_pref_hist = ditukar
    try:
        yield
    finally:
        RLRolloutAgent._build_pref_hist = asli


class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}


def muat(stem, seed, gate_nol=False):
    ck = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}.pt")
    mp = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}_meta.json")
    if not (os.path.exists(ck) and os.path.exists(mp)):
        return None, None
    m = json.load(open(mp))
    c = PPPOPolicy if m.get("policy_cls") == "PPPOPolicy" else HPPOPolicy
    kw = dict(n_critics=m.get("n_critics", 1))
    if c is PPPOPolicy:
        kw.update(pref_d_lstm=m.get("pref_d_lstm", 64), pref_d_attn=m.get("pref_d_attn", 64))
    p = c(m["obs_dim"], m["critic_obs_dim"], m["N"], **kw)
    p.load_state_dict(torch.load(ck))
    gate = float(p.pref_gate.item()) if hasattr(p, "pref_gate") else float("nan")
    if gate_nol and hasattr(p, "pref_gate"):
        with torch.no_grad():
            p.pref_gate.fill_(0.0)
    p.eval()
    return (lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0,
                                             threshold=0.20)), gate


def ev(fac, seed):
    with mode_trust(MODE):
        sim = common.fresh_sim(DS)
        random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    return dict(gini=float(common.gini(sv)), acc=float(c.mean()) if c.size else 0.0,
                wait=float(w.mean()) if w.size else 0.0, trust=float(tr.mean()),
                served=int(sv.sum()))


def main():
    out = {"aturan": MODE, "seeds": SEEDS, "hasil": {}, "gate": {}}
    print(f"aturan={MODE}  seed={SEEDS}  (checkpoint identik di ketiga kondisi)\n", flush=True)
    H = "%-16s %-14s %8s %8s %8s %8s" % ("lengan", "kondisi", "gini", "acc", "wait", "trust")
    print(H); print("-" * len(H))

    for stem in ("pppo_30d_abs", "pppo_30d_sgn"):
        for lbl, gate_nol, tukar in (("A0 kontrol", False, False),
                                     ("A1 gate=0", True, False),
                                     ("A2 riwayat tukar", False, True)):
            runs, gates = [], []
            for sd in SEEDS:
                fac, g = muat(stem, sd, gate_nol=gate_nol)
                if fac is None:
                    continue
                gates.append(g)
                if tukar:
                    with riwayat_ditukar(sd):
                        runs.append(ev(fac, sd))
                else:
                    runs.append(ev(fac, sd))
            if not runs:
                print("%-16s %-14s (checkpoint belum ada)" % (stem, lbl)); continue
            a = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            a["gini_sd"] = float(np.std([r["gini"] for r in runs]))
            a["per_seed_gini"] = [r["gini"] for r in runs]
            out["hasil"][f"{stem}|{lbl}"] = a
            out["gate"][stem] = gates
            print("%-16s %-14s %8.4f %8.3f %8.1f %8.4f" % (
                stem, lbl, a["gini"], a["acc"], a["wait"], a["trust"]), flush=True)
        print(flush=True)

    # Pembanding: H-PPO pada kondisi yang sama -- A1 seharusnya mendekatinya bila modul
    # memang satu-satunya pembeda.
    for stem in ("hppo_30d_abs", "hppo_30d_sgn"):
        runs = []
        for sd in SEEDS:
            fac, _ = muat(stem, sd)
            if fac is not None:
                runs.append(ev(fac, sd))
        if runs:
            a = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            out["hasil"][f"{stem}|pembanding"] = a
            print("%-16s %-14s %8.4f %8.3f %8.1f %8.4f" % (
                stem, "(pembanding)", a["gini"], a["acc"], a["wait"], a["trust"]), flush=True)

    print("\n=== SELISIH terhadap kontrol A0 ===")
    print("%-16s %12s %12s" % ("lengan", "A1 - A0", "A2 - A0"))
    print("-" * 42)
    for stem in ("pppo_30d_abs", "pppo_30d_sgn"):
        k0 = f"{stem}|A0 kontrol"
        if k0 not in out["hasil"]:
            continue
        g0 = out["hasil"][k0]["gini"]
        d1 = out["hasil"].get(f"{stem}|A1 gate=0", {}).get("gini")
        d2 = out["hasil"].get(f"{stem}|A2 riwayat tukar", {}).get("gini")
        out["hasil"][k0]["d_A1"] = None if d1 is None else d1 - g0
        out["hasil"][k0]["d_A2"] = None if d2 is None else d2 - g0
        print("%-16s %+12.4f %+12.4f" % (stem, (d1 or 0) - g0, (d2 or 0) - g0))
    print("\n(+ berarti MEMBURUK saat modul/informasinya dihapus -> modul berkontribusi)")

    common.save_json(out, f"uji_ablasi_preferensi_{MODE}.json")
    print(f"\nSAVED -> outputs/uji_ablasi_preferensi_{MODE}.json")


if __name__ == "__main__":
    main()
