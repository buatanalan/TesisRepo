"""Perbandingan final Tahap 2 (trust STATIS) vs Tahap 3 (trust DINAMIS) --
seluruh metode x seluruh metrik, TERMASUK trust akhir.

Trust dicatat dari `User.trust` (alpha/(alpha+beta)) -- BUKAN `trust_effective`:
  * mode DINAMIS : `trust` = trust sungguhan yang dipakai keputusan
  * mode STATIS  : `trust` = trust BAYANGAN -- `update_trust` tetap berjalan, tapi
                   `trust_effective` dibekukan 0,5 oleh `constant_trust_shadow`.
                   Jadi ia menunjukkan APA YANG AKAN TERJADI pada trust seandainya
                   loop tidak diputus -- diagnostik, bukan yang menggerakkan keputusan.
"""
import sys, os, json, random, contextlib, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.experiments.ablations import constant_trust_shadow
from marl_spklu.agents.greedy_agent import GreedyAgent

EVAL_SEEDS = [0, 1, 2]
TRUST_STATIS = 0.5
DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")


class VW(ForecasterBase):
    def predict(self, spklus, time_now_min=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {s: 0.0 for s in spklus}
        return {s: float(sim.compute_virtual_wait(user, v, time_now_min))
                for s, v in spklus.items()}


def evaluasi(fac, dinamis, seed):
    ctx = contextlib.nullcontext() if dinamis else constant_trust_shadow(value=TRUST_STATIS)
    with ctx:
        sim = common.fresh_sim(DS)
        random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))

    served = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([L["wait_time"] for L in sim.logs], float)
    comp = np.array([bool(L["complied"]) for L in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)     # lihat catatan modul
    wr = (float(w[comp].mean() / w[~comp].mean())
          if comp.any() and (~comp).any() and w[~comp].mean() > 1e-9 else float("nan"))
    return dict(gini=float(common.gini(served)), acceptance=float(comp.mean()) if comp.size else 0.0,
                wait=float(w.mean()) if w.size else 0.0, wait_ratio=wr,
                served=int(served.sum()), trust=float(tr.mean()),
                trust_std=float(tr.std()), trust_min=float(tr.min()))


def lengan(dinamis):
    fc = VW()
    L = [("S0 natural", lambda sim: None),
         ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2)),
         ("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2))]
    from marl_spklu.rl.policy import HPPOPolicy
    from marl_spklu.rl.p_ppo_policy import PPPOPolicy
    from marl_spklu.rl.rollout import InferenceAgent
    stems = ([("H-PPO", "hppo_t3"), ("P-PPO", "pppo_t3")] if dinamis
             else [("H-PPO", "hppo_K1_sb4x_bnd"), ("P-PPO", "pppo_sb4x_d16_bnd")])
    for lbl, stem in stems:
        for sd in EVAL_SEEDS:
            mp = os.path.join(common.OUTDIR, f"t2_{stem}_seed{sd}_meta.json")
            ck = os.path.join(common.OUTDIR, f"t2_{stem}_seed{sd}.pt")
            if not (os.path.exists(mp) and os.path.exists(ck)):
                continue
            m = json.load(open(mp))
            cls = PPPOPolicy if m.get("policy_cls") == "PPPOPolicy" else HPPOPolicy
            kw = dict(n_critics=m.get("n_critics", 1))
            if cls is PPPOPolicy:
                kw["pref_d_lstm"] = m.get("pref_d_lstm", 64)
                kw["pref_d_attn"] = m.get("pref_d_attn", 64)
            pol = cls(m["obs_dim"], m["critic_obs_dim"], m["N"], **kw)
            pol.load_state_dict(torch.load(ck)); pol.eval()
            L.append((f"{lbl} seed{sd}",
                      lambda sim, p=pol: InferenceAgent(p, sim, fc, k=2,
                                                        epsilon=0.0, threshold=0.20)))
    return L


def main():
    out = {}
    for dinamis, judul in [(False, "TAHAP 2 - trust STATIS (kontrol)"),
                           (True, "TAHAP 3 - trust DINAMIS (perlakuan)")]:
        print(f"\n=== {judul} ===")
        print("%-16s %17s %8s %9s %9s %7s %17s %7s" % (
            "lengan", "gini", "accept", "wait", "wait_rt", "served", "trust", "tr_min"))
        print("-" * 104)
        for lbl, fac in lengan(dinamis):
            rs = [evaluasi(fac, dinamis, sd) for sd in EVAL_SEEDS]
            g = lambda k: np.nanmean([r[k] for r in rs])
            s = lambda k: np.nanstd([r[k] for r in rs])
            key = ("T3 " if dinamis else "T2 ") + lbl
            out[key] = {k: float(g(k)) for k in rs[0]}
            print("%-16s %8.4f±%-8.4f %8.3f %9.1f %9.2f %7d %8.4f±%-8.4f %7.3f" % (
                lbl, g("gini"), s("gini"), g("acceptance"), g("wait"), g("wait_ratio"),
                int(g("served")), g("trust"), s("trust"), g("trust_min")), flush=True)
    common.save_json(out, "banding_final_t2_t3.json")
    print("\nSAVED -> banding_final_t2_t3.json")


if __name__ == "__main__":
    main()
