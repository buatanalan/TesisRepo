"""Evaluasi Tahap 4 (90 hari, trust dinamis) pada kedua aturan trust.

Setiap aturan dievaluasi dgn baseline-nya SENDIRI (S0/greedy dijalankan ulang di bawah
aturan itu). Membandingkan RL aturan-signed thd greedy aturan-abs akan mencampur dua
faktor sekaligus, jadi baseline tak boleh dipakai ulang lintas mode.

Kebijakan RL dievaluasi pada aturan yang MELATIHNYA (matched). Sel silang (dilatih abs,
diuji signed dst) juga dihitung -- itu yang menunjukkan apakah keunggulan berasal dari
kebijakan yang dipelajari atau semata dari lingkungan pengujian yang lebih longgar.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
import marl_spklu.env.user as U
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.rollout import InferenceAgent
from _tahap4_90d import mode_trust, DATASET_90D, SEED

DS = os.path.join(common.ROOT, DATASET_90D)


class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}


def pol(stem):
    m = json.load(open(f"outputs/t2_{stem}_seed{SEED}_meta.json"))
    c = PPPOPolicy if m.get("policy_cls") == "PPPOPolicy" else HPPOPolicy
    kw = dict(n_critics=m.get("n_critics", 1))
    if c is PPPOPolicy:
        kw.update(pref_d_lstm=m.get("pref_d_lstm", 64), pref_d_attn=m.get("pref_d_attn", 64))
    p = c(m["obs_dim"], m["critic_obs_dim"], m["N"], **kw)
    p.load_state_dict(torch.load(f"outputs/t2_{stem}_seed{SEED}.pt"))
    p.eval()
    return lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0, threshold=0.20)


def ev(fac, mode):
    with mode_trust(mode):
        sim = common.fresh_sim(DS)
        random.seed(SEED); np.random.seed(SEED)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    d = np.array([l["wait_time"] - l["est_wait"] for l in sim.logs if l.get("complied")], float)
    return dict(gini=float(common.gini(sv)), acc=float(c.mean()) if c.size else 0.0,
                wait=float(w.mean()) if w.size else 0.0, served=int(sv.sum()),
                trust=float(tr.mean()), tr_std=float(tr.std()), tr_min=float(tr.min()),
                turun=float(0.5 - tr.mean()), turun_pct=float(100 * (0.5 - tr.mean()) / 0.5),
                n_janji=int(d.size),
                pct_telat=float(100 * np.mean(d >= U.DELTAW_TOL_HIGH)) if d.size else 0.0,
                pct_cepat=float(100 * np.mean(-d >= U.DELTAW_TOL_HIGH)) if d.size else 0.0)


def main():
    out, rows = {}, []
    for mode in ("abs", "signed"):
        tag = "abs" if mode == "abs" else "sgn"
        arms = [("S0 natural", lambda sim: None),
                ("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2)),
                ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2)),
                (f"H-PPO 90d[{tag}]", pol(f"hppo_90d_{tag}")),
                (f"P-PPO 90d[{tag}]", pol(f"pppo_90d_{tag}"))]
        print(f"\n=== aturan trust: {mode} | horizon 90 hari | seed {SEED} ===", flush=True)
        print("%-18s %8s %8s %8s %8s %8s %8s %8s" % (
            "lengan", "gini", "trust", "turun%", "accept", "wait", "%telat", "%cepat"))
        print("-" * 82)
        for lbl, fac in arms:
            r = ev(fac, mode); out[f"{lbl}|{mode}"] = r
            rows.append((mode, lbl, r))
            print("%-18s %8.4f %8.4f %7.1f%% %8.3f %8.2f %7.1f%% %7.1f%%" % (
                lbl, r["gini"], r["trust"], r["turun_pct"], r["acc"], r["wait"],
                r["pct_telat"], r["pct_cepat"]), flush=True)

    # Sel silang: kebijakan dilatih di satu aturan, diuji di aturan lain.
    print("\n=== sel silang (dilatih -> diuji) ===", flush=True)
    print("%-28s %8s %8s %8s" % ("kombinasi", "gini", "trust", "turun%"))
    print("-" * 56)
    for stem, tr_mode in [("hppo_90d_abs", "abs"), ("hppo_90d_sgn", "signed"),
                          ("pppo_90d_abs", "abs"), ("pppo_90d_sgn", "signed")]:
        for te_mode in ("abs", "signed"):
            if te_mode == tr_mode:
                continue
            r = ev(pol(stem), te_mode); out[f"{stem}|latih={tr_mode}|uji={te_mode}"] = r
            print("%-28s %8.4f %8.4f %7.1f%%" % (
                f"{stem} -> {te_mode}", r["gini"], r["trust"], r["turun_pct"]), flush=True)

    common.save_json(out, "tahap4_90d_eval.json")
    print("\nSAVED -> outputs/tahap4_90d_eval.json")


if __name__ == "__main__":
    main()
