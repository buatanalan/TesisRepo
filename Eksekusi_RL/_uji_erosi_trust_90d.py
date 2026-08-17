"""Uji hipotesis: trust yang terus menurun mengembalikan sistem ke kondisi natural.

Kebijakan TERLATIH (dari Tahap 3, 30 hari) dijalankan pada horizon 90 HARI dgn trust
DINAMIS, sambil merekam trust & Gini tiap 10 hari. Tanpa training ulang -- murni menguji
keberlanjutan intervensi.

Dasar hipotesis: trust = alpha/(alpha+beta) menuju asimtot SUM_a/(SUM_a+SUM_b), yang dari
pengukuran 30 hari ada di 0,12-0,27 -- jauh di bawah 0,5. Pada nilai itu keputusan pengguna
didominasi P_pref, yakni kondisi natural.
"""
import sys, os, json, random, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.agents.greedy_agent import GreedyAgent

DS90 = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x_90d.json")

class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None: return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}

def jalan(fac, seed=0, tiap_hari=10):
    sim = common.fresh_sim(DS90); random.seed(seed); np.random.seed(seed)
    ag = fac(sim)
    per = int(tiap_hari * 24 * 60 / sim.dt_minutes)
    jejak = []
    for step in range(sim.max_steps):
        sim.step_once(step, agent=ag)
        if (step + 1) % per == 0:
            sv = np.array([s.total_served for s in sim.spklus.values()], float)
            tr = np.array([u.trust for u in sim.users], float)
            jejak.append(dict(hari=int((step + 1) * sim.dt_minutes / 1440),
                              trust=float(tr.mean()), gini=float(common.gini(sv)),
                              served=int(sv.sum())))
    return jejak

from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.rollout import InferenceAgent
def pol(stem, sd):
    m = json.load(open(f"outputs/t2_{stem}_seed{sd}_meta.json"))
    c = PPPOPolicy if m.get("policy_cls") == "PPPOPolicy" else HPPOPolicy
    kw = dict(n_critics=m.get("n_critics", 1))
    if c is PPPOPolicy:
        kw.update(pref_d_lstm=m.get("pref_d_lstm", 64), pref_d_attn=m.get("pref_d_attn", 64))
    p = c(m["obs_dim"], m["critic_obs_dim"], m["N"], **kw)
    p.load_state_dict(torch.load(f"outputs/t2_{stem}_seed{sd}.pt")); p.eval()
    return lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0, threshold=0.20)

LENGAN = [("S0 natural", lambda sim: None),
          ("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2)),
          ("H-PPO t3s1", pol("hppo_t3", 1)),
          ("P-PPO t3s2", pol("pppo_t3", 2))]

if __name__ == "__main__":
    out = {}
    for lbl, fac in LENGAN:
        j = jalan(fac)
        out[lbl] = j
        print("%-14s trust: %s" % (lbl, " ".join("%.3f" % x["trust"] for x in j)), flush=True)
        print("%-14s gini : %s" % ("", " ".join("%.3f" % x["gini"] for x in j)), flush=True)
    common.save_json(dict(hari=[x["hari"] for x in out["S0 natural"]], jejak=out),
                     "uji_erosi_trust_90d.json")
    print("\nSAVED -> uji_erosi_trust_90d.json")
