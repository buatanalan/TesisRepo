"""Uji kapasitas: kebijakan yang SUDAH dilatih dievaluasi pada trust DIBEKUKAN 0,75
(bukan 0,5). Tanpa training ulang.

Trust = bobot P_rec dlm keputusan pengguna. Menaikkannya 0,5 -> 0,75 memberi metode
KAPASITAS MEMBUJUK lebih besar tanpa mengubah kebijakannya. Yang dijawab:
  * membaik banyak  -> rekomendasinya sudah bagus, KEPATUHAN yang jadi penghambat
  * membaik sedikit -> rekomendasinya sendiri yang jadi batas

CATATAN: kebijakan dilatih pd trust 0,5 (T2) atau dinamis (T3), jadi evaluasi di 0,75
bersifat OUT-OF-DISTRIBUTION -- `hist_lstm` melihat pola kepatuhan yang beda dari saat
latih. Itu bagian dari yang diuji (generalisasi lintas rezim trust), bukan cacat.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.experiments.ablations import constant_trust_shadow
from marl_spklu.agents.greedy_agent import GreedyAgent

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
SEEDS = [0, 1, 2]

class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None: return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}

def ev(fac, trust, seed):
    with constant_trust_shadow(value=trust):
        sim = common.fresh_sim(DS); random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    comp = np.array([bool(l["complied"]) for l in sim.logs], bool)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    return (common.gini(sv), comp.mean() if comp.size else 0.0,
            w.mean() if w.size else 0.0, int(sv.sum()))

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

ARMS = [("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2)),
        ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2)),
        ("H-PPO T2 s0", pol("hppo_K1_sb4x_bnd", 0)),
        ("H-PPO T2 s1", pol("hppo_K1_sb4x_bnd", 1)),
        ("P-PPO T2 s2", pol("pppo_sb4x_d16_bnd", 2)),
        ("H-PPO T3 s1", pol("hppo_t3", 1)),
        ("P-PPO T3 s2", pol("pppo_t3", 2))]

print("%-14s %-27s %-27s %s" % ("", "--- trust 0,50 ---", "--- trust 0,75 ---", "selisih"))
print("%-14s %8s %7s %8s %8s %7s %8s %9s" % ("lengan","gini","accept","served","gini","accept","served","d_gini"))
print("-"*96)
out={}
for lbl, fac in ARMS:
    a = np.array([ev(fac, 0.50, s) for s in SEEDS]).mean(0)
    b = np.array([ev(fac, 0.75, s) for s in SEEDS]).mean(0)
    out[lbl] = dict(t050=list(a), t075=list(b))
    print("%-14s %8.4f %7.3f %8d %8.4f %7.3f %8d %+9.4f" % (
        lbl, a[0], a[1], int(a[3]), b[0], b[1], int(b[3]), b[0]-a[0]), flush=True)
common.save_json(out, "uji_trust075.json")
print("\nSAVED -> uji_trust075.json")
