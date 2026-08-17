"""Trust DINAMIS dgn kondisi awal 0,75 (bukan 0,5) -- kapasitas awal lebih besar,
tapi tetap BEREVOLUSI (bisa terkikis). Menguji apakah kapasitas awal yang lebih tinggi
BERTAHAN atau justru terkikis lebih cepat.

Prior diubah alpha0=1,5 / beta0=0,5 -> mean 0,75 dgn KEKUATAN prior SAMA (total 2,0)
spt default (1,0/1,0 -> 0,5). Jadi yang berubah hanya titik awalnya, bukan seberapa
lengket trust terhadap bukti baru.
"""
import sys, os, json, random, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.env.user import User

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
SEEDS = [0, 1, 2]

@contextlib.contextmanager
def init_trust(a0, b0):
    orig = User.__init__
    def patched(self, *a, **kw):
        # WAJIB override, BUKAN setdefault: simulator.py meneruskan trust_alpha0/beta0
        # secara EKSPLISIT (u.get(..., TRUST_ALPHA0)), jadi setdefault tak pernah aktif.
        kw["trust_alpha0"] = a0; kw["trust_beta0"] = b0
        orig(self, *a, **kw)
    User.__init__ = patched
    try: yield
    finally: User.__init__ = orig

class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None: return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}

def ev(fac, a0, b0, seed):
    with init_trust(a0, b0):
        sim = common.fresh_sim(DS); random.seed(seed); np.random.seed(seed)
        sim.run(max_steps=sim.max_steps, agent=fac(sim))
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    wr = (float(w[c].mean()/w[~c].mean()) if c.any() and (~c).any() and w[~c].mean()>1e-9 else np.nan)
    return dict(gini=common.gini(sv), acc=c.mean() if c.size else 0.0,
                wait=w.mean() if w.size else 0.0, wr=wr, served=int(sv.sum()),
                trust=tr.mean(), tr_std=tr.std(), tr_min=tr.min())

from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.rollout import InferenceAgent
def pol(stem, sd):
    m = json.load(open(f"outputs/t2_{stem}_seed{sd}_meta.json"))
    c = PPPOPolicy if m.get("policy_cls")=="PPPOPolicy" else HPPOPolicy
    kw = dict(n_critics=m.get("n_critics",1))
    if c is PPPOPolicy: kw.update(pref_d_lstm=m.get("pref_d_lstm",64), pref_d_attn=m.get("pref_d_attn",64))
    p = c(m["obs_dim"], m["critic_obs_dim"], m["N"], **kw)
    p.load_state_dict(torch.load(f"outputs/t2_{stem}_seed{sd}.pt")); p.eval()
    return lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0, threshold=0.20)

ARMS = [("S0 natural", lambda sim: None),
        ("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2)),
        ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2)),
        ("H-PPO T3 s1", pol("hppo_t3",1)), ("H-PPO T3 s2", pol("hppo_t3",2)),
        ("P-PPO T3 s0", pol("pppo_t3",0)), ("P-PPO T3 s2", pol("pppo_t3",2))]

# Swakuji: S0 tak pernah memicu update_trust, jadi trust-nya HARUS persis prior.
_chk = ev(lambda sim: None, 1.5, 0.5, 0)["trust"]
assert abs(_chk - 0.75) < 1e-9, f"patch init trust GAGAL: S0 trust={_chk:.4f}, harusnya 0,75"
print(f"swakuji OK: S0 pada prior 1,5/0,5 -> trust {_chk:.4f}", flush=True)

out={}
for lbl, fac in ARMS:
    for tag,(a0,b0) in [("init0.50",(1.0,1.0)), ("init0.75",(1.5,0.5))]:
        rs=[ev(fac,a0,b0,s) for s in SEEDS]
        out[f"{lbl}|{tag}"]={k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
        print(f"{lbl:<14} {tag}  gini={out[f'{lbl}|{tag}']['gini']:.4f} "
              f"trust={out[f'{lbl}|{tag}']['trust']:.4f}", flush=True)
common.save_json(out,"uji_init075_dinamis.json"); print("\nSAVED")
