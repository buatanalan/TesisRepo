"""Sapuan `phantom_weight` CongestionAwareVWF -- TANPA retraining kebijakan (kebijakan
sudah dilatih pakai VirtualWaitForecaster biasa; forecaster HANYA dipakai menghasilkan
janji yang ditampilkan/dinilai trust, tidak mengubah keputusan aktor). Ukur apakah
korelasi rec_activity-vs-|Delta W| (diagnosis sebelumnya, rho=0,359) berkurang."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOPolicy, MasterEVPPOInferenceAgent,
                                                MasterEVPPORolloutAgent)
from marl_spklu.rl.forecaster import CongestionAwareVWF, CongestionAwareVWFv2
from scipy.stats import spearmanr, mannwhitneyu

CKPT = os.path.join(common.OUTDIR, "master_ev_ppo_eq1_vwf_seimbang4x_K4_gap_sig1_actor_seed0.pt")
DATASET = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")

sim0 = common.fresh_sim(DATASET)
N = len(sim0.spklus)
pol = MasterEVPPOPolicy(N, n_critics=4)
pol.load_state_dict(torch.load(CKPT, map_location="cpu"))
pol.eval()


def run_weight(w, n_seed=3, horizon_steps=2):
    side, records = {}, []
    orig_decide = MasterEVPPORolloutAgent.on_decision
    def patched_decide(self, user, chosen_spklu_id, recs, feasible_spklus):
        if self._pending is not None:
            side[id(self._pending[0])] = self._pending[5]
        return orig_decide(self, user, chosen_spklu_id, recs, feasible_spklus)
    MasterEVPPORolloutAgent.on_decision = patched_decide

    orig_complete = MasterEVPPORolloutAgent.on_charge_complete
    def patched_complete(self, user):
        tr = self._user_trip_tr.get(user.user_id)
        if tr is not None and tr.complied:
            rrc = side.pop(id(tr), None)
            if rrc is not None:
                records.append((rrc, user.wait_time - tr.disp_estwait))
        return orig_complete(self, user)
    MasterEVPPORolloutAgent.on_charge_complete = patched_complete

    fc = CongestionAwareVWFv2(phantom_weight=w, horizon_steps=horizon_steps)
    for s in range(n_seed):
        sim = common.fresh_sim(DATASET)
        random.seed(s); np.random.seed(s)
        agent = MasterEVPPOInferenceAgent(pol, sim, fc, k=3)
        sim.run(max_steps=sim.max_steps, agent=agent)

    MasterEVPPORolloutAgent.on_decision = orig_decide
    MasterEVPPORolloutAgent.on_charge_complete = orig_complete

    rrc = np.array([r[0] for r in records], dtype=float)
    dw = np.array([r[1] for r in records], dtype=float)
    adw = np.abs(dw)
    rho, p = spearmanr(rrc, adw)
    p25, p75 = np.percentile(rrc, [25, 75])
    low, high = adw[rrc <= p25], adw[rrc >= p75]
    return dict(w=w, n=len(records), adw_mean=float(adw.mean()), adw_median=float(np.median(adw)),
               rho=float(rho), p=float(p),
               adw_low=float(low.mean()), adw_high=float(high.mean()),
               ratio_high_low=float(high.mean() / max(low.mean(), 1e-9)),
               frac_over10_low=float((low > 10).mean()), frac_over10_high=float((high > 10).mean()))


print("phantom_weight |  n  | |DW| mean | |DW| rendah | |DW| tinggi | rasio | rho(rec_act,|DW|)")
rows = []
for w in [0.0, 1.0, 2.0, 5.0, 10.0]:
    r = run_weight(w, n_seed=3, horizon_steps=2)
    rows.append(r)
    print(f"  {r['w']:>5.2f}       | {r['n']:>5d} | {r['adw_mean']:>9.2f} | {r['adw_low']:>10.2f} | "
         f"{r['adw_high']:>10.2f} | {r['ratio_high_low']:>4.2f}x | rho={r['rho']:+.4f} (p={r['p']:.1e})",
         flush=True)

common.save_json(rows, "kalibrasi_congestion_aware_vwf.json")
print("\nOK -- ditulis ke outputs/kalibrasi_congestion_aware_vwf.json")
