"""Diagnosis: apakah `recent_recs` (rec_activity, sinyal yg SUDAH dilihat aktor) berkorelasi
dgn ketidakakuratan janji VWF (Delta W = wait_aktual - disp_estwait) pada trip yg PATUH?

Hipotesis kerja (sesi diskusi): aktor tahu risiko penumpukan (rec_activity tinggi), tapi
VWF yg dipakai menghasilkan janji BUTA thd sinyal itu -- makin ramai rec_activity stasiun
saat keputusan diambil, makin besar Delta W (janji under-promise, realisasi jauh lebih lama).

Kalau terkonfirmasi (korelasi positif jelas): solusi #1 (VWF sadar rec_activity) layak
dicoba. Dijalankan pada checkpoint eq1 (H1a, koordinasi murni tanpa preferensi, forecaster
oracle vwf) -- lengan yg datanya paling lengkap (3 seed, 300 update).
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOPolicy, MasterEVPPOInferenceAgent,
                                                MasterEVPPORolloutAgent)
from marl_spklu.rl.forecaster import VirtualWaitForecaster

CKPT = os.path.join(common.OUTDIR, "master_ev_ppo_eq1_vwf_seimbang4x_K4_gap_sig1_actor_seed0.pt")
DATASET = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
assert os.path.exists(CKPT), f"checkpoint tak ditemukan: {CKPT}"
assert os.path.exists(DATASET), f"dataset tak ditemukan: {DATASET}"

sim0 = common.fresh_sim(DATASET)
N = len(sim0.spklus)
pol = MasterEVPPOPolicy(N, n_critics=4)
pol.load_state_dict(torch.load(CKPT, map_location="cpu"))
pol.eval()

# --- Instrumentasi: tangkap recent_rec_count SAAT keputusan (on_decision) & ---
# --- Delta W SAAT sesi selesai (on_charge_complete), dipasangkan via id(tr). ---
side = {}
records = []   # (recent_rec_count, disp_estwait, wait_aktual, delta_w)

orig_decide = MasterEVPPORolloutAgent.on_decision
def patched_decide(self, user, chosen_spklu_id, recs, feasible_spklus):
    if self._pending is not None:
        tr = self._pending[0]
        recent_rec_count = self._pending[5]
        side[id(tr)] = recent_rec_count
    return orig_decide(self, user, chosen_spklu_id, recs, feasible_spklus)
MasterEVPPORolloutAgent.on_decision = patched_decide

orig_complete = MasterEVPPORolloutAgent.on_charge_complete
def patched_complete(self, user):
    tr = self._user_trip_tr.get(user.user_id)
    if tr is not None and tr.complied:
        rrc = side.pop(id(tr), None)
        if rrc is not None:
            dw = user.wait_time - tr.disp_estwait
            records.append((rrc, tr.disp_estwait, user.wait_time, dw))
    return orig_complete(self, user)
MasterEVPPORolloutAgent.on_charge_complete = patched_complete

print(f"Checkpoint: {CKPT}")
print("Menjalankan 5 seed evaluasi (forecaster=VWF, k=3, pref=False)...", flush=True)
for s in range(5):
    sim = common.fresh_sim(DATASET)
    random.seed(s); np.random.seed(s)
    agent = MasterEVPPOInferenceAgent(pol, sim, VirtualWaitForecaster(), k=3)
    sim.run(max_steps=sim.max_steps, agent=agent)
    print(f"  seed={s} selesai, kumulatif n_trip_patuh={len(records)}", flush=True)

MasterEVPPORolloutAgent.on_decision = orig_decide
MasterEVPPORolloutAgent.on_charge_complete = orig_complete

rrc = np.array([r[0] for r in records], dtype=float)
disp = np.array([r[1] for r in records], dtype=float)
wait_akt = np.array([r[2] for r in records], dtype=float)
dw = np.array([r[3] for r in records], dtype=float)

print(f"\nN trip patuh dgn data lengkap: {len(records)}")
print(f"rec_activity saat keputusan: mean={rrc.mean():.2f} median={np.median(rrc):.1f} "
     f"p90={np.percentile(rrc,90):.1f} max={rrc.max():.0f}")
print(f"Delta W (wait_aktual - disp_estwait): mean={dw.mean():.2f} median={np.median(dw):.2f} "
     f"sd={dw.std():.2f} p90={np.percentile(dw,90):.2f}")

from scipy.stats import spearmanr, pearsonr
rho, p_rho = spearmanr(rrc, dw)
r, p_r = pearsonr(rrc, dw)
print(f"\nSpearman rho(rec_activity, Delta W) = {rho:.4f}  (p={p_rho:.2e})")
print(f"Pearson  r  (rec_activity, Delta W) = {r:.4f}  (p={p_r:.2e})")

# Cek juga versi biner: rec_activity tinggi (>p75) vs rendah (<=p25)
p25, p75 = np.percentile(rrc, [25, 75])
low = dw[rrc <= p25]
high = dw[rrc >= p75]
print(f"\nDelta W saat rec_activity RENDAH (<=p25={p25:.1f}): mean={low.mean():.2f} (n={len(low)})")
print(f"Delta W saat rec_activity TINGGI (>=p75={p75:.1f}): mean={high.mean():.2f} (n={len(high)})")
from scipy.stats import mannwhitneyu
if len(low) > 0 and len(high) > 0:
    u = mannwhitneyu(high, low, alternative="greater")
    print(f"Mann-Whitney (tinggi > rendah): p={u.pvalue:.2e}")

common.save_json(dict(
    n=len(records), rec_activity=rrc.tolist(), disp_estwait=disp.tolist(),
    wait_aktual=wait_akt.tolist(), delta_w=dw.tolist(),
    spearman_rho=float(rho), spearman_p=float(p_rho),
    pearson_r=float(r), pearson_p=float(p_r),
    dw_mean_low_rec_activity=float(low.mean()) if len(low) else None,
    dw_mean_high_rec_activity=float(high.mean()) if len(high) else None,
), "diagnosis_rec_activity_vs_deltaW.json")
print("\nOK -- hasil ditulis ke outputs/diagnosis_rec_activity_vs_deltaW.json")
