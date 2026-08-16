"""Instrumentasi diagnostik utk menguji 2 hipotesis mekanisme (dibahas sesi kerja):
kenapa H-PPO `preferensi_murni` (beta_prox MURNI, tanpa alpha_gini/alpha_flock) bisa
mengalahkan greedy di Gini MESKI tak pernah diberi tahu soal pemerataan sama sekali.

Hipotesis 1 (heterogenitas populasi -> sebaran alami): distribusi rekomendasi
preferensi_murni per-stasiun LEBIH RATA drpd greedy, krn preferensi tiap pengguna
berbeda-beda (bukan semua mengejar "yg tersepi saat ini" spt greedy).

Hipotesis 2 (penalti kemacetan implisit via P_rec): stasiun yg SEDANG PADAT saat
direkomendasikan (antrean/rec_activity tinggi) berkorelasi dgn PROX LEBIH RENDAH
(krn P_rec = softmax(-gamma*wait) menekan kepatuhan saat wait tinggi -> pengguna
lari ke P_pref -> pilihan aktual beda dari rekomendasi -> Prox turun) -- MESKI
reward tak pernah eksplisit menghukum kemacetan.

DiagnosticWrapper generik: bekerja utk AGEN APA PUN (Greedy/H-PPO/PDQN) yang punya
get_recommendation + predict_waits. Independen dari state internal agen -- tak
mengubah kode training/inferensi manapun, murni instrumentasi tambahan di lapisan
luar (dipasang saat evaluasi/analisis, bukan bagian pipeline utama)."""
import sys, os, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
import torch
import common
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.experiments.ablations import constant_trust_shadow
from marl_spklu.agents.greedy_agent import GreedyAgent


class VirtualWaitForecaster(ForecasterBase):
    def predict(self, spklus, time_now_min=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {sid: 0.0 for sid in spklus}
        return {sid: float(sim.compute_virtual_wait(user, s, time_now_min)) for sid, s in spklus.items()}


class DiagnosticWrapper:
    """Bungkus AGEN APA PUN, catat per-keputusan: stasiun PRIMER direkomendasikan,
    antrean & rec_activity stasiun itu SAAT direkomendasikan (sebelum tahu pilihan
    aktual), kepatuhan, dan skor Prox aktual (feat rekomendasi vs feat pilihan nyata,
    formula SAMA PERSIS `RLRolloutAgent._station_feat`/`RewardCalculator.prox`)."""

    def __init__(self, base_agent, sim):
        self.base = base_agent
        self.sim = sim
        self.rc = RewardCalculator()  # cuma dipakai fungsi prox(), bobot tak relevan
        self.pos_scale = max(1e-6, float(np.std(
            [s.location[0] for s in sim.spklus.values()] +
            [s.location[1] for s in sim.spklus.values()])))
        self.wait_scale = 60.0
        self.log = []
        self._pending = None  # (recs, wait_hat, queue_primary, rec_activity_primary)

    def _station_feat(self, sid, wait_hat):
        feat = self.sim.spklu_features.get(sid, {})
        loc = feat.get('loc', (0.0, 0.0))
        conn = feat.get('conn', 1.0)
        w = wait_hat.get(sid, 0.0)
        return np.array([loc[0] / self.pos_scale, loc[1] / self.pos_scale,
                         conn, w / self.wait_scale])

    def get_recommendation(self, feasible_spklus):
        recs = self.base.get_recommendation(feasible_spklus)
        if recs:
            primary = recs[0]
            queue_primary = self.sim.spklus[primary].get_queue_length()
            recact_primary = self.sim.recent_recs.get(primary, 0)
            self._pending = (recs, queue_primary, recact_primary)
        else:
            self._pending = None
        return recs

    def predict_waits(self, feasible_spklus):
        if hasattr(self.base, "predict_waits"):
            return self.base.predict_waits(feasible_spklus)
        return {s: 0.0 for s in feasible_spklus}

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        if hasattr(self.base, "on_decision"):
            self.base.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        if self._pending is None:
            return
        _, queue_primary, recact_primary = self._pending
        primary = recs[0] if recs else None
        complied = bool(chosen_spklu_id in set(recs))
        wait_hat = self.sim.get_spklu_wait_estimates()
        if primary is not None:
            feat_rec = self._station_feat(primary, wait_hat)
            feat_chosen = self._station_feat(chosen_spklu_id, wait_hat)
            prox_value = float(self.rc.prox(feat_rec, feat_chosen))
        else:
            prox_value = None
        self.log.append(dict(
            user=user.user_id, primary=primary, chosen=chosen_spklu_id,
            complied=complied, prox=prox_value,
            queue_primary=queue_primary, rec_activity_primary=recact_primary,
        ))
        self._pending = None


def run_diagnostic(agent_factory, ds, trust_value, seed=0, label=""):
    """agent_factory(sim) -> agen dasar (Greedy / InferenceAgent H-PPO / dst)."""
    with constant_trust_shadow(value=trust_value):
        sim = common.fresh_sim(ds)
        random.seed(seed); np.random.seed(seed)
        base = agent_factory(sim)
        wrapped = DiagnosticWrapper(base, sim)
        sim.run(max_steps=sim.max_steps, agent=wrapped)
    served = {sid: sp.total_served for sid, sp in sim.spklus.items()}
    return dict(label=label, distribusi=served, gini=common.gini(
        np.array(list(served.values()), dtype=float)), log=wrapped.log)


def analisis_hipotesis(hasil: dict):
    """Ringkas log mentah -> statistik utk Hipotesis 1 (sebaran) & 2 (korelasi
    kemacetan-vs-prox), tanpa menyimpan log mentah penuh (bisa besar)."""
    log = hasil["log"]
    served = np.array(list(hasil["distribusi"].values()), dtype=float)
    ringkas = dict(
        label=hasil["label"], gini=hasil["gini"],
        distribusi=hasil["distribusi"],
        # Hipotesis 1: rasio max/min & std/mean (koefisien variasi) -- makin rendah,
        # makin rata sebarannya.
        max_min_ratio=float(served.max() / max(served.min(), 1.0)),
        cv_distribusi=float(served.std() / max(served.mean(), 1e-9)),
        n_keputusan=len(log),
    )
    if log:
        queues = np.array([L["queue_primary"] for L in log], dtype=float)
        recacts = np.array([L["rec_activity_primary"] for L in log], dtype=float)
        proxs = np.array([L["prox"] for L in log if L["prox"] is not None], dtype=float)
        complied = np.array([L["complied"] for L in log], dtype=bool)
        # Hipotesis 2: korelasi kemacetan (queue/rec_activity) stasiun primer SAAT
        # direkomendasikan dgn prox aktual & tingkat kepatuhan.
        if len(proxs) > 1 and queues.std() > 0:
            ringkas["korelasi_queue_vs_prox"] = float(np.corrcoef(queues, proxs)[0, 1])
        if len(proxs) > 1 and recacts.std() > 0:
            ringkas["korelasi_recact_vs_prox"] = float(np.corrcoef(recacts, proxs)[0, 1])
        ringkas["acceptance_keseluruhan"] = float(complied.mean())
        ringkas["prox_mean"] = float(proxs.mean()) if len(proxs) else None
        # Kepatuhan dibelah 2 kelompok: queue primer di atas/bawah median, bandingkan
        # tingkat kepatuhan -- Hipotesis 2 memprediksi kepatuhan LEBIH RENDAH saat
        # queue primer tinggi (krn P_rec turun akibat wait naik).
        med_q = float(np.median(queues))
        mask_tinggi = queues > med_q
        if mask_tinggi.any() and (~mask_tinggi).any():
            ringkas["acceptance_queue_tinggi"] = float(complied[mask_tinggi].mean())
            ringkas["acceptance_queue_rendah"] = float(complied[~mask_tinggi].mean())
    return ringkas


if __name__ == "__main__":
    ds = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
    TRUST = 0.5
    hasil_semua = {}

    # --- Greedy (baseline pembanding, tak butuh checkpoint) ---
    for mode in ["queue", "utilization"]:
        r = run_diagnostic(lambda sim, m=mode: GreedyAgent(mode=m, top_k=2), ds, TRUST,
                           label=f"greedy_{mode}")
        hasil_semua[f"greedy_{mode}"] = analisis_hipotesis(r)
        print(f"greedy_{mode}:", json.dumps(hasil_semua[f"greedy_{mode}"], indent=1,
                                            default=str)[:600])

    # --- H-PPO preferensi_murni (CALIBRATED) -- HANYA jalan kalau checkpoint sudah ada ---
    ckpt = os.path.join(common.OUTDIR, "hppo_t0.5_preferensi_murni_CALIBRATED_seed0.pt")
    meta_path = os.path.join(common.OUTDIR, "hppo_t0.5_preferensi_murni_CALIBRATED_seed0_meta.json")
    if os.path.exists(ckpt) and os.path.exists(meta_path):
        from marl_spklu.rl.policy import HPPOPolicy
        from marl_spklu.rl.rollout import InferenceAgent
        meta = json.load(open(meta_path))
        policy = HPPOPolicy(meta["obs_dim"], meta["critic_obs_dim"], meta["N"])
        policy.load_state_dict(torch.load(ckpt))
        policy.eval()
        r = run_diagnostic(
            lambda sim: InferenceAgent(policy, sim, forecaster=VirtualWaitForecaster(),
                                       k=2, epsilon=0.0),
            ds, TRUST, label="hppo_preferensi_murni")
        hasil_semua["hppo_preferensi_murni"] = analisis_hipotesis(r)
        print("hppo_preferensi_murni:", json.dumps(
            hasil_semua["hppo_preferensi_murni"], indent=1, default=str)[:600])
    else:
        print(f"[LEWATI] checkpoint belum ada: {ckpt}")

    common.save_json(hasil_semua, "analisis_mekanisme_preferensi_hasil.json")
    print("SAVED -> analisis_mekanisme_preferensi_hasil.json")
