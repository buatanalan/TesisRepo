"""Uji empiris klaim: "pemerataan utilisasi & waktu tunggu berkorelasi positif secara
tidak langsung" (yakni Gini rendah -> wait rendah).

Dasar teori: waktu tunggu antrean KONVEKS thd utilisasi rho (meledak saat rho->1). Dgn
permintaan total tetap, ketaksamaan Jensen: sum_i W(rho_i) MINIMUM saat semua rho_i sama.
Jadi utk stasiun homogen, pemerataan = minimasi total tunggu -> keduanya sejajar.

Uji ini memakai SELURUH checkpoint Tahap 2 yg sudah dilatih (rentang Gini 0,037-0,19) sbg
titik data, mengukur Gini DAN rata-rata waktu tunggu pada eval bersih yg SAMA, lalu
menghitung korelasi Pearson & Spearman.

Semua eval: trust=0,5 (constant_trust_shadow), 30 hari penuh, epsilon=0, seed=0 --
identik dgn protokol eval Tahap 2 lainnya."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.experiments.ablations import constant_trust_shadow
from marl_spklu.agents.greedy_agent import GreedyAgent

OUT = common.OUTDIR


class VirtualWaitForecaster(ForecasterBase):
    def predict(self, spklus, time_now_min=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {sid: 0.0 for sid in spklus}
        return {sid: float(sim.compute_virtual_wait(user, s, time_now_min))
                for sid, s in spklus.items()}


def run_eval(agent_factory, label, ds, trust=0.5, seed=0):
    with constant_trust_shadow(value=trust):
        sim = common.fresh_sim(ds)
        random.seed(seed); np.random.seed(seed)
        agent = agent_factory(sim)
        sim.run(max_steps=sim.max_steps, agent=agent)

    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
    waits = np.array([s.total_wait_time for s in sim.spklus.values()], dtype=float)
    utils = np.array([s.get_utilization() for s in sim.spklus.values()], dtype=float)
    total_served = float(served.sum())
    return dict(
        label=label,
        gini_served=float(common.gini(served)),
        gini_util=float(common.gini(utils)),
        mean_wait=float(waits.sum() / max(total_served, 1.0)),
        total_wait=float(waits.sum()),
        total_served=int(total_served),
        served=served.tolist(),
    )


def main():
    ds = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
    fc = VirtualWaitForecaster()
    results = []

    # ---- Greedy (tanpa checkpoint) ----
    for mode in ["queue", "utilization"]:
        results.append(run_eval(
            lambda sim, m=mode: GreedyAgent(mode=m, top_k=2), f"greedy_{mode}", ds))
        print(results[-1]["label"], results[-1]["gini_served"], results[-1]["mean_wait"], flush=True)

    # ---- Keluarga PPO (HPPOPolicy / PPPOPolicy) ----
    from marl_spklu.rl.policy import HPPOPolicy
    from marl_spklu.rl.p_ppo_policy import PPPOPolicy
    from marl_spklu.rl.rollout import InferenceAgent

    ppo_specs = [
        ("hppo_original", "hppo_t0.5_gabungan_seed0", HPPOPolicy, None),
        ("pppo_gated", "pppo_gated_t0.5_gabungan_seed0", PPPOPolicy, None),
        # v1 dilatih SEBELUM gate ada -> state_dict tanpa `pref_gate`; muat non-strict
        # lalu set gate=1.0 utk mereproduksi perilaku aslinya (kontribusi penuh).
        ("pppo_naif_v1", "pppo_t0.5_gabungan_seed0", PPPOPolicy, 1.0),
    ]
    for label, stem, cls, gate in ppo_specs:
        meta_p = os.path.join(OUT, stem + "_meta.json")
        ckpt_p = os.path.join(OUT, stem + ".pt")
        if not (os.path.exists(meta_p) and os.path.exists(ckpt_p)):
            print("[LEWATI]", label, flush=True); continue
        meta = json.load(open(meta_p))
        pol = cls(meta["obs_dim"], meta["critic_obs_dim"], meta["N"])
        sd = torch.load(ckpt_p)
        missing = pol.load_state_dict(sd, strict=(gate is None))
        if gate is not None:
            with torch.no_grad():
                pol.pref_gate.fill_(float(gate))
            print(f"   ({label}: muat non-strict, gate di-set {gate}; missing={missing})", flush=True)
        pol.eval()
        results.append(run_eval(
            lambda sim, p=pol: InferenceAgent(p, sim, fc, k=2, epsilon=0.0, threshold=0.20),
            label, ds))
        print(results[-1]["label"], results[-1]["gini_served"], results[-1]["mean_wait"], flush=True)

    # ---- Keluarga PDQN ----
    from marl_spklu.rl.pdqn_policy import PDQNQNetwork
    from marl_spklu.rl.pdqn_advanced_policy import PDQNAdvancedQNetwork, PDQNQMixQNetwork
    from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent

    pdqn_specs = [
        ("pdqn_orig", "pdqn_t0.5_gabungan_seed0", PDQNQNetwork),
        ("pdqn_gabungan_v2", "pdqn_t0.5_gabungan_v2_seed0", PDQNQNetwork),
        ("pdqn_gabungan_v3", "pdqn_t0.5_gabungan_v3_seed0", PDQNQNetwork),
        ("pdqn_preferensi_murni", "pdqn_t0.5_preferensi_murni_seed0", PDQNQNetwork),
        ("pdqn_advanced", "pdqn_advanced_t0.5_gabungan_seed0", PDQNAdvancedQNetwork),
        ("pdqn_qmix", "pdqn_qmix_t0.5_gabungan_seed0", PDQNQMixQNetwork),
    ]
    for label, stem, cls in pdqn_specs:
        meta_p = os.path.join(OUT, stem + "_meta.json")
        ckpt_p = os.path.join(OUT, stem + ".pt")
        if not (os.path.exists(meta_p) and os.path.exists(ckpt_p)):
            print("[LEWATI]", label, flush=True); continue
        meta = json.load(open(meta_p))
        q = cls(meta["obs_dim"], meta["N"], n_types=meta["n_types"],
                use_preference=meta["use_preference"],
                pref_feature_mode=meta["pref_feature_mode"])
        q.load_state_dict(torch.load(ckpt_p))
        q.eval()

        def mk(sim, qq=q, pfm=meta["pref_feature_mode"]):
            a = PDQNInferenceAgent(qq, forecaster=fc, pref_feature_mode=pfm)
            a.bind_to_sim(sim)
            return a
        results.append(run_eval(mk, label, ds))
        print(results[-1]["label"], results[-1]["gini_served"], results[-1]["mean_wait"], flush=True)

    # ---- Korelasi ----
    g = np.array([r["gini_served"] for r in results])
    gu = np.array([r["gini_util"] for r in results])
    w = np.array([r["mean_wait"] for r in results])
    n = len(results)

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    summary = dict(
        n_titik=n,
        pearson_gini_served_vs_wait=pearson(g, w),
        spearman_gini_served_vs_wait=spearman(g, w),
        pearson_gini_util_vs_wait=pearson(gu, w),
        spearman_gini_util_vs_wait=spearman(gu, w),
        rentang_gini=[float(g.min()), float(g.max())],
        rentang_wait=[float(w.min()), float(w.max())],
    )
    print("\n=== RINGKASAN ===")
    print(json.dumps(summary, indent=1))
    print("\n=== TITIK DATA (urut Gini) ===")
    for r in sorted(results, key=lambda r: r["gini_served"]):
        print(f"{r['label']:24s} gini={r['gini_served']:.4f} gini_util={r['gini_util']:.4f} "
              f"wait={r['mean_wait']:7.2f}m served={r['total_served']}")

    common.save_json(dict(summary=summary, results=results),
                     "uji_korelasi_gini_wait.json")
    print("\nSAVED -> uji_korelasi_gini_wait.json")


if __name__ == "__main__":
    main()
