"""Uji apakah temuan "Gini baik <-> waktu tunggu buruk" KOKOH atau sekadar derau.

Dua sumber derau dipisahkan:
  (a) DERAU ANTAR-SEED  -> jalankan eval yg SAMA dgn 5 seed lingkungan berbeda
      (kebijakan DETERMINISTIK/epsilon=0; yg berubah hanya stokastisitas lingkungan:
      sampling pilihan pengguna, dsb). Kalau selisih antar-METODE >> sebaran antar-SEED,
      temuan kokoh.
  (b) EKOR DISTRIBUSI   -> rata-rata bisa didominasi segelintir pengguna yg menunggu
      sangat lama. Karena itu median & p90 dilaporkan juga, plus jumlah RENEGE
      (pengguna menyerah krn melampaui patience) yg justru MEMBUANG tunggu panjang dari
      catatan (bias ke bawah).

Data per-sesi diambil dari `sim.logs` (dicatat saat sesi charging dimulai, memuat
`wait_time` aktual per pengguna) -- bukan agregat per-stasiun, shg distribusinya utuh."""
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
SEEDS = [0, 1, 2, 3, 4]


class VirtualWaitForecaster(ForecasterBase):
    def predict(self, spklus, time_now_min=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {sid: 0.0 for sid in spklus}
        return {sid: float(sim.compute_virtual_wait(user, s, time_now_min))
                for sid, s in spklus.items()}


def run_once(agent_factory, ds, seed, trust=0.5):
    with constant_trust_shadow(value=trust):
        sim = common.fresh_sim(ds)
        random.seed(seed); np.random.seed(seed)
        agent = agent_factory(sim)
        sim.run(max_steps=sim.max_steps, agent=agent)

    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
    w = np.array([L["wait_time"] for L in sim.logs], dtype=float)
    reneged = sum(1 for u in sim.users if getattr(u, "reneged_count", 0))
    return dict(
        gini_served=float(common.gini(served)),
        mean_wait=float(w.mean()) if w.size else 0.0,
        median_wait=float(np.median(w)) if w.size else 0.0,
        p90_wait=float(np.percentile(w, 90)) if w.size else 0.0,
        p99_wait=float(np.percentile(w, 99)) if w.size else 0.0,
        frac_wait_zero=float((w <= 1e-9).mean()) if w.size else 0.0,
        n_sesi=int(w.size),
        total_served=int(served.sum()),
        n_user_reneged=int(reneged),
    )


def build_factories(ds):
    fc = VirtualWaitForecaster()
    F = {}
    for mode in ["queue", "utilization"]:
        F[f"greedy_{mode}"] = (lambda sim, m=mode: GreedyAgent(mode=m, top_k=2))

    from marl_spklu.rl.policy import HPPOPolicy
    from marl_spklu.rl.p_ppo_policy import PPPOPolicy
    from marl_spklu.rl.rollout import InferenceAgent
    for label, stem, cls in [("hppo_original", "hppo_t0.5_gabungan_seed0", HPPOPolicy),
                             ("pppo_gated", "pppo_gated_t0.5_gabungan_seed0", PPPOPolicy)]:
        meta = json.load(open(os.path.join(OUT, stem + "_meta.json")))
        pol = cls(meta["obs_dim"], meta["critic_obs_dim"], meta["N"])
        pol.load_state_dict(torch.load(os.path.join(OUT, stem + ".pt")))
        pol.eval()
        F[label] = (lambda sim, p=pol: InferenceAgent(p, sim, fc, k=2, epsilon=0.0,
                                                      threshold=0.20))

    from marl_spklu.rl.pdqn_advanced_policy import PDQNQMixQNetwork
    from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent
    stem = "pdqn_qmix_t0.5_gabungan_seed0"
    meta = json.load(open(os.path.join(OUT, stem + "_meta.json")))
    q = PDQNQMixQNetwork(meta["obs_dim"], meta["N"], n_types=meta["n_types"],
                         use_preference=meta["use_preference"],
                         pref_feature_mode=meta["pref_feature_mode"])
    q.load_state_dict(torch.load(os.path.join(OUT, stem + ".pt")))
    q.eval()

    def mk_qmix(sim, qq=q, pfm=meta["pref_feature_mode"]):
        a = PDQNInferenceAgent(qq, forecaster=fc, pref_feature_mode=pfm)
        a.bind_to_sim(sim)
        return a
    F["pdqn_qmix"] = mk_qmix
    return F


def main():
    ds = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
    F = build_factories(ds)
    hasil = {}
    for label, fac in F.items():
        runs = []
        for sd in SEEDS:
            r = run_once(fac, ds, sd)
            runs.append(r)
            print(f"  {label} seed={sd} gini={r['gini_served']:.4f} "
                  f"mean={r['mean_wait']:.2f} med={r['median_wait']:.2f} "
                  f"p90={r['p90_wait']:.2f} nol={r['frac_wait_zero']:.2f}", flush=True)
        hasil[label] = runs

    print("\n=== RINGKASAN (rata-rata +/- simpangan baku atas %d seed) ===" % len(SEEDS))
    hdr = f"{'metode':<20}{'gini':>16}{'mean_wait':>18}{'median':>14}{'p90':>14}"
    print(hdr); print("-" * len(hdr))
    ring = {}
    for label, runs in hasil.items():
        def ms(k):
            a = np.array([r[k] for r in runs], dtype=float)
            return a.mean(), a.std()
        g, gs = ms("gini_served"); m, msd = ms("mean_wait")
        md, mds = ms("median_wait"); p9, p9s = ms("p90_wait")
        ring[label] = dict(gini=[g, gs], mean_wait=[m, msd],
                           median_wait=[md, mds], p90_wait=[p9, p9s])
        print(f"{label:<20}{g:>8.4f}±{gs:<7.4f}{m:>9.2f}±{msd:<8.2f}"
              f"{md:>7.2f}±{mds:<6.2f}{p9:>7.2f}±{p9s:<6.2f}")

    common.save_json(dict(seeds=SEEDS, ringkasan=ring, mentah=hasil),
                     "uji_derau_waktu_tunggu.json")
    print("\nSAVED -> uji_derau_waktu_tunggu.json")


if __name__ == "__main__":
    main()
