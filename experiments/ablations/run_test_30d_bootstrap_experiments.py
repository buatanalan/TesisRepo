import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Uji hipotesis "lingkaran setan trust" (dibahas di percakapan): agen tak bisa memeratakan
karena weight_ai = trust*w_i tetap kecil sepanjang horizon (trust hanya naik pelan dari
pengalaman kepatuhan, dan kepatuhan butuh rekomendasi sesuai preferensi -- yang justru
anti-pemerataan). Dua intervensi diuji terpisah & gabungan, 30 hari/seed=0, dataset & arsitektur
identik dgn run_test_30d_single_seed.py (FormulaForecaster, reward default sbg baseline):

  1. INIT TRUST TINGGI  : trust semua user di-set 1.0 di awal (bukan 0.5) -- melepas bottleneck
     "weight_ai kecil di awal" secara langsung, tanpa menyentuh mekanisme belajar trust itu sendiri.
  2. FAIRNESS LONGGAR    : alpha_gini=alpha_flock=0 -- agen bebas mengejar preferensi/wait-
     improvement tanpa dihukum herding/Gini, utk lihat apakah acceptance melonjak mendekati
     batas teoritis (~79%, dihitung dari ceiling penebak 73.6% & weight_ai rata-rata).
  3. KOMBINASI           : keduanya sekaligus.

Metrik yang dibandingkan: reward, acceptance, trust, ENTROPY kebijakan (apakah mulai
terkonsentrasi begitu insentif berubah), Gini SERVED (fairness aktual -- bukan cuma reward
proxy), herding_events, total served.
"""
import json
from collections import defaultdict

import numpy as np

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator

DATASET = "scenario_dataset.json"
SEED = 0
CHUNK = 96
N_UPDATES = 30

VARIANTS = {
    "baseline":       {"init_trust": None, "alpha_gini": 0.5, "alpha_flock": 0.3},
    "high_trust":     {"init_trust": 1.0,  "alpha_gini": 0.5, "alpha_flock": 0.3},
    "loose_fairness": {"init_trust": None, "alpha_gini": 0.0, "alpha_flock": 0.0},
    "combined":       {"init_trust": 1.0,  "alpha_gini": 0.0, "alpha_flock": 0.0},
}


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def run_variant(name, init_trust, alpha_gini, alpha_flock):
    print(f"\n{'=' * 70}\n=== VARIAN: {name} (init_trust={init_trust}, "
         f"alpha_gini={alpha_gini}, alpha_flock={alpha_flock}) ===\n{'=' * 70}")

    rc = RewardCalculator(alpha_gini=alpha_gini, alpha_flock=alpha_flock)
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, reward_calc=rc, honest_estwait=True)
    forecaster = FormulaForecaster()
    sim = tr._fresh_sim()
    if init_trust is not None:
        for u in sim.users:
            u.trust = float(init_trust)

    agent = RLRolloutAgent(tr.policy, sim, tr.rc, forecaster, tr.k, honest_estwait=tr.honest_estwait)
    sids = list(sim.spklus.keys())

    day_records = []
    session_log = []
    prev_sim_logs_len = 0
    step = 0

    for day in range(N_UPDATES):
        boundary = False
        for _ in range(CHUNK):
            sim.step_once(step, agent=agent)
            step += 1
            if step >= sim.max_steps:
                boundary = True
                break
        if boundary:
            for t in agent.transitions:
                t.resolved = True
        resolved = [t for t in agent.transitions if t.resolved]
        pending = [t for t in agent.transitions if not t.resolved]

        new_sessions = sim.logs[prev_sim_logs_len:]
        prev_sim_logs_len = len(sim.logs)
        for s in new_sessions:
            session_log.append({"day": day, **s})

        if resolved:
            stats = tr.ppo.update(resolved)
            rewards = np.array([t.reward for t in resolved])
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            trust_vals = [u.trust for u in sim.users]
            day_rec = {
                "day": day,
                "mean_reward": float(rewards.mean()),
                "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                "mean_flock_penalty": float(np.mean([t.flock_penalty for t in resolved])),
                "mean_trust_allusers": float(np.mean(trust_vals)),
                "gini_served_cum": _gini(served),
                "total_served_cum": int(served.sum()),
                "entropy_final": stats.get("entropy_final", 0.0),
                "approx_kl": stats.get("approx_kl", 0.0),
                "explained_var": stats.get("explained_var", 0.0),
            }
            day_records.append(day_rec)
            print(f"  [Hari {day + 1:02d}/{N_UPDATES}] R={day_rec['mean_reward']:+.4f} "
                 f"accept={day_rec['acceptance_rate']:.2f} trust={day_rec['mean_trust_allusers']:.3f} "
                 f"gini_served={day_rec['gini_served_cum']:.3f} ent={day_rec['entropy_final']:.2f} "
                 f"served={day_rec['total_served_cum']}")

        if boundary:
            break
        agent.transitions = pending

    n_report = min(5, len(day_records))
    served_final = np.array([s.total_served for s in sim.spklus.values()], float)

    result = {
        "variant": name, "init_trust": init_trust, "alpha_gini": alpha_gini, "alpha_flock": alpha_flock,
        "reward_first%d" % n_report: float(np.mean([d["mean_reward"] for d in day_records[:n_report]])),
        "reward_last%d" % n_report: float(np.mean([d["mean_reward"] for d in day_records[-n_report:]])),
        "acceptance_first%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[:n_report]])),
        "acceptance_last%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[-n_report:]])),
        "trust_first%d" % n_report: float(np.mean([d["mean_trust_allusers"] for d in day_records[:n_report]])),
        "trust_last%d" % n_report: float(np.mean([d["mean_trust_allusers"] for d in day_records[-n_report:]])),
        "entropy_first%d" % n_report: float(np.mean([d["entropy_final"] for d in day_records[:n_report]])),
        "entropy_last%d" % n_report: float(np.mean([d["entropy_final"] for d in day_records[-n_report:]])),
        "gini_served_final": _gini(served_final),
        "herding_events": sim.herding_events,
        "total_served_final": int(served_final.sum()),
        "day_records": day_records,
    }
    return result


def main():
    all_results = {}
    for name, cfg in VARIANTS.items():
        all_results[name] = run_variant(name, cfg["init_trust"], cfg["alpha_gini"], cfg["alpha_flock"])

    with open("test_30d_bootstrap_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'=' * 90}\n=== TABEL PERBANDINGAN AKHIR ===\n{'=' * 90}")
    cols = ["acceptance_last5", "trust_last5", "entropy_last5", "gini_served_final",
           "herding_events", "total_served_final", "reward_last5"]
    header = f"{'variant':<18}" + "".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        row = f"{name:<18}"
        for c in cols:
            v = r.get(c)
            row += f"{v:>18.4f}" if isinstance(v, float) else f"{str(v):>18}"
        print(row)

    print("\n[INFO] Ringkasan lengkap -> test_30d_bootstrap_experiments_summary.json")


if __name__ == "__main__":
    main()
