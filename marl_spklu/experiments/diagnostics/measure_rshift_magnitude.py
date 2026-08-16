"""Analisis #1 (diminta user): ukur distribusi magnitudo r_shift vs r_rec per-keputusan
di bawah lingkungan keputusan PDQN sesungguhnya (binary_recommendation_mode, mu_hat nyata),
utk menjawab apakah suku r_shift "tenggelam" krn jarang tak-nol / terlalu kecil dibanding
r_rec -- salah satu dugaan kenapa personalisasi tak tereksploitasi (§3.9 laporan).

Dijalankan dgn agen GreedyAgent(mode="utilization") sbg proksi "rekomendasi yg mendekati
kebijakan mendekati-optimal PDQN thd r_rec" (bukan PDQN sungguhan -- lebih murah & cukup
utk mengukur STATISTIK PERILAKU PENGGUNA yg dipicu campuran mu_hat, yg jadi sumber r_shift).

Pemakaian:
    python -m marl_spklu.experiments.diagnostics.measure_rshift_magnitude
"""
import random

import numpy as np

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.env.simulator import Simulator
from marl_spklu.experiments import harness
from marl_spklu.experiments.ablations import binary_recommendation_mode, constant_trust
from marl_spklu.experiments.diagnostics.alignment_test_reward_candidates import record_decisions

DATASET = "scenario_dataset_7d.json"


def measure(mu_hat: float, stochastic: bool = True, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    with binary_recommendation_mode(mu_hat, stochastic), constant_trust(mu_hat):
        sim = Simulator({}, [], None, user_willingness_radius_km=None,
                        user_willingness_ratio=harness.DEFAULT_WILLINGNESS_RATIO)
        sim.load_from_dataset(DATASET)
        max_steps = getattr(sim, "max_steps", max(sim.spawn_schedule) + 1)
        sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=max_steps)
        agent = GreedyAgent(mode="utilization")

        records = []
        with record_decisions(sim, records):
            sim.run(max_steps=max_steps, agent=agent)

    r_recs, r_shifts = [], []
    n_shift_nonzero = 0
    n_complied = 0
    for rec in records:
        u = rec["utils"]
        feas = rec["feasible_idx"]
        a_hat, default, chosen = rec["a_hat_idx"], rec["default_idx"], rec["chosen_idx"]
        u_feas = u[feas]
        r_rec = float(u_feas.mean() - u[a_hat])
        r_shift = float(u[default] - u[chosen])
        r_recs.append(r_rec)
        r_shifts.append(r_shift)
        if abs(r_shift) > 1e-9:
            n_shift_nonzero += 1
        if chosen == a_hat:
            n_complied += 1

    r_recs = np.array(r_recs)
    r_shifts = np.array(r_shifts)
    n = len(records)
    print(f"\n{'='*70}\nmu_hat={mu_hat}  stochastic={stochastic}  n_keputusan={n}\n{'='*70}")
    print(f"Kepatuhan (chosen == a_hat)     : {n_complied/n:.1%}")
    print(f"r_shift != 0 (pengguna BERGESER dari default menuju stasiun BUKAN default)")
    print(f"                                 : {n_shift_nonzero/n:.1%} dari keputusan")
    print(f"|r_rec|   mean={np.abs(r_recs).mean():.4f}  std={r_recs.std():.4f}  "
         f"[p10={np.percentile(r_recs,10):.4f}, p90={np.percentile(r_recs,90):.4f}]")
    print(f"|r_shift| mean={np.abs(r_shifts).mean():.4f}  std={r_shifts.std():.4f}  "
         f"[p10={np.percentile(r_shifts,10):.4f}, p90={np.percentile(r_shifts,90):.4f}]")
    print(f"Rasio skala mean|r_shift| / mean|r_rec| = "
         f"{np.abs(r_shifts).mean() / max(np.abs(r_recs).mean(), 1e-9):.3f}")
    # Hanya di antara keputusan dgn r_shift != 0 -- brp besar efeknya SAAT terjadi
    nz = np.abs(r_shifts) > 1e-9
    if nz.any():
        print(f"|r_shift| (hanya saat != 0) mean={np.abs(r_shifts[nz]).mean():.4f}  "
             f"n={nz.sum()}")
    return dict(mu_hat=mu_hat, n=n, compliance=n_complied/n, frac_shift_nonzero=n_shift_nonzero/n,
               mean_abs_rrec=float(np.abs(r_recs).mean()), mean_abs_rshift=float(np.abs(r_shifts).mean()))


def main():
    results = [measure(mu, stochastic=True) for mu in (0.2, 0.5, 0.8)]
    print(f"\n{'='*70}\nRINGKASAN\n{'='*70}")
    for r in results:
        print(f"mu={r['mu_hat']}: kepatuhan={r['compliance']:.1%}  "
             f"frac_shift_nonzero={r['frac_shift_nonzero']:.1%}  "
             f"mean|r_rec|={r['mean_abs_rrec']:.4f}  mean|r_shift|={r['mean_abs_rshift']:.4f}  "
             f"rasio={r['mean_abs_rshift']/max(r['mean_abs_rrec'],1e-9):.3f}")


if __name__ == "__main__":
    main()
