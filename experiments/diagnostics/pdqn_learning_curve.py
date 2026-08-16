import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Kurva belajar PDQN: latih secara bertahap, evaluasi Gini di tiap checkpoint, untuk
menentukan berapa PASS training yang benar-benar dibutuhkan sebelum kinerja mendatar.

`loss` dan `q_mean` di log training TIDAK cukup untuk menjawab ini: keduanya bisa mendatar
sementara kebijakan masih membaik (atau sebaliknya, q_mean menanjak karena overestimasi
tanpa kebijakan membaik sama sekali). Yang menentukan adalah metrik tujuan (Gini) yang
diukur langsung dari kebijakan hasil training di tiap tahap.

Pakai:
    python experiments/diagnostics/pdqn_learning_curve.py --mu 0.5,0.8 --total 400 --every 50
"""
import argparse
import random

import numpy as np

from marl_spklu.experiments import harness
from marl_spklu.experiments.ablations import binary_recommendation_mode, constant_trust
from marl_spklu.experiments.pdqn_baseline import PDQN_REWARD
from marl_spklu.rl.dqn_trainer import DQNContinuingTrainer
from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent


def curve(dataset, mu, total, every, rollout_steps, eval_seeds, horizon):
    with binary_recommendation_mode(mu), constant_trust(mu):
        tr = DQNContinuingTrainer(dataset, rollout_steps=rollout_steps, seed=0,
                                  verbose=False, horizon=horizon, reward_calc=PDQN_REWARD)
        # Jadwal epsilon dipatok ke TOTAL anggaran, bukan potongan per-checkpoint,
        # supaya peluruhan tetap sama dgn run utuh berukuran `total`.
        tr._total_chunks = total

        steps_per_pass = horizon or 288
        rows = []
        done = 0
        while done < total:
            n = min(every, total - done)
            tr.train(n)
            done += n
            random.seed(0); np.random.seed(0)
            agg = harness.run_multi_seed(
                dataset, agent_factory=lambda: PDQNInferenceAgent(tr.q_net),
                seeds=eval_seeds)
            rows.append((done, done * rollout_steps / steps_per_pass,
                         tr._n_updates,
                         agg["gini_mean"]["mean"], agg["gini_mean"]["std"],
                         agg["acceptance_overall"]["mean"], agg["wait_mean"]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scenario_dataset_3d.json")
    p.add_argument("--mu", default="0.5,0.8")
    p.add_argument("--total", type=int, default=400, help="total chunk training")
    p.add_argument("--every", type=int, default=50, help="evaluasi tiap N chunk")
    p.add_argument("--rollout-steps", type=int, default=96)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--horizon", type=int, default=288)
    args = p.parse_args()

    for mu in [float(x) for x in args.mu.split(",")]:
        print(f"\n=== mu_hat = {mu} ===")
        hdr = (f"{'chunk':>7}{'pass':>8}{'grad':>9}{'gini_mean':>12}{'sd':>8}"
               f"{'accept':>9}{'wait':>8}")
        print(hdr); print("-" * len(hdr))
        rows = curve(args.dataset, mu, args.total, args.every, args.rollout_steps,
                     range(args.seeds), args.horizon)
        for c, ps, gr, g, sd, acc, w in rows:
            print(f"{c:>7}{ps:>8.0f}{gr:>9}{g:>12.4f}{sd:>8.4f}{acc:>9.3f}{w:>8.1f}")
        g = np.array([r[3] for r in rows])
        best = int(np.argmin(g))
        # "cukup" = checkpoint pertama yang sudah dalam 1 sd dari Gini terbaik
        tol = rows[best][4]
        enough = next(i for i in range(len(g)) if g[i] <= g[best] + tol)
        print(f"  -> terbaik pd chunk {rows[best][0]} ({rows[best][1]:.0f} pass), "
              f"gini={g[best]:.4f}")
        print(f"  -> sudah dalam 1 sd dari terbaik sejak chunk {rows[enough][0]} "
              f"({rows[enough][1]:.0f} pass, {rows[enough][2]} langkah gradien)")


if __name__ == "__main__":
    main()
