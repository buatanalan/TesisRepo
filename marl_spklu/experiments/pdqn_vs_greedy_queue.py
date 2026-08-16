"""Uji rigor PDQN vs Greedy-queue (baseline "minimum queuing mechanism" milik Lin et al.
2024) -- jalur A dari diskusi metodologi: reward & hyperparameter DIBEKUKAN sebelum
evaluasi ini (PDQN_REWARD = EquityRewardCalculator(alpha_rec=1.0, alpha_shift=1.0), tak
diubah lagi berdasar hasil), dilatih dgn BEBERAPA train_seed (menangkap variansi
pelatihan, bukan cuma variansi lingkungan), dievaluasi 10 eval_seed, diuji Wilcoxon
signed-rank berpasangan per seed.

Konfigurasi (ditetapkan sebelum menjalankan, tidak diubah berdasar hasil):
  dataset   = scenario_dataset_7d.json
  keputusan = SAMPLING (softmax lalu disampel, sesuai bacaan literal spesifikasi)
  mu_hat    = 0.2, 0.5, 0.8
  train_seed= 0, 1, 2       (3 model independen per mu_hat)
  eval_seed = 0..9          (10 seed evaluasi, sama utk PDQN maupun Greedy-queue)

Pakai:
    python -m marl_spklu.experiments.pdqn_vs_greedy_queue --out hasil_rigor_pdqn_vs_gqueue.json
"""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.experiments import harness
from marl_spklu.experiments.ablations import binary_recommendation_mode, constant_trust
from marl_spklu.experiments.pdqn_baseline import PDQN_REWARD, train_pdqn
from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent

DATASET = "scenario_dataset_7d.json"
STOCHASTIC = True
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = range(10)


def gini_per_seed(dataset, agent_factory, mu, seeds):
    """Gini_mean utk TIAP seed terpisah (bukan diringkas mean/std) -- dibutuhkan Wilcoxon
    berpasangan."""
    out = []
    with binary_recommendation_mode(mu, STOCHASTIC), constant_trust(mu):
        for s in seeds:
            r = harness.run_scenario(dataset, agent_factory=agent_factory, seed=s)
            out.append(r["gini_mean"])
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-updates", type=int, default=100)
    p.add_argument("--rollout-steps", type=int, default=96)
    p.add_argument("--out", default="hasil_rigor_pdqn_vs_gqueue.json")
    args = p.parse_args()

    results = {}
    for mu in (0.2, 0.5, 0.8):
        print(f"\n{'='*70}\nmu_hat = {mu}\n{'='*70}")

        # --- Greedy-queue: tak perlu training, langsung evaluasi 10 seed ---
        gq = gini_per_seed(DATASET, lambda: GreedyAgent(mode="queue"), mu, EVAL_SEEDS)
        print(f"Greedy-queue (10 seed): mean={gq.mean():.4f} sd={gq.std():.4f}")

        # --- PDQN: latih 3 model independen (train_seed berbeda), REWARD DIBEKUKAN ---
        pdqn_by_trainseed = []
        for ts in TRAIN_SEEDS:
            print(f"  melatih PDQN train_seed={ts} ...")
            q_net, _ = train_pdqn(DATASET, mu, n_updates=args.n_updates,
                                  rollout_steps=args.rollout_steps, seed=ts,
                                  verbose=False, stochastic=STOCHASTIC,
                                  reward_calc=PDQN_REWARD)
            vals = gini_per_seed(DATASET, lambda qn=q_net: PDQNInferenceAgent(qn),
                                 mu, EVAL_SEEDS)
            pdqn_by_trainseed.append(vals)
            print(f"    eval 10 seed: mean={vals.mean():.4f} sd={vals.std():.4f}")

        pdqn_by_trainseed = np.array(pdqn_by_trainseed)   # (3 train_seed, 10 eval_seed)
        # Rata-rata lintas train_seed per eval_seed -> 10 nilai "PDQN representatif",
        # meredam variansi pelatihan sebelum uji berpasangan thd Greedy-queue (yg tak
        # punya variansi pelatihan sama sekali, deterministik).
        pdqn_avg = pdqn_by_trainseed.mean(axis=0)
        train_seed_spread = pdqn_by_trainseed.mean(axis=1)   # mean per train_seed

        diff = gq - pdqn_avg   # positif = PDQN lebih baik (Gini lebih rendah)
        stat, pval = wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)

        results[mu] = {
            "greedy_queue_per_seed": gq.tolist(),
            "pdqn_avg_per_seed": pdqn_avg.tolist(),
            "pdqn_per_trainseed_mean": train_seed_spread.tolist(),
            "diff_mean": float(diff.mean()), "diff_std": float(diff.std()),
            "wilcoxon_stat": float(stat), "wilcoxon_p": float(pval),
            "pdqn_better": bool(diff.mean() > 0 and pval < 0.05),
        }
        print(f"\nSelisih (Greedy-queue - PDQN) rata2 = {diff.mean():+.4f} +- {diff.std():.4f}")
        print(f"Wilcoxon signed-rank: statistic={stat:.2f} p={pval:.4f} "
              f"-> {'PDQN SIGNIFIKAN lebih baik' if results[mu]['pdqn_better'] else 'TIDAK signifikan'}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] -> {args.out}")

    print(f"\n{'='*70}\nRINGKASAN\n{'='*70}")
    for mu, r in results.items():
        print(f"mu={mu}: diff={r['diff_mean']:+.4f}  p={r['wilcoxon_p']:.4f}  "
              f"{'SIGNIFIKAN' if r['pdqn_better'] else 'tidak signifikan'}")


if __name__ == "__main__":
    main()
