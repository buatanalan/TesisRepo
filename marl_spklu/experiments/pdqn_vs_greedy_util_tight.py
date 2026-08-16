"""Uji PDQN vs Greedy-util pada lingkungan KANDIDAT LEBIH SEMPIT -- opsi #2 dari diskusi
"bagaimana agar PDQN menang".

Diagnosis (lihat LAPORAN_IMPLEMENTASI_PDQN.md §3.11): pada `willingness_ratio` default
(5.0), 86% keputusan melihat SEMUA 8 SPKLU feasible -- `argmin utilisasi SAAT INI` nyaris
selalu benar krn nyaris tak ada trade-off nyata (SPKLU jauh/beban-tersembunyi jarang jadi
soal). Greedy-util = penurunan langsung pd besaran penyusun Gini, jadi pd kondisi ini sulit
dikalahkan siapa pun.

Menyempitkan `willingness_ratio` ke 1.5 (median 2 kandidat feasible, bukan 8) memaksa
trade-off nyata: SPKLU "paling sepi SAAT INI" bisa jadi bukan pilihan yg akan tetap sepi
setelah EV dlm perjalanan tiba (wait_hat, SUDAH ada di observasi PDQN, TAK dipakai Greedy
sama sekali). Inilah blind spot myopia Greedy yg coba dieksploitasi PDQN di sini.

Metodologi identik `pdqn_vs_greedy_queue.py` (reward dibekukan, 3 train_seed, 10 eval_seed,
Wilcoxon signed-rank berpasangan) -- hanya `willingness_ratio` dan baseline pembanding yg
beda, supaya hasil bisa dibandingkan apple-to-apple dgn uji rigor sebelumnya.

Pakai:
    python -m marl_spklu.experiments.pdqn_vs_greedy_util_tight --out hasil_rigor_tight.json
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
WILLINGNESS_RATIO = 1.5   # median 2 kandidat feasible (vs 8 pd default 5.0)
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = range(10)


def gini_per_seed(dataset, agent_factory, mu, seeds, willingness_ratio):
    out = []
    with binary_recommendation_mode(mu, STOCHASTIC), constant_trust(mu):
        for s in seeds:
            r = harness.run_scenario(dataset, agent_factory=agent_factory, seed=s,
                                     willingness_ratio=willingness_ratio)
            out.append(r["gini_mean"])
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-updates", type=int, default=100)
    p.add_argument("--rollout-steps", type=int, default=96)
    p.add_argument("--out", default="hasil_rigor_tight.json")
    args = p.parse_args()

    results = {}
    for mu in (0.2, 0.5, 0.8):
        print(f"\n{'='*70}\nmu_hat = {mu}  (willingness_ratio={WILLINGNESS_RATIO})\n{'='*70}")

        gu = gini_per_seed(DATASET, lambda: GreedyAgent(mode="utilization"), mu,
                           EVAL_SEEDS, WILLINGNESS_RATIO)
        print(f"Greedy-util (10 seed): mean={gu.mean():.4f} sd={gu.std():.4f}")

        pdqn_by_trainseed = []
        for ts in TRAIN_SEEDS:
            print(f"  melatih PDQN train_seed={ts} ...")
            q_net, _ = train_pdqn(DATASET, mu, n_updates=args.n_updates,
                                  rollout_steps=args.rollout_steps, seed=ts,
                                  verbose=False, stochastic=STOCHASTIC,
                                  reward_calc=PDQN_REWARD,
                                  willingness_ratio=WILLINGNESS_RATIO)
            vals = gini_per_seed(DATASET, lambda qn=q_net: PDQNInferenceAgent(qn),
                                 mu, EVAL_SEEDS, WILLINGNESS_RATIO)
            pdqn_by_trainseed.append(vals)
            print(f"    eval 10 seed: mean={vals.mean():.4f} sd={vals.std():.4f}")

        pdqn_by_trainseed = np.array(pdqn_by_trainseed)
        pdqn_avg = pdqn_by_trainseed.mean(axis=0)
        train_seed_spread = pdqn_by_trainseed.mean(axis=1)

        diff = gu - pdqn_avg
        stat, pval = wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)

        results[mu] = {
            "greedy_util_per_seed": gu.tolist(),
            "pdqn_avg_per_seed": pdqn_avg.tolist(),
            "pdqn_per_trainseed_mean": train_seed_spread.tolist(),
            "diff_mean": float(diff.mean()), "diff_std": float(diff.std()),
            "wilcoxon_stat": float(stat), "wilcoxon_p": float(pval),
            "pdqn_better": bool(diff.mean() > 0 and pval < 0.05),
        }
        print(f"\nSelisih (Greedy-util - PDQN) rata2 = {diff.mean():+.4f} +- {diff.std():.4f}")
        print(f"Wilcoxon signed-rank: statistic={stat:.2f} p={pval:.4f} "
              f"-> {'PDQN SIGNIFIKAN lebih baik' if results[mu]['pdqn_better'] else 'TIDAK signifikan'}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] -> {args.out}")

    print(f"\n{'='*70}\nRINGKASAN (willingness_ratio={WILLINGNESS_RATIO})\n{'='*70}")
    for mu, r in results.items():
        print(f"mu={mu}: diff={r['diff_mean']:+.4f}  p={r['wilcoxon_p']:.4f}  "
              f"{'SIGNIFIKAN' if r['pdqn_better'] else 'tidak signifikan'}")


if __name__ == "__main__":
    main()
