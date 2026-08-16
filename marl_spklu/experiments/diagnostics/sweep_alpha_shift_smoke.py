"""Analisis #2 (diminta user): uji CEPAT (smoke test, bukan rigor penuh) apakah menaikkan
bobot alpha_shift relatif thd alpha_rec membuka ruang bagi PDQN melampaui greedy_util di
mu_hat=0.8 -- titik SATU-SATUNYA mu_hat yg signifikan kalah (§4.4 laporan).

Motivasi: analisis #1 (measure_rshift_magnitude.py) membuktikan r_shift BUKAN sinyal kecil/
langka (50% keputusan tak-nol, rasio magnitudo 0.72x r_rec di mu=0.8) -- jadi bukan soal
skala. Uji ini menaikkan alpha_shift jauh di atas alpha_rec utk memaksa jaringan memprioritas-
kan suku personalisasi saat belajar, sbg tes langsung apakah personalisasi BISA dieksploitasi
kalau diberi bobot dominan (bukan cuma disingkirkan lewat pengurangan alpha_rec).

Anggaran DIKECILKAN sengaja (1 train-seed, 3 eval-seed, n_updates lebih sedikit) --
tujuannya sinyal ARAH cepat, bukan klaim statistik final. Kalau arahnya menjanjikan, baru
scale-up ke metodologi rigor penuh (3 train-seed x 10 eval-seed x Wilcoxon).

Pemakaian:
    python -m marl_spklu.experiments.diagnostics.sweep_alpha_shift_smoke
"""
import numpy as np

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.experiments import harness
from marl_spklu.experiments.ablations import binary_recommendation_mode, constant_trust
from marl_spklu.experiments.pdqn_baseline import train_pdqn
from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent
from marl_spklu.rl.rewards import EquityRewardCalculator

DATASET = "scenario_dataset_7d.json"
MU = 0.8
STOCHASTIC = True
TRAIN_SEED = 0
EVAL_SEEDS = range(5)
N_UPDATES = 60

VARIANTS = {
    "default_1_1":      EquityRewardCalculator(alpha_rec=1.0, alpha_shift=1.0),
    "shift_heavy_1_3":  EquityRewardCalculator(alpha_rec=1.0, alpha_shift=3.0),
    "shift_only_0_1":   EquityRewardCalculator(alpha_rec=0.0, alpha_shift=1.0),
}


def gini_per_seed(agent_factory, seeds):
    out = []
    with binary_recommendation_mode(MU, STOCHASTIC), constant_trust(MU):
        for s in seeds:
            r = harness.run_scenario(DATASET, agent_factory=agent_factory, seed=s)
            out.append(r["gini_mean"])
    return np.array(out)


def main():
    gu = gini_per_seed(lambda: GreedyAgent(mode="utilization"), EVAL_SEEDS)
    print(f"Greedy-util (mu={MU}, {len(list(EVAL_SEEDS))} seed): "
         f"mean={gu.mean():.4f} sd={gu.std():.4f}")

    for name, reward_calc in VARIANTS.items():
        print(f"\n--- melatih PDQN varian '{name}' ---")
        q_net, _ = train_pdqn(DATASET, MU, n_updates=N_UPDATES, seed=TRAIN_SEED,
                              verbose=False, stochastic=STOCHASTIC, reward_calc=reward_calc)
        pdqn_gini = gini_per_seed(lambda qn=q_net: PDQNInferenceAgent(qn), EVAL_SEEDS)
        diff = gu.mean() - pdqn_gini.mean()
        verdict = "PDQN lebih baik" if diff > 0 else "Greedy-util lebih baik"
        print(f"PDQN[{name}]: mean={pdqn_gini.mean():.4f} sd={pdqn_gini.std():.4f}  "
             f"selisih(Greedy-PDQN)={diff:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
