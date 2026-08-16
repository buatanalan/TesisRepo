"""Bagian E (rencana_pengujian.md) -- System test (E.1) + Ablation study (E.2).

Ini BUKAN pytest: butuh training aktual (bukan hanya cek struktural cepat), jadi
dijalankan sebagai script yang mencetak tabel perbandingan metrik akhir per varian.

  --quick  : horizon & n_updates KECIL (verifikasi struktural: semua varian jalan tanpa
             crash/NaN, arah metrik masuk akal). BUKAN kriteria kelulusan E.1 yang
             sesungguhnya -- dokumen mensyaratkan T-3 (500 SPKLU/pengguna, 5000 langkah,
             training penuh berhari-hari, 3 seed). Gunakan default (tanpa --quick) dgn
             --dataset mengarah ke dataset T-3 kalibrasi utk itu.

Baseline (Bagian A.3): no-recommendation, random-k, greedy shortest-wait, static
round-robin. Ablasi (E.2): 1) tanpa prediktor sadar-koordinasi, 2) tanpa encoder LSTM
(riwayat di-nol-kan), 3) tanpa penalti asimetris (alpha=beta trust), 4) tanpa continuing
task (reset per update), 5) tanpa parameter sharing (TIDAK didukung arsitektur saat ini --
dilaporkan sebagai N/A, bukan disimulasikan).
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.env.user import User
from marl_spklu.rl import rollout as rollout_mod
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.ppo import TorchContinuingTrainer, TorchPolicyTrainer
from marl_spklu.rl.rollout import evaluate_policy
from marl_spklu.rl.training import _fresh_sim, train_mode_a
from marl_spklu.rl.rewards import _gini

DEFAULT_DATASET = os.path.join(REPO_ROOT, "scenario_dataset.json")


def _metrics_from_sim(sim):
    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
    waits = [L["wait_time"] for L in sim.logs]
    return {
        "gini_served": _gini(served),
        "total_served": int(served.sum()),
        "herding_events": sim.herding_events,
        "mean_wait": float(np.mean(waits)) if waits else 0.0,
    }


# --------------------------------------------------------------------------------
# Bagian A.3 -- Baseline
# --------------------------------------------------------------------------------
class RandomKAgent:
    """Rekomendasi acak dari k kandidat feasible terdekat."""

    def __init__(self, k=3, seed=0):
        self.k = k
        self.rng = np.random.default_rng(seed)

    def predict_waits(self, spklus):
        return {sid: 0.0 for sid in spklus}

    def get_recommendation(self, spklus):
        ids = list(spklus.keys())
        if not ids:
            return []
        return [str(self.rng.choice(ids))]


class StaticRoundRobinAgent:
    """Distribusi rekomendasi tetap (round-robin antar SPKLU feasible)."""

    def __init__(self):
        self._ptr = 0

    def predict_waits(self, spklus):
        return {sid: 0.0 for sid in spklus}

    def get_recommendation(self, spklus):
        ids = list(spklus.keys())
        if not ids:
            return []
        sid = ids[self._ptr % len(ids)]
        self._ptr += 1
        return [sid]


def run_baselines(dataset, horizon):
    results = {}
    # No-recommendation: agent=None -> user hanya ikut P_MXL.
    sim = _fresh_sim(dataset); sim.run(max_steps=horizon, agent=None)
    results["no_recommendation"] = _metrics_from_sim(sim)

    sim = _fresh_sim(dataset); sim.run(max_steps=horizon, agent=RandomKAgent())
    results["random_k"] = _metrics_from_sim(sim)

    sim = _fresh_sim(dataset); sim.run(max_steps=horizon, agent=GreedyAgent())
    results["greedy_shortest_wait"] = _metrics_from_sim(sim)

    sim = _fresh_sim(dataset); sim.run(max_steps=horizon, agent=StaticRoundRobinAgent())
    results["static_round_robin"] = _metrics_from_sim(sim)
    return results


# --------------------------------------------------------------------------------
# Bagian E.2 -- Ablasi
# --------------------------------------------------------------------------------
def _train_and_eval(dataset, horizon, n_updates, continuing=True, learned_forecaster=True,
                    zero_hist=False, symmetric_trust=False, seed=0):
    TrainerCls = TorchContinuingTrainer if continuing else TorchPolicyTrainer
    trainer = TrainerCls(dataset, rollout_steps=min(horizon, 96), verbose=False,
                         epochs=2, minibatch=64, seed=seed)

    orig_build_hist = rollout_mod.RLRolloutAgent._build_hist
    if zero_hist:
        rollout_mod.RLRolloutAgent._build_hist = lambda self, user: np.zeros_like(
            orig_build_hist(self, user))

    orig_defaults = User.update_trust.__defaults__
    if symmetric_trust:
        User.update_trust.__defaults__ = (1.0, 1.0)   # alpha=rho -> ablasi asimetri (E.2 Ablasi 3)

    try:
        policy, forecaster = train_mode_a(dataset, trainer, total_updates=n_updates,
                                          learned_forecaster=learned_forecaster,
                                          collect_steps=min(horizon, 500))
        sim = _fresh_sim(dataset)
        metrics = evaluate_policy(sim, policy, forecaster, max_steps=horizon)
    finally:
        rollout_mod.RLRolloutAgent._build_hist = orig_build_hist
        User.update_trust.__defaults__ = orig_defaults
    return metrics


def run_ablations(dataset, horizon, n_updates):
    results = {}
    results["full_system"] = _train_and_eval(dataset, horizon, n_updates, seed=0)
    results["ablasi1_no_coord_predictor"] = _train_and_eval(
        dataset, horizon, n_updates, learned_forecaster=False, seed=1)
    results["ablasi2_no_lstm_encoder"] = _train_and_eval(
        dataset, horizon, n_updates, zero_hist=True, seed=2)
    results["ablasi3_no_asymmetric_penalty"] = _train_and_eval(
        dataset, horizon, n_updates, symmetric_trust=True, seed=3)
    results["ablasi4_no_continuing_task"] = _train_and_eval(
        dataset, horizon, n_updates, continuing=False, seed=4)
    results["ablasi5_no_parameter_sharing"] = None   # N/A: lihat docstring modul
    return results


def _print_table(title, results):
    print(f"\n=== {title} ===")
    cols = ["gini_served", "total_served", "herding_events", "mean_wait"]
    header = f"{'varian':<32}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        if m is None:
            print(f"{name:<32}{'N/A (lihat catatan)':>64}")
            continue
        print(f"{name:<32}" + "".join(f"{m[c]:>16.4f}" if isinstance(m[c], float) else f"{m[c]:>16d}"
                                      for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--quick", action="store_true", help="horizon & n_updates skala-smoke")
    args = ap.parse_args()

    if args.quick:
        horizon, n_updates = 200, 6
        print("[--quick] Skala SMOKE (horizon=200 langkah, n_updates=6). Ini HANYA verifikasi "
             "struktural (tak crash, arah metrik masuk akal) -- BUKAN kriteria kelulusan E.1/E.2 "
             "sesungguhnya. Jalankan tanpa --quick pada dataset T-3 kalibrasi utk hasil riil.")
    else:
        horizon, n_updates = 2880, 100
        print(f"Horizon={horizon} langkah, n_updates={n_updates}. Untuk kepatuhan penuh thd "
             f"dokumen (environment T-3: 500 SPKLU/pengguna, 5000 langkah, 3 seed), sediakan "
             f"dataset T-3 via --dataset dan ulangi run ini dgn seed berbeda secara manual.")

    print(f"Dataset: {args.dataset}")

    baselines = run_baselines(args.dataset, horizon)
    _print_table("Bagian A.3 -- Baseline", baselines)

    ablations = run_ablations(args.dataset, horizon, n_updates)
    _print_table("Bagian E.2 -- Ablation Study", ablations)

    full = ablations["full_system"]
    print("\n=== Bagian E.1 -- Cek relatif thd baseline (arah, bukan ambang absolut) ===")
    print(f"full_system vs greedy_shortest_wait -- herding: "
         f"{full['herding_events']} vs {baselines['greedy_shortest_wait']['herding_events']} "
         f"({'OK: lebih rendah' if full['herding_events'] < baselines['greedy_shortest_wait']['herding_events'] else 'PERINGATAN: tidak lebih rendah'})")
    print(f"full_system vs no_recommendation -- gini_served: "
         f"{full['gini_served']:.4f} vs {baselines['no_recommendation']['gini_served']:.4f} "
         f"({'OK: lebih rendah' if full['gini_served'] < baselines['no_recommendation']['gini_served'] else 'PERINGATAN: tidak lebih rendah'})")
    print("\nCatatan Ablasi 5 (tanpa parameter sharing): arsitektur saat ini EKSPLISIT menolak "
         "parameter per-agen (lihat marl_spklu/rl/policy.py) -- ablasi ini butuh implementasi "
         "terpisah (bukan flag) dan sengaja tidak dijalankan otomatis di sini.")


if __name__ == "__main__":
    main()
