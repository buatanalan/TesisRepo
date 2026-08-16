import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Skenario eksperimen UTAMA tesis (Bab IV, "Evaluation"): membandingkan policy MARL
terlatih dengan tiga baseline di bawah konfigurasi lingkungan yang IDENTIK (dataset,
seed, batas jangkauan, kondisi awal timpang):

    S0  Tanpa intervensi   -- pengguna murni mengikuti distribusi Mixed Logit.
    S1  Greedy least-loaded -- heuristik antrean terpendek saat ini (rentan flocking).
    S3  OP-SRL (adaptasi)  -- rekomendasi berbasis prediksi wait, kepatuhan PENUH eksogen.
    S4  MARL (usulan)      -- policy PPO terlatih, kepatuhan endogen (trust dinamis).

Semua skenario dijalankan lewat harness yang sama (marl_spklu.experiments.harness) agar
KPI (Gini utilisasi, waktu tunggu, herding, entropi rekomendasi) terukur secara sebanding.
"""
import argparse
import json

from marl_spklu.experiments.harness import compare_scenarios, format_comparison, DEFAULT_DATASET
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.training import train_mode_a
from marl_spklu.rl.rollout import InferenceAgent


def train_marl_policy(dataset, updates, rollout_steps, k, seed, honest_estwait, verbose=True):
    """Latih policy MARL (MAPPO/PPO, continuing task) dan kembalikan (policy, forecaster)."""
    tr = TorchContinuingTrainer(dataset, k=k, rollout_steps=rollout_steps, seed=seed,
                                verbose=verbose, honest_estwait=honest_estwait)
    policy, forecaster = train_mode_a(dataset, tr, total_updates=updates)
    return policy, forecaster, tr


def run_main_experiment(dataset=DEFAULT_DATASET, updates=200, rollout_steps=96, k=3,
                        seed=0, honest_estwait=False, eval_seeds=5, max_steps=None,
                        scenarios=None, out_path="main_experiment_results.json"):
    # honest_estwait=False (default solusi utama, desain_final_solusi.md §3.3): delta_i
    # AKTIF memodifikasi EstWait yang ditampilkan -- bukan sekadar disampel/dilatih tanpa
    # efek. honest_estwait=True tetap tersedia via --manipulative TERBALIK (lihat CLI di
    # bawah) sbg ablasi terpisah, bukan lagi default diam-diam.
    print(f"=== Melatih policy MARL ({updates} chunk-update x {rollout_steps} langkah) ===")
    policy, forecaster, tr = train_marl_policy(dataset, updates, rollout_steps, k, seed,
                                               honest_estwait)

    # InferenceAgent butuh `sim` di konstruktor (utk memetakan sids -> idx observasi),
    # tapi harness membangun `sim` SETELAH agent_factory() dipanggil. `_DeferredMarlAgent`
    # menunda pembuatan InferenceAgent nyata sampai harness memanggil `bind_to_sim(sim)`.
    extra_scenarios = {"S4_marl": (lambda: _DeferredMarlAgent(policy, forecaster, k, honest_estwait), False)}

    print(f"\n=== Menjalankan skenario utama (S0/S1/S3/S4) x {eval_seeds} seed ===")
    names = scenarios or ["S0_no_intervention", "S1_greedy", "S3_opsrl", "S4_marl"]
    res = compare_scenarios(dataset, scenario_names=names, seeds=range(eval_seeds),
                            max_steps=max_steps, extra_scenarios=extra_scenarios)

    table = format_comparison(res)
    print("\n" + table)

    slim = {n: {k2: v2 for k2, v2 in agg.items() if k2 != "_runs"} for n, agg in res.items()}
    with open(out_path, "w") as f:
        json.dump({"config": {"dataset": dataset, "updates": updates,
                              "rollout_steps": rollout_steps, "k": k, "seed": seed,
                              "honest_estwait": honest_estwait, "eval_seeds": eval_seeds},
                  "train_history": tr.history, "comparison": slim}, f, indent=2)
    print(f"\n[INFO] Hasil disimpan ke {out_path}")
    return res, policy, forecaster


class _DeferredMarlAgent:
    """Agen S4: InferenceAgent hanya bisa dibangun setelah `sim` tersedia (butuh daftar
    SPKLU/user milik sim itu). Harness memanggil get_recommendation/predict_waits per
    keputusan tanpa memberi akses ke `sim` di titik konstruksi agent_factory(), jadi
    InferenceAgent nyata dibangun lazy saat panggilan pertama (sim sudah aktif)."""

    def __init__(self, policy, forecaster, k, honest_estwait):
        self._policy = policy
        self._forecaster = forecaster
        self._k = k
        self._honest = honest_estwait
        self._real = None

    def bind_to_sim(self, sim):
        self._real = InferenceAgent(self._policy, sim, self._forecaster, self._k,
                                    honest_estwait=self._honest)

    def get_recommendation(self, feasible_spklus):
        return self._real.get_recommendation(feasible_spklus)

    def predict_waits(self, feasible_spklus):
        return self._real.predict_waits(feasible_spklus)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--updates", type=int, default=200, help="jumlah chunk-update latih MARL")
    ap.add_argument("--rollout-steps", type=int, default=96, help="ukuran chunk (96=1 hari)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--honest", action="store_true",
                    help="Ablasi: EstWait JUJUR (delta_i disampel/dilatih tapi TAK "
                         "ditampilkan/berpengaruh). Default (tanpa flag ini): delta_i "
                         "AKTIF memodifikasi EstWait tampilan (solusi utama, "
                         "desain_final_solusi.md §3.3).")
    ap.add_argument("--out", default="main_experiment_results.json")
    a = ap.parse_args()
    run_main_experiment(a.dataset, a.updates, a.rollout_steps, a.k, a.seed,
                        honest_estwait=a.honest, eval_seeds=a.eval_seeds,
                        max_steps=a.max_steps, out_path=a.out)
