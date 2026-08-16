import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""SOLUSI UTAMA -- entry point kanonis desain_final_solusi.md.

Ini BUKAN eksperimen ablasi/eksploratif seperti skrip lain di experiments/ -- ini adalah
titik masuk TUNGGAL yang merepresentasikan konfigurasi solusi final tesis, dengan setiap
parameter diberi label tingkat kepastian eksplisit (Bagian 1 dokumen):

    LOCKED     -- konsekuensi struktural masalah, tidak bisa diganti tanpa merumuskan
                  ulang masalah.
    JUSTIFIED  -- pilihan defensible via literatur, ada alternatif valid (diverifikasi
                  via ablasi di experiments/validation/run_design_validation.py).
    EMPIRICAL  -- nilai default masuk akal, TAPI belum ditetapkan lewat grid search
                  penuh (Bagian 4 dokumen) -- lihat TIER_REPORT di bawah utk status
                  masing-masing.

Jalankan `python experiments/training/run_solusi_utama.py --tier-report` utk mencetak
tabel klasifikasi tanpa menjalankan training (dokumentasi cepat).
"""
import argparse
import json

from marl_spklu.experiments.harness import compare_scenarios, format_comparison
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.training import train_mode_a
from marl_spklu.rl.rollout import InferenceAgent
from marl_spklu.rl.rewards import RewardCalculator

# =====================================================================================
# Konfigurasi SOLUSI UTAMA -- setiap parameter berlabel tingkat kepastian (Bagian 1-5
# desain_final_solusi.md). Ini SUMBER KEBENARAN TUNGGAL; skrip eksperimen/ablasi lain
# boleh menyimpang secara sadar (didokumentasikan di skrip masing2), tapi skrip INI
# selalu merepresentasikan solusi final apa adanya.
# =====================================================================================

DATASET_DEFAULT = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "scenario_dataset_calibrated_main.json")

TIER_REPORT = [
    # (parameter, nilai default, tier, alasan singkat)
    ("Formulasi masalah", "Dec-POMDP (env+agent+encoder terpisah)", "LOCKED",
     "konsekuensi 4 karakteristik masalah (III.1.2); tak ada formulasi lain yg akomodasi semua"),
    ("Kondisi awal", "Gini(served) baseline ~0,834 (scenario_dataset_calibrated_main.json)",
     "LOCKED", "harus mulai dari ketimpangan empiris Jabodetabek, bukan seimbang (III.1.2)"),
    ("Task formulation", "continuing (TorchContinuingTrainer, tanpa reset antar-chunk)",
     "LOCKED", "konsekuensi kondisi-awal-timpang; reset menghapus sinyal yg harus dipelajari"),
    ("Reward, sumber informasi", "outcome teramati saja (TIDAK akses T_i/beta_i/P_MXL)",
     "LOCKED", "requirement validitas -- diverifikasi otomatis (test_B4_3 rencana_pengujian.md)"),
    ("Parameter sharing", "SATU set parameter utk semua agen (bukan per-user)", "LOCKED",
     "500+ agen aktif -> parameter terpisah tak scale/sample-efficient (Yu 2022)"),
    ("CTDE", "actor terdesentralisasi + critic tersentralisasi (state global)", "LOCKED",
     "reward global (Gini/flocking) tak bisa dinilai dari observasi lokal manapun"),
    ("Algoritma pembelajaran", "MAPPO (PPOTrainer, avg_reward=True, clip=0.2)", "JUSTIFIED",
     "diverifikasi vs IPPO/MADDPG-proxy/QMIX-proxy: laporan_validasi_desain.md B.5.1-B.5.3"),
    ("Encoder riwayat", "LSTM (hist_hidden=16, HIST_K=5)", "JUSTIFIED",
     "diverifikasi vs GRU/MLP-window: B.4.1-B.4.2 -- R^2(c_t->T_i) BELUM stabil >0.5 lintas run, MASIH DIPANTAU"),
    ("Struktur aksi", "(k_i diskrit, delta_i kontinu [-10,10] menit, delta_i AKTIF)", "JUSTIFIED",
     "honest_estwait=False -- delta_i memodifikasi EstWait tampilan, bukan vestigial"),
    ("Prediktor waktu tunggu", "MLP sadar-koordinasi (LearnedForecaster, phi_activity aktif)",
     "JUSTIFIED", "train_mode_a(learned_forecaster=True) default -- diverifikasi B.3.1/C.3.1"),
    ("Model pilihan pengguna (lingkungan)", "MXL 5-faktor terkalibrasi", "JUSTIFIED",
     "bagian lingkungan simulasi, bukan artefak -- lihat marl_spklu/env/user.py"),
    ("Model trust", "asimetris (alpha=1.0, beta/rho=0.5, eta=trust_lr=0.1)", "JUSTIFIED",
     "bentuk fungsi terkunci; arah asimetri terverifikasi B.2.2 tapi dampak PRAKTIS blm signifikan pd skala kecil"),
    ("Bobot reward (w1,w2,wp,lambda1,lambda2)",
     "alpha_wait=1.0, alpha_honesty=1.0, beta_prox=0.1, alpha_gini=0.5, alpha_flock=0.3",
     "EMPIRICAL", "rasio komponen <10:1 terverifikasi (test_B4_4), TAPI belum grid-search penuh"),
    ("Chunk size H (rollout_steps)", "96 langkah (=1 hari @15 menit)", "EMPIRICAL",
     "harus > median waktu tempuh intra-klaster; nilai spesifik BELUM disapu {20,30,60,100}"),
    ("Rentang delta [-10,10] menit", "delta_max=10.0", "EMPIRICAL",
     "default dokumen; distribusi |delta| optimal blm dianalisis utk longgar/ketat"),
    ("Panjang riwayat K", "5", "EMPIRICAL",
     "B.4.3 (laporan_validasi_desain.md) TIDAK konklusif -- optimal berpindah antar-run (5..200), blm stabil"),
    ("Jumlah kandidat k per rekomendasi", "3", "EMPIRICAL", "default dokumen, blm disapu {2,3,5}"),
]


def print_tier_report():
    print("=" * 100)
    print("SOLUSI UTAMA -- Klasifikasi Tingkat Kepastian (desain_final_solusi.md Bagian 1-5)")
    print("=" * 100)
    for name, val, tier, why in TIER_REPORT:
        print(f"\n[{tier}] {name}")
        print(f"    nilai  : {val}")
        print(f"    alasan : {why}")
    print("\n" + "=" * 100)
    n_locked = sum(1 for r in TIER_REPORT if r[2] == "LOCKED")
    n_just = sum(1 for r in TIER_REPORT if r[2] == "JUSTIFIED")
    n_emp = sum(1 for r in TIER_REPORT if r[2] == "EMPIRICAL")
    print(f"Ringkasan: {n_locked} LOCKED, {n_just} JUSTIFIED, {n_emp} EMPIRICAL "
         f"({len(TIER_REPORT)} total)")
    print("Item EMPIRICAL memakai nilai default dokumen -- BELUM lolos grid search penuh "
         "(Bagian 4 desain_final_solusi.md). Lihat experiments/validation/ utk hasil ablasi "
         "yg sudah dijalankan atas item JUSTIFIED.")


def train_solusi_utama(dataset, updates, rollout_steps, k, seed, collect_steps=500, verbose=True):
    """Latih MAPPO (JUSTIFIED) sbg continuing task (LOCKED) dgn prediktor sadar-koordinasi
    (JUSTIFIED) & delta_i AKTIF (honest_estwait=False -- keputusan solusi utama).

    collect_steps (EMPIRICAL, default 500): panjang trajektori baseline utk pretrain
    LearnedForecaster (train_mode_a). 500 langkah (~5 hari) sudah menghasilkan puluhan
    ribu baris fitur -- cukup utk MLP kecil, jauh lebih cepat dari default modul
    (2880 = horizon PENUH, terlalu lambat utk iterasi). Naikkan mendekati 2880 utk run
    publikasi final."""
    reward_calc = RewardCalculator()   # bobot EMPIRICAL default, lihat TIER_REPORT
    tr = TorchContinuingTrainer(dataset, k=k, rollout_steps=rollout_steps, seed=seed,
                                verbose=verbose, honest_estwait=False, reward_calc=reward_calc)
    policy, forecaster = train_mode_a(dataset, tr, total_updates=updates,
                                      learned_forecaster=True, collect_steps=collect_steps)
    return policy, forecaster, tr


class _DeferredMarlAgent:
    """InferenceAgent butuh `sim` yg baru tersedia setelah harness membangun skenario."""

    def __init__(self, policy, forecaster, k):
        self._policy = policy; self._forecaster = forecaster; self._k = k
        self._real = None

    def bind_to_sim(self, sim):
        self._real = InferenceAgent(self._policy, sim, self._forecaster, self._k,
                                    honest_estwait=False)

    def get_recommendation(self, feasible_spklus):
        return self._real.get_recommendation(feasible_spklus)

    def predict_waits(self, feasible_spklus):
        return self._real.predict_waits(feasible_spklus)


def run_solusi_utama(dataset=DATASET_DEFAULT, updates=200, rollout_steps=96, k=3,
                     seed=0, eval_seeds=5, max_steps=None, collect_steps=500,
                     out_path="solusi_utama_results.json"):
    print_tier_report()
    print(f"\n=== Melatih SOLUSI UTAMA ({updates} chunk-update x {rollout_steps} langkah, "
         f"dataset={dataset}) ===")
    policy, forecaster, tr = train_solusi_utama(dataset, updates, rollout_steps, k, seed,
                                                collect_steps=collect_steps)

    extra_scenarios = {"S4_marl_solusi_utama": (lambda: _DeferredMarlAgent(policy, forecaster, k), False)}

    print(f"\n=== Evaluasi vs baseline (S0/S1/S3/S4) x {eval_seeds} seed ===")
    names = ["S0_no_intervention", "S1_greedy", "S3_opsrl", "S4_marl_solusi_utama"]
    res = compare_scenarios(dataset, scenario_names=names, seeds=range(eval_seeds),
                            max_steps=max_steps, extra_scenarios=extra_scenarios)
    table = format_comparison(res)
    print("\n" + table)

    slim = {n: {k2: v2 for k2, v2 in agg.items() if k2 != "_runs"} for n, agg in res.items()}
    with open(out_path, "w") as f:
        json.dump({
            "config": {"dataset": dataset, "updates": updates, "rollout_steps": rollout_steps,
                      "k": k, "seed": seed, "honest_estwait": False,
                      "learned_forecaster": True, "eval_seeds": eval_seeds},
            "tier_report": [{"param": n, "value": v, "tier": t, "reason": w}
                            for n, v, t, w in TIER_REPORT],
            "train_history": tr.history, "comparison": slim,
        }, f, indent=2)
    print(f"\n[INFO] Hasil disimpan ke {out_path}")
    return res, policy, forecaster


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--rollout-steps", type=int, default=96)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--collect-steps", type=int, default=500,
                    help="panjang trajektori pretrain forecaster (EMPIRICAL, default 500)")
    ap.add_argument("--out", default="solusi_utama_results.json")
    ap.add_argument("--tier-report", action="store_true",
                    help="Cetak klasifikasi LOCKED/JUSTIFIED/EMPIRICAL saja, tanpa training.")
    a = ap.parse_args()
    if a.tier_report:
        print_tier_report()
    else:
        run_solusi_utama(a.dataset, a.updates, a.rollout_steps, a.k, a.seed,
                         eval_seeds=a.eval_seeds, max_steps=a.max_steps,
                         collect_steps=a.collect_steps, out_path=a.out)
