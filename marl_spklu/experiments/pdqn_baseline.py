"""Tahap 0 (spesifikasi_teknis_pdqn_baseline.md): latih PDQN diskrit pada mu_hat statis
{0.2, 0.5, 0.8} lalu bandingkan Gini utilisasi terhadap Greedy (S1, sudah dinormalisasi
kapasitas) dan Natural (S0, agent=None) -- SEMUA di bawah trust dibekukan mu_hat yang sama,
supaya perbandingan adil (Greedy/Natural pun terkena efek mu_hat via mixing trust*w_i di
User.decide_spklu, meski keduanya tak melatih apa pun terhadapnya).

Reward: RewardCalculator yang SUDAH berjalan di marl_spklu/rl/rewards.py (wait-
improvement+prox-honesty individual, -Gini-flocking global) -- BUKAN formula
1/(tau_tr+tau_qu+tau_ch) di draf spec (instruksi eksplisit: pakai reward yang ada,
jangan reward bawaan minimalkan waktu tunggu/pelayanan).

CATATAN EKSPERIMEN (dicoba & TERBUKTI LEBIH BURUK, dipertahankan sbg referensi):
SIMPLE_REWARD di bawah (R = -alpha_gini*Gini(utilisasi) SAJA, suku individual dinolkan)
sempat dicoba sbg penyederhanaan fokus-pemerataan. Hasilnya (100 chunk, 3 mu, 5 seed)
justru gini_mean PDQN LEBIH BURUK drpd reward gabungan (mis. mu=0.5: 0.521 vs 0.486) --
sinyal -Gini per-langkah terlalu DIFUS (satu skalar global disiarkan rata ke SEMUA
transisi tertunda tanpa membedakan kontribusi aksi spesifik), kredit belajar lebih
lemah drpd suku wait_reward yg spesifik per-keputusan meski delayed. Reward gabungan
default TETAP dipakai karena empiris lebih baik.

Pemakaian:
    python -m marl_spklu.experiments.pdqn_baseline --dataset scenario_dataset.json \
        --n-updates 200 --seeds 3 --mu 0.2,0.5,0.8
"""
import argparse
import json

from marl_spklu.experiments import harness
from marl_spklu.experiments.ablations import binary_recommendation_mode, constant_trust
from marl_spklu.rl.dqn_trainer import DQNContinuingTrainer
from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent
from marl_spklu.rl.rewards import EquityRewardCalculator, RewardCalculator

# --- Reward historis (TIDAK dipakai; disimpan sbg catatan eksperimen) ---------------
# SIMPLE_REWARD: hanya -Gini global. Dicoba, gini_mean PDQN justru LEBIH BURUK drpd
#   reward gabungan -- sinyal -Gini per-langkah terlalu difus (satu skalar global
#   disiarkan rata ke semua transisi, kontribusi tiap keputusan tak terbedakan).
# LEGACY_REWARD: reward gabungan (wait-improvement + Prox - Gini - flocking). Terbukti
#   TIDAK SELARAS lewat uji keselarasan: kebijakan anti_greedy (Gini TERBURUK) justru
#   memperoleh reward TERTINGGI, korelasi(reward, Gini) = +0.55 pd mu_hat=0.8.
SIMPLE_REWARD = RewardCalculator(alpha_wait=0.0, beta_prox=0.0, alpha_honesty=0.0,
                                 alpha_gini=1.0, alpha_flock=0.0)
LEGACY_REWARD = RewardCalculator(alpha_honesty=0.0)

# --- Reward default PDQN baseline (DIBEKUKAN, dipakai klaim rigor Bagian 4 laporan) ---
# R = (u_rata2_feasible - u(a_hat)) + (u(default) - u(dipilih)); lihat EquityRewardCalculator.
# Lolos uji keselarasan di kedua level mu_hat: greedy_util (Gini terbaik) memperoleh reward
# tertinggi, korelasi(reward, Gini) = -0.57 (mu=0.5) dan -0.71 (mu=0.8).
PDQN_REWARD = EquityRewardCalculator(alpha_rec=1.0, alpha_shift=1.0)

# --- Varian eksperimental: + suku ANTI-OVERSHOOT (opsi #1 "bagaimana agar PDQN menang") -
# Menambah penalti merekomendasikan SPKLU yg baru saja ramai direkomendasikan (recent_recs),
# menyasar blind spot Greedy yg stateless antar-rekomendasi (lihat EquityRewardCalculator).
# alpha_flock=0.5 dipilih sbg titik-awal netral (sepadan skala alpha_rec/alpha_shift=1.0
# setelah dibagi rec_activity_scale=10 -- belum disapu/dikalibrasi).
PDQN_REWARD_ANTIFLOCK = EquityRewardCalculator(alpha_rec=1.0, alpha_shift=1.0, alpha_flock=0.5)


def train_pdqn(dataset_path, mu_hat: float, n_updates: int = 200, rollout_steps: int = 384,
               seed: int = 0, verbose: bool = True, reward_calc=None, horizon=None,
               stochastic: bool = False, n_types=None, use_preference: bool = True,
               willingness_ratio: float = 5.0):
    """Latih satu PDQN dengan mu_hat statis (µ̂_statis) dibekukan sepanjang training via
    ablations.constant_trust. Kembalikan q_net terlatih. reward_calc default=RewardCalculator()
    (gabungan individual+global, terbukti empiris lebih baik drpd SIMPLE_REWARD).
    horizon: panjang satu pass simulasi sebelum reset (None -> horizon penuh dataset).

    Dilatih di bawah model keputusan pengguna PDQN baseline (binary_recommendation_mode):
    f_rec one-hot dicampur di skala utilitas dgn bobot mu_hat murni. constant_trust ikut
    dipasang supaya atribut `trust` (tak dibaca di mode ini) tetap konsisten di pelaporan
    metrik, bukan melayang oleh update_trust yang sudah tak bermakna."""
    with binary_recommendation_mode(mu_hat, stochastic), constant_trust(mu_hat):
        trainer = DQNContinuingTrainer(dataset_path, rollout_steps=rollout_steps,
                                       seed=seed, verbose=verbose, horizon=horizon,
                                       n_types=n_types, use_preference=use_preference,
                                       willingness_ratio=willingness_ratio,
                                       reward_calc=reward_calc or PDQN_REWARD)
        q_net = trainer.train(n_updates)
    return q_net, trainer


def compare_pdqn_vs_baselines(dataset_path, mu_hat: float, q_net, seeds=range(5), max_steps=None,
                              stochastic: bool = False):
    """Bandingkan Gini(PDQN, mu_hat) vs Gini(Greedy, mu_hat) vs Gini(Natural, mu_hat).

    KETIGANYA dijalankan di dalam binary_recommendation_mode(mu_hat) yang SAMA -- ini
    syarat keabsahan: kalau PDQN memakai campur-utilitas berbobot mu_hat sementara Greedy
    memakai campur-probabilitas berbobot trust*w_i, keduanya diuji di lingkungan yang
    berbeda dan selisih Gini-nya tak bisa diatribusikan ke agen. Greedy tak perlu diubah:
    `recommendations` yang sudah dikembalikannya dikonversi jadi f_rec one-hot di dalam
    decide_spklu. Natural (tanpa agen) otomatis konsisten -> f_rec nol -> murni preferensi.
    Kriteria Bagian 5.2: Gini(PDQN) < Gini(Greedy) < Gini(Natural)."""
    extra = {
        f"P_pdqn_mu{mu_hat}": (lambda: PDQNInferenceAgent(q_net), False),
    }
    with binary_recommendation_mode(mu_hat, stochastic), constant_trust(mu_hat):
        results = harness.compare_scenarios(
            dataset_path,
            scenario_names=["S0_no_intervention", "S1_greedy", f"P_pdqn_mu{mu_hat}"],
            seeds=seeds, max_steps=max_steps, extra_scenarios=extra)
    return results


def run_tahap0(dataset_path, mu_values=(0.2, 0.5, 0.8), n_updates=200, rollout_steps=384,
              train_seed=0, eval_seeds=range(5), max_steps=None, verbose=True,
              stochastic: bool = False):
    """Jalankan protokol Tahap 0 lengkap: latih P-low/P-mid/P-high, bandingkan tiap satu
    dengan Greedy & Natural pada mu_hat yang sama. Return dict {mu_hat: comparison_result}.
    `max_steps` dipakai sbg horizon TRAINING sekaligus batas EVALUASI -- keduanya harus
    sama, kalau tidak kebijakan dilatih pada rezim horizon yang berbeda dari yang diuji."""
    all_results = {}
    for mu in mu_values:
        if verbose:
            print(f"\n=== Melatih PDQN, mu_hat_statis={mu} ===")
        q_net, _ = train_pdqn(dataset_path, mu, n_updates=n_updates, horizon=max_steps,
                              rollout_steps=rollout_steps, seed=train_seed, verbose=verbose,
                              stochastic=stochastic)
        if verbose:
            print(f"\n=== Membandingkan PDQN(mu={mu}) vs Greedy vs Natural (mu={mu}) ===")
        res = compare_pdqn_vs_baselines(dataset_path, mu, q_net, seeds=eval_seeds,
                                        max_steps=max_steps, stochastic=stochastic)
        all_results[mu] = res
        if verbose:
            print(harness.format_comparison(res))
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tahap 0: PDQN baseline vs Greedy vs Natural.")
    p.add_argument("--dataset", type=str, default=harness.DEFAULT_DATASET)
    p.add_argument("--n-updates", type=int, default=200, help="jumlah chunk training DQN")
    p.add_argument("--rollout-steps", type=int, default=384, help="langkah simulasi per chunk")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5, help="jumlah seed evaluasi (0..N-1)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--mu", type=str, default="0.2,0.5,0.8")
    p.add_argument("--stochastic", action="store_true",
                   help="pilihan pengguna disampel dari softmax(score), bukan argmax")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    mu_values = tuple(float(x) for x in args.mu.split(","))
    all_res = run_tahap0(args.dataset, mu_values=mu_values, n_updates=args.n_updates,
                         rollout_steps=args.rollout_steps, train_seed=args.train_seed,
                         eval_seeds=range(args.seeds), max_steps=args.max_steps,
                         stochastic=args.stochastic)

    if args.out:
        slim = {str(mu): {name: {k: v for k, v in agg.items() if k != "_runs"}
                          for name, agg in res.items()}
               for mu, res in all_res.items()}
        with open(args.out, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"\n[INFO] hasil ditulis ke {args.out}")
