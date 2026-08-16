import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.experiments.ablations import initial_trust

# Perluasan Tahap 3 (permintaan user): sapuan TRUST AWAL berbeda (bukan dibekukan --
# trust tetap DINAMIS sesudahnya, `initial_trust()` beda dari `constant_trust()`) x
# horizon 90-hari (bukan 30-hari kanonik) -- coba lihat apakah performativitas lebih
# terlihat dgn titik awal berbeda & waktu kontinu lebih panjang utk trust bergerak.
# anggaran_chunk DIPERTAHANKAN sama (300) drpd Tahap 3 pokok -- shg total budget-training
# (step lingkungan) SEBANDING, tapi jumlah reset horizon berkurang (~3,3x drpd ~10x di
# 30-hari) krn horizon 90-hari 3x lebih panjang -- trust dapat waktu kontinu lbh lama
# sebelum tiap reset "PASS-BARU".
DATASET_90D = os.path.join(common.ROOT, "scenario_dataset_klaster12_90d.json")

CONFIG = dict(
    dataset_path=DATASET_90D,
    initial_trust_values=[0.3, 0.5, 0.7],
    n_train_seed=3,
    anggaran_chunk=300,
    k=2,
    rollout_steps=288,
    lr=1e-4,
    vf_coef=0.25,
    ent_coef=0.002,
    max_step_gap=4,
)
common.save_json(CONFIG, "03_config_initialtrust90d_beku.json")
print("CONFIG sapuan trust-awal x horizon-90d:", CONFIG, flush=True)

results = []
t0 = time.time()
for it in CONFIG["initial_trust_values"]:
    for seed in range(CONFIG["n_train_seed"]):
        print(f"=== initial_trust={it} horizon=90d train_seed={seed} ===", flush=True)
        with initial_trust(value=it):
            tr = TorchContinuingTrainer(
                CONFIG["dataset_path"], k=CONFIG["k"], rollout_steps=CONFIG["rollout_steps"],
                reward_calc=RewardCalculator.seimbang(), seed=seed, verbose=True,
                log_path=os.path.join(common.OUTDIR, f"03_train_log_it{it}_90d_seed{seed}.jsonl"),
                lr=CONFIG["lr"], vf_coef=CONFIG["vf_coef"], ent_coef=CONFIG["ent_coef"],
                max_step_gap=CONFIG["max_step_gap"])
            policy = tr.train(FormulaForecaster(), n_updates=CONFIG["anggaran_chunk"])
        ckpt_path = os.path.join(common.OUTDIR, f"03_pdqn_it{it}_90d_seed{seed}.pt")
        torch.save(policy.state_dict(), ckpt_path)
        results.append(dict(initial_trust=it, seed=seed, ckpt=ckpt_path, history=tr.history,
                            obs_dim=tr.obs_dim, critic_obs_dim=tr.critic_obs_dim, N=tr.N))
        common.save_json(results, "03_training_results_initialtrust90d.json")
        print(f"initial_trust={it} seed={seed} DONE, elapsed={time.time()-t0:.1f}s", flush=True)

print("ALL DONE (sapuan trust-awal x 90d), total elapsed", time.time() - t0, flush=True)
