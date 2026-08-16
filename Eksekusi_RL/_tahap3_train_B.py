import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import FormulaForecaster

# Lengan B (Tahap 3): trust DINAMIS PENUH -- TANPA constant_trust (satu2nya faktor yg
# beda dari Lengan A). Konfigurasi SISANYA identik Tahap 2 (02_config_beku.json).
CONFIG = dict(
    dataset_path=common.DATASET_KANONIK,
    n_train_seed=3,
    anggaran_chunk=300,
    k=2,
    rollout_steps=288,
    lr=1e-4,
    vf_coef=0.25,
    ent_coef=0.002,
    max_step_gap=4,
)
common.save_json(CONFIG, "03_config_B_beku.json")
print("CONFIG Lengan B (trust dinamis):", CONFIG, flush=True)

results = []
t0 = time.time()
for seed in range(CONFIG["n_train_seed"]):
    print(f"=== Lengan B train_seed={seed} (trust DINAMIS, no ablation) ===", flush=True)
    tr = TorchContinuingTrainer(
        CONFIG["dataset_path"], k=CONFIG["k"], rollout_steps=CONFIG["rollout_steps"],
        reward_calc=RewardCalculator.seimbang(), seed=seed, verbose=True,
        log_path=os.path.join(common.OUTDIR, f"03_train_log_B_seed{seed}.jsonl"),
        lr=CONFIG["lr"], vf_coef=CONFIG["vf_coef"], ent_coef=CONFIG["ent_coef"],
        max_step_gap=CONFIG["max_step_gap"])
    policy = tr.train(FormulaForecaster(), n_updates=CONFIG["anggaran_chunk"])
    ckpt_path = os.path.join(common.OUTDIR, f"03_pdqn_B_seed{seed}.pt")
    torch.save(policy.state_dict(), ckpt_path)
    results.append(dict(seed=seed, ckpt=ckpt_path, history=tr.history,
                        obs_dim=tr.obs_dim, critic_obs_dim=tr.critic_obs_dim, N=tr.N))
    common.save_json(results, "03_training_results_B.json")
    print(f"Lengan B seed={seed} DONE, elapsed={time.time()-t0:.1f}s", flush=True)

print("ALL DONE (Lengan B), total elapsed", time.time() - t0, flush=True)
