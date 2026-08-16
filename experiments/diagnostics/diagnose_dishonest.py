import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""
diagnose_dishonest.py
Melatih S5_HONEST dan S5_DISHONEST dengan logging per-update penuh,
lalu mencetak tabel progressi metrik kunci setiap 10 update.

Output: diagnosis_progress.json (per-update detail), ringkasan tabel di stdout.
"""
import json, os, random
import numpy as np
import torch

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.training import train_mode_a, _fresh_sim
from marl_spklu.rl.rollout import evaluate_policy
from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.rl.rewards import RewardCalculator

TRAIN_DS  = "scenario_train_1w.json"
TEST_DS   = "scenario_test_1d.json"
SEED      = 0
UPDATES   = 210

def get_test_stats(sim):
    waits    = [log["wait_time"]   for log in sim.logs]
    trusts   = [log["trust_after"] for log in sim.logs]
    complied = [log["complied"]    for log in sim.logs]
    served   = np.array([s.total_served for s in sim.spklus.values()], float)
    def _gini(a):
        a = np.clip(a, 0, None)
        if a.sum() == 0: return 0.0
        a = np.sort(a); n = len(a); idx = np.arange(1, n+1)
        return float(np.sum((2*idx - n - 1)*a) / (n*a.sum()))
    return {
        "gini_served":     _gini(served),
        "total_served":    int(served.sum()),
        "n_active":        int((served > 0).sum()),
        "mean_wait":       float(np.mean(waits))    if waits    else 0.0,
        "mean_trust":      float(np.mean(trusts))   if trusts   else 0.5,
        "acceptance_rate": float(np.mean(complied)) if complied else 0.0,
    }

def eval_on_test(policy, forecaster, honest):
    sim = _fresh_sim(TEST_DS)
    sim.log_actor_states = True
    sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=96)
    evaluate_policy(sim, policy, forecaster, k=3, max_steps=96, honest_estwait=honest)
    return get_test_stats(sim)

def train_with_progress(label, honest, rc):
    """Latih dan kumpulkan history per-update + tes setiap 30 update."""
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    tr = TorchContinuingTrainer(
        TRAIN_DS, k=3, rollout_steps=96, seed=SEED, verbose=True,
        reward_calc=rc, lr=1e-4, ent_coef=0.01, gamma=0.99,
        minibatch=64, epochs=4, honest_estwait=honest
    )
    # Ambil forecaster dari train_mode_a tapi jalankan manual agar kita bisa cek progress
    from marl_spklu.rl.training import train_mode_a
    from marl_spklu.rl.forecaster import FormulaForecaster
    forecaster = FormulaForecaster()

    checkpoints = []  # (update, train_history_slice, test_stats)

    # Latih dalam blok 30 update, cek di setiap blok
    block_size = 30
    for blk in range(UPDATES // block_size):
        tr.train(forecaster, n_updates=block_size)
        update_no = (blk + 1) * block_size
        # Snapshot history blok ini
        blk_history = tr.history[blk * block_size : (blk + 1) * block_size]
        mean_reward  = float(np.mean([h.get("mean_reward", 0)      for h in blk_history]))
        mean_accept  = float(np.mean([h.get("acceptance_rate", 0)  for h in blk_history]))
        mean_gini    = float(np.mean([h.get("gini_served", 0)      for h in blk_history]))
        mean_kl      = float(np.mean([h.get("approx_kl", 0)        for h in blk_history]))
        mean_entropy = float(np.mean([h.get("entropy_final", 0)    for h in blk_history]))
        mean_ev      = float(np.mean([h.get("explained_var", -1)   for h in blk_history]))
        train_summary = {
            "update": update_no,
            "mean_reward": mean_reward,
            "acceptance_rate": mean_accept,
            "gini_train": mean_gini,
            "approx_kl": mean_kl,
            "entropy": mean_entropy,
            "explained_var": mean_ev,
        }
        # Evaluasi pada test dataset
        test_stats = eval_on_test(tr.policy, forecaster, honest=honest)
        checkpoints.append({**train_summary, "test": test_stats})
        print(f"\n[{label}] Update {update_no:3d} | "
              f"R={mean_reward:+.4f} | acc_train={mean_accept:.3f} | gini_train={mean_gini:.3f} | "
              f"KL={mean_kl:.4f} | Ent={mean_entropy:.3f} | EV={mean_ev:.3f}")
        print(f"           TEST => gini={test_stats['gini_served']:.4f} | "
              f"trust={test_stats['mean_trust']:.4f} | acc={test_stats['acceptance_rate']:.4f} | "
              f"wait={test_stats['mean_wait']:.3f} | n_active={test_stats['n_active']}")

    return tr, forecaster, checkpoints


def main():
    if not os.path.exists(TRAIN_DS) or not os.path.exists(TEST_DS):
        print("Dataset tidak ditemukan. Jalankan generate_train_test.py terlebih dahulu.")
        return

    rc = RewardCalculator(alpha_wait=1.0, alpha_fair=0.3, alpha_accept=0.1,
                          alpha_shape=0.5, xi=0.5)

    print("\n" + "="*70)
    print("DIAGNOSIS: S5_HONEST — Progressi 210 Update (cek setiap 30)")
    print("="*70)
    _, _, cp_honest = train_with_progress("S5_HONEST", honest=True, rc=rc)

    print("\n" + "="*70)
    print("DIAGNOSIS: S5_DISHONEST — Progressi 210 Update (cek setiap 30)")
    print("="*70)
    _, _, cp_dishonest = train_with_progress("S5_DISHONEST", honest=False, rc=rc)

    # ---- Tabel ringkasan ----
    print("\n\n" + "="*90)
    print("TABEL PROGRESSI: TEST Gini | Trust | Acceptance | Mean Wait")
    print(f"{'Update':>6} | {'Honest Gini':>12} | {'Dishon Gini':>12} | "
          f"{'Hon Trust':>10} | {'Dis Trust':>10} | {'Hon Acc':>8} | {'Dis Acc':>8} | "
          f"{'Dis Wait':>9}")
    print("-"*90)
    for h, d in zip(cp_honest, cp_dishonest):
        print(f"{h['update']:>6} | {h['test']['gini_served']:>12.4f} | "
              f"{d['test']['gini_served']:>12.4f} | "
              f"{h['test']['mean_trust']:>10.4f} | {d['test']['mean_trust']:>10.4f} | "
              f"{h['test']['acceptance_rate']:>8.4f} | {d['test']['acceptance_rate']:>8.4f} | "
              f"{d['test']['mean_wait']:>9.4f}")
    print("="*90)

    # ---- Tabel diagnostik training ----
    print("\n\nDIAGNOSTIK TRAINING S5_DISHONEST (KL, Entropy, EV)")
    print(f"{'Update':>6} | {'Reward':>8} | {'Acc_train':>10} | "
          f"{'ApproxKL':>9} | {'Entropy':>8} | {'ExpVar':>8} | {'Gini_train':>10}")
    print("-"*70)
    for d in cp_dishonest:
        print(f"{d['update']:>6} | {d['mean_reward']:>8.4f} | {d['acceptance_rate']:>10.4f} | "
              f"{d['approx_kl']:>9.5f} | {d['entropy']:>8.4f} | {d['explained_var']:>8.4f} | "
              f"{d['gini_train']:>10.4f}")
    print("="*70)

    # Simpan data mentah
    output = {"S5_HONEST": cp_honest, "S5_DISHONEST": cp_dishonest}
    with open("diagnosis_progress.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nDisimpan ke: diagnosis_progress.json")


if __name__ == "__main__":
    main()
