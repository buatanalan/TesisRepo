import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Uji horizon 30 hari, satu seed, untuk agen MARL (HPPO) -- mengecek apakah agen
BELAJAR selama training continuing-task 30 hari (2880 step, chunk 96 step = 1 hari).

scenario_dataset.json sudah berhorizon 2880 step (=30 hari x 96 step) sehingga
30 update chunk = tepat SATU pass, satu seed, tanpa reset trust/state di tengah run
(pas dengan definisi TorchContinuingTrainer: reset hanya terjadi di batas horizon).

Mencatat 3 level log ke JSONL (test_30d_seed0_log.jsonl):
  - "decision": tiap transisi RL (aksi a1=SPKLU dipilih, a2=delta EstWait, reward,
    complied, disp_estwait, value, logp) saat resolve (sesi user selesai).
  - "session": tiap sesi charging selesai (trust_after, wait_time, est_wait, complied)
    persis field yang direkam Simulator ke sim.logs.
  - "day_summary": agregat harian (reward, acceptance, trust rata2 semua user, KL,
    explained_var, entropy, dst dari PPOTrainer.update) -- untuk menilai kurva belajar.

Ringkasan akhir (test_30d_seed0_summary.json) membandingkan 5 hari pertama vs 5 hari
terakhir pada reward/acceptance/trust sebagai indikator kasar apakah agen membaik.
"""
import json
import numpy as np

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster

SEED = 0
DATASET = "scenario_dataset.json"
CHUNK = 96          # 1 hari = 96 step (dt=15 menit)
N_UPDATES = 30       # 30 hari; dataset max_steps=2880 = 30*96 -> tepat satu pass, satu seed

OUT_JSONL = "test_30d_seed0_log.jsonl"
OUT_SUMMARY = "test_30d_seed0_summary.json"


def main():
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, honest_estwait=True)
    forecaster = FormulaForecaster()
    sim = tr._fresh_sim()
    if sim.max_steps != N_UPDATES * CHUNK:
        print(f"[WARN] dataset max_steps={sim.max_steps}, diharapkan {N_UPDATES * CHUNK} "
              f"(30 hari x 96). Lanjut, tapi run bisa berhenti lebih awal/telat.")

    agent = RLRolloutAgent(tr.policy, sim, tr.rc, forecaster, tr.k,
                           honest_estwait=tr.honest_estwait)
    sids = list(sim.spklus.keys())

    day_records = []
    decision_log = []
    session_log = []
    prev_sim_logs_len = 0
    step = 0

    with open(OUT_JSONL, "w") as flog:
        flog.write(json.dumps({"event": "config", "seed": SEED, "dataset": DATASET,
                               "n_spklu": len(sids), "n_users": len(sim.users),
                               "chunk": CHUNK, "n_updates": N_UPDATES}) + "\n")

        for day in range(N_UPDATES):
            boundary = False
            for _ in range(CHUNK):
                sim.step_once(step, agent=agent)
                step += 1
                if step >= sim.max_steps:
                    boundary = True
                    break
            if boundary:
                for t in agent.transitions:
                    t.resolved = True

            resolved = [t for t in agent.transitions if t.resolved]
            pending = [t for t in agent.transitions if not t.resolved]

            # ---- log per-keputusan (aksi + reward) ----
            for t in resolved:
                rec = {
                    "day": day, "step": t.step,
                    "chosen_idx": t.chosen_idx, "chosen_spklu": sids[t.chosen_idx],
                    "delta": t.delta, "reward": t.reward, "complied": t.complied,
                    "disp_estwait": t.disp_estwait, "wait_default": t.wait_default,
                    "flock_penalty": t.flock_penalty, "value": t.value, "logp": t.logp,
                }
                decision_log.append(rec)
                flog.write(json.dumps({"event": "decision", **rec}) + "\n")

            # ---- log per-sesi selesai (trust) ----
            new_sessions = sim.logs[prev_sim_logs_len:]
            prev_sim_logs_len = len(sim.logs)
            for s in new_sessions:
                srec = {"day": day, **s}
                session_log.append(srec)
                flog.write(json.dumps({"event": "session", **srec}) + "\n")

            # ---- update PPO + ringkasan harian ----
            if resolved:
                stats = tr.ppo.update(resolved)
                rewards = np.array([t.reward for t in resolved])
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                trust_vals = [u.trust for u in sim.users]
                day_rec = {
                    "day": day,
                    "mean_reward": float(rewards.mean()),
                    "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                    "mean_flock_penalty": float(np.mean([t.flock_penalty for t in resolved])),
                    "mean_trust_allusers": float(np.mean(trust_vals)),
                    "median_trust_allusers": float(np.median(trust_vals)),
                    "n_transitions": len(resolved),
                    "n_pending": len(pending),
                    "total_served_cum": int(served.sum()),
                    "approx_kl": stats.get("approx_kl", 0.0),
                    "clip_frac": stats.get("clip_frac", 0.0),
                    "explained_var": stats.get("explained_var", 0.0),
                    "entropy_final": stats.get("entropy_final", 0.0),
                    "grad_norm": stats.get("grad_norm", 0.0),
                    "pi_loss": stats.get("pi_loss", 0.0),
                    "v_loss": stats.get("v_loss", 0.0),
                }
                day_records.append(day_rec)
                flog.write(json.dumps({"event": "day_summary", **day_rec}) + "\n")
                print(f"[Hari {day + 1:02d}/{N_UPDATES}] R={day_rec['mean_reward']:+.4f} "
                      f"accept={day_rec['acceptance_rate']:.2f} "
                      f"trust={day_rec['mean_trust_allusers']:.3f} "
                      f"KL={day_rec['approx_kl']:.4f} EV={day_rec['explained_var']:.2f} "
                      f"ent={day_rec['entropy_final']:.2f} served={day_rec['total_served_cum']}")

            if boundary:
                break
            agent.transitions = pending

    n_report = min(5, len(day_records))

    def _seg_mean(key, sl):
        vals = [d[key] for d in day_records[sl]]
        return float(np.mean(vals)) if vals else None

    summary = {
        "seed": SEED, "dataset": DATASET, "horizon_days": len(day_records),
        "n_decisions": len(decision_log), "n_sessions": len(session_log),
        "reward_first%d_days_mean" % n_report: _seg_mean("mean_reward", slice(0, n_report)),
        "reward_last%d_days_mean" % n_report: _seg_mean("mean_reward", slice(-n_report, None)),
        "acceptance_first%d_days_mean" % n_report: _seg_mean("acceptance_rate", slice(0, n_report)),
        "acceptance_last%d_days_mean" % n_report: _seg_mean("acceptance_rate", slice(-n_report, None)),
        "trust_first%d_days_mean" % n_report: _seg_mean("mean_trust_allusers", slice(0, n_report)),
        "trust_last%d_days_mean" % n_report: _seg_mean("mean_trust_allusers", slice(-n_report, None)),
        "day_records": day_records,
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RINGKASAN (indikator belajar: first-N vs last-N hari) ===")
    for k, v in summary.items():
        if k != "day_records":
            print(f"  {k}: {v}")
    print(f"\n[INFO] Log rinci -> {OUT_JSONL}")
    print(f"[INFO] Ringkasan -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
