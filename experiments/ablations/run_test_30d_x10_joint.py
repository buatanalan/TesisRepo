import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Uji horizon 30 hari x 10 PASS, policy PPO dan forecaster dilatih BERSAMA (Mode D,
selaras train_mode_d di training.py): tiap chunk (1 hari = 96 step) sekaligus
(a) mengumpulkan pasangan supervised (fitur SPKLU saat spawn, virtual-wait) utk
forecaster dan (b) menjalankan rollout RL yang OBSERVASINYA memakai forecaster
TERKINI (bukan beku) -- lalu di akhir chunk: PPO update policy DAN partial_fit
forecaster dari data chunk itu. Forecaster mulai dari fallback formula (belum fit)
sampai partial_fit pertama, sesuai catatan train_mode_d.

"10 pass" = 10x lintasan penuh dataset scenario_dataset.json (2880 step = 30 hari).
Di batas tiap pass, simulasi di-reset (trust & state user kembali awal) TAPI
policy & forecaster TIDAK direset -- meniru semantik TorchContinuingTrainer utk
multi-pass, ditambah forecaster ikut belajar tiap chunk (gabungan Continuing + Mode D).
300 update total (10 pass x 30 hari).

Mengukur & mencatat 3 level performa sama seperti skrip sebelumnya, DITAMBAH kurva
per-pass (mengecek apakah performa pass ke-10 lebih baik dari pass ke-1 -- indikasi
transfer belajar policy+forecaster lintas pass, walau trust user direset tiap pass)
dan kurva forecaster (in-training MAE per hari, harus menurun jika forecaster belajar).
"""
import json
import math
from collections import defaultdict

import numpy as np
import torch

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import LearnedForecaster, extract_features
from marl_spklu.env.simulator import Simulator

SEED = 0
DATASET = "scenario_dataset.json"
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
N_UPDATES = DAYS_PER_PASS * N_PASSES   # 300

OUT_JSONL = "test_30d_x10_joint_log.jsonl"
OUT_SUMMARY = "test_30d_x10_joint_summary.json"


def collect_chunk_forecast_pairs(sim, step_start, step_end):
    """Kumpulkan (X, y) supervised dari event spawn dalam rentang step [start, end) --
    label = virtual wait (ground truth antrean), sama seperti TorchOnlineTrainer.train_step."""
    X_rows, y_rows = [], []
    sids = list(sim.spklus.keys())
    for step in range(step_start, step_end):
        time_now = step * sim.dt_minutes
        for spawn_tuple in sim.spawn_schedule.get(step, []):
            if len(spawn_tuple) == 3:
                user, spawn_loc, soc = spawn_tuple
            else:
                user, spawn_loc = spawn_tuple
                soc = 50.0
            for sid in sids:
                recent = sim.recent_recs.get(sid, 0)
                dist = float(math.dist(spawn_loc, sim.spklus[sid].location))
                feat = extract_features(sim.spklus[sid], time_now, user=user, soc=soc,
                                        recent_recs_count=recent, dist_override=dist)
                w = sim.compute_virtual_wait(user, sim.spklus[sid], time_now)
                if np.isfinite(w):
                    X_rows.append(feat)
                    y_rows.append(w)
    if not X_rows:
        return np.zeros((0, 19)), np.zeros(0)
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)


def main():
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, honest_estwait=True)
    forecaster = LearnedForecaster()   # belum fit -> fallback FormulaForecaster sampai partial_fit pertama

    day_records = []          # semua 300 chunk
    pass_records = []         # ringkas per-pass (10 baris)
    session_log_by_pass = defaultdict(list)
    decision_log_by_pass = defaultdict(list)

    global_day = 0
    with open(OUT_JSONL, "w") as flog:
        flog.write(json.dumps({"event": "config", "seed": SEED, "dataset": DATASET,
                               "forecaster": "LearnedForecaster (Mode D, online, joint)",
                               "chunk": CHUNK, "days_per_pass": DAYS_PER_PASS,
                               "n_passes": N_PASSES, "n_updates": N_UPDATES}) + "\n")

        for pass_idx in range(N_PASSES):
            sim = tr._fresh_sim()
            agent = RLRolloutAgent(tr.policy, sim, tr.rc, forecaster, tr.k,
                                   honest_estwait=tr.honest_estwait)
            sids = list(sim.spklus.keys())
            prev_sim_logs_len = 0
            step = 0

            for day_in_pass in range(DAYS_PER_PASS):
                step_start = step
                boundary = False
                for _ in range(CHUNK):
                    sim.step_once(step, agent=agent)
                    step += 1
                    if step >= sim.max_steps:
                        boundary = True
                        break
                step_end = step

                if boundary:
                    for t in agent.transitions:
                        t.resolved = True
                resolved = [t for t in agent.transitions if t.resolved]
                pending = [t for t in agent.transitions if not t.resolved]

                for t in resolved:
                    rec = {"pass": pass_idx, "day_in_pass": day_in_pass, "step": t.step,
                          "chosen_idx": t.chosen_idx, "chosen_spklu": sids[t.chosen_idx],
                          "delta": t.delta, "reward": t.reward, "complied": t.complied,
                          "disp_estwait": t.disp_estwait, "wait_default": t.wait_default,
                          "flock_penalty": t.flock_penalty}
                    decision_log_by_pass[pass_idx].append(rec)
                    flog.write(json.dumps({"event": "decision", **rec}) + "\n")

                new_sessions = sim.logs[prev_sim_logs_len:]
                prev_sim_logs_len = len(sim.logs)
                for s in new_sessions:
                    srec = {"pass": pass_idx, "day_in_pass": day_in_pass, **s}
                    session_log_by_pass[pass_idx].append(srec)
                    flog.write(json.dumps({"event": "session", **srec}) + "\n")

                # ---- forecaster: kumpulkan pasangan chunk ini & partial_fit ----
                X_chunk, y_chunk = collect_chunk_forecast_pairs(sim, step_start, step_end)
                fc_loss_before = fc_loss_after = None
                if len(X_chunk) > 0:
                    if forecaster.model is not None:
                        with torch.no_grad():
                            pred_before = forecaster.model(torch.tensor(X_chunk, dtype=torch.float32)).numpy()
                        fc_loss_before = float(np.mean((pred_before - y_chunk) ** 2))
                    forecaster.partial_fit(X_chunk, y_chunk)
                    with torch.no_grad():
                        pred_after = forecaster.model(torch.tensor(X_chunk, dtype=torch.float32)).numpy()
                    fc_loss_after = float(np.mean((pred_after - y_chunk) ** 2))

                # ---- PPO update policy ----
                day_rec = {"pass": pass_idx, "day_in_pass": day_in_pass, "global_day": global_day,
                          "n_forecast_pairs": int(len(X_chunk)),
                          "fc_mse_before": fc_loss_before, "fc_mse_after": fc_loss_after}
                if resolved:
                    stats = tr.ppo.update(resolved)
                    rewards = np.array([t.reward for t in resolved])
                    served = np.array([s.total_served for s in sim.spklus.values()], float)
                    trust_vals = [u.trust for u in sim.users]
                    day_rec.update({
                        "mean_reward": float(rewards.mean()),
                        "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                        "mean_flock_penalty": float(np.mean([t.flock_penalty for t in resolved])),
                        "mean_trust_allusers": float(np.mean(trust_vals)),
                        "n_transitions": len(resolved), "n_pending": len(pending),
                        "total_served_cum": int(served.sum()),
                        "approx_kl": stats.get("approx_kl", 0.0),
                        "explained_var": stats.get("explained_var", 0.0),
                        "entropy_final": stats.get("entropy_final", 0.0),
                    })
                day_records.append(day_rec)
                flog.write(json.dumps({"event": "day_summary", **day_rec}) + "\n")
                print(f"[Pass {pass_idx+1:02d}/{N_PASSES} Hari {day_in_pass+1:02d}/{DAYS_PER_PASS} "
                      f"(#{global_day+1:03d}/{N_UPDATES})] "
                      f"R={day_rec.get('mean_reward', 0):+.3f} "
                      f"accept={day_rec.get('acceptance_rate', 0):.2f} "
                      f"trust={day_rec.get('mean_trust_allusers', 0):.3f} "
                      f"fc_mse(before->after)={fc_loss_before}->{fc_loss_after} "
                      f"n_pairs={len(X_chunk)}")

                global_day += 1
                if boundary:
                    break
                agent.transitions = pending

            # ---- ringkasan per-pass ----
            recs_this_pass = [d for d in day_records if d["pass"] == pass_idx and "mean_reward" in d]
            sessions_this_pass = session_log_by_pass[pass_idx]
            pass_rec = {
                "pass": pass_idx,
                "mean_reward": float(np.mean([d["mean_reward"] for d in recs_this_pass])),
                "mean_acceptance": float(np.mean([d["acceptance_rate"] for d in recs_this_pass])),
                "trust_start": float(recs_this_pass[0]["mean_trust_allusers"]) if recs_this_pass else None,
                "trust_end": float(recs_this_pass[-1]["mean_trust_allusers"]) if recs_this_pass else None,
                "n_sessions": len(sessions_this_pass),
                "total_served": recs_this_pass[-1]["total_served_cum"] if recs_this_pass else 0,
            }
            if sessions_this_pass:
                pred = np.array([s["est_wait"] for s in sessions_this_pass])
                act = np.array([s["wait_time"] for s in sessions_this_pass])
                err = pred - act
                pass_rec.update({
                    "forecaster_mae": float(np.mean(np.abs(err))),
                    "forecaster_bias": float(np.mean(err)),
                    "forecaster_corr": float(np.corrcoef(pred, act)[0, 1]) if len(pred) > 1 else None,
                })
            pass_records.append(pass_rec)
            print(f"\n=== Pass {pass_idx+1}/{N_PASSES} selesai: R={pass_rec['mean_reward']:+.3f} "
                  f"accept={pass_rec['mean_acceptance']:.2f} trust {pass_rec['trust_start']:.3f}->"
                  f"{pass_rec['trust_end']:.3f} fc_MAE={pass_rec.get('forecaster_mae')} ===\n")

    # ================= ANALISIS AKHIR =================
    all_sessions = [s for lst in session_log_by_pass.values() for s in lst]

    # ---- 1. Performa keseluruhan ----
    valid_days = [d for d in day_records if "mean_reward" in d]
    overall = {
        "n_passes": N_PASSES, "days_per_pass": DAYS_PER_PASS, "n_updates": len(valid_days),
        "reward_pass1_mean": pass_records[0]["mean_reward"],
        "reward_pass10_mean": pass_records[-1]["mean_reward"],
        "acceptance_pass1_mean": pass_records[0]["mean_acceptance"],
        "acceptance_pass10_mean": pass_records[-1]["mean_acceptance"],
        "trust_end_pass1": pass_records[0]["trust_end"],
        "trust_end_pass10": pass_records[-1]["trust_end"],
        "served_pass1": pass_records[0]["total_served"],
        "served_pass10": pass_records[-1]["total_served"],
    }

    # ---- 2. Performa individu: bandingkan pass pertama vs terakhir ----
    def individual_stats(sessions):
        by_user = defaultdict(list)
        for s in sessions:
            by_user[s["user"]].append(s)
        n_up = n_down = n_flat = 0
        for uid, recs in by_user.items():
            t0, t1 = recs[0]["trust_after"], recs[-1]["trust_after"]
            if t1 > t0 + 0.01:
                n_up += 1
            elif t1 < t0 - 0.01:
                n_down += 1
            else:
                n_flat += 1
        acc = [np.mean([r["complied"] for r in recs]) for recs in by_user.values() if len(recs) >= 2]
        return {"n_users": len(by_user), "trust_naik": n_up, "trust_turun": n_down,
               "trust_flat": n_flat, "mean_acceptance_per_user": float(np.mean(acc)) if acc else None}

    individual = {
        "pass1": individual_stats(session_log_by_pass[0]),
        "pass10": individual_stats(session_log_by_pass[N_PASSES - 1]),
    }

    # ---- 3. Performa forecaster lintas pass ----
    forecaster_perf = {
        "mae_per_pass": [p.get("forecaster_mae") for p in pass_records],
        "bias_per_pass": [p.get("forecaster_bias") for p in pass_records],
        "corr_per_pass": [p.get("forecaster_corr") for p in pass_records],
        "mae_pass1": pass_records[0].get("forecaster_mae"),
        "mae_pass10": pass_records[-1].get("forecaster_mae"),
    }

    summary = {"overall": overall, "individual": individual, "forecaster": forecaster_perf,
              "pass_records": pass_records, "day_records": day_records}
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 1. PERFORMA KESELURUHAN (pass 1 vs pass 10) ===")
    print(json.dumps(overall, indent=2))
    print("\n=== 2. PERFORMA INDIVIDU (pass 1 vs pass 10) ===")
    print(json.dumps(individual, indent=2))
    print("\n=== 3. PERFORMA FORECASTER per-pass ===")
    print(json.dumps(forecaster_perf, indent=2))
    print(f"\n[INFO] Log rinci -> {OUT_JSONL}")
    print(f"[INFO] Ringkasan -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
