import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Ablasi: latih agen HPPO 30 hari/seed=0 TANPA forecaster yang perlu dilatih/diprediksi --
diganti "OracleForecaster" yang langsung memanggil sim.compute_virtual_wait() (mekanisme
antrean virtual milik simulator sendiri, deterministik, tanpa model ML sama sekali).

Tujuan: mengisolasi apakah performa agen sebelumnya (FormulaForecaster, test_30d_seed0_summary.json)
tertahan OLEH ketidakakuratan forecaster, atau memang policy-nya sendiri yang jadi bottleneck.
OracleForecaster memberi observasi & EstWait SEDEKAT MUNGKIN ke ground truth (bukan prediksi
belajar) -- ini bukan "coba tanpa fitur wait sama sekali", tapi "coba tanpa komponen prediktor
yang bisa salah", supaya adil dibandingkan dgn baseline yg sudah ada (arsitektur observasi &
reward sama persis, cuma sumber wait-estimate yang beda).

Dataset, seed, horizon, chunk SAMA PERSIS dengan run_test_30d_single_seed.py supaya selisih
hasil murni disebabkan sumber prediksi wait.
"""
import json
from collections import defaultdict

import numpy as np

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import ForecasterBase

SEED = 0
DATASET = "scenario_dataset.json"
CHUNK = 96
N_UPDATES = 30

OUT_JSONL = "test_30d_seed0_oracle_log.jsonl"
OUT_SUMMARY = "test_30d_seed0_oracle_summary.json"


class OracleForecaster(ForecasterBase):
    """Tanpa parameter/pelatihan -- langsung pakai antrean virtual simulator (ground truth
    hipotetis, bukan model ML). 'Prediksi sempurna' sejauh mekanisme antrean virtual itu sendiri
    akurat (durasi charge EV lain masih diaproksimasi mean_charge_time, tapi state antrean &
    komitmen traveling EV riil, bukan estimasi formula sederhana)."""

    def predict(self, spklus: dict, time_now_min: float = 0.0, user=None, soc: float = 50.0,
               sim=None) -> dict:
        if sim is None:
            return {sid: 0.0 for sid in spklus}
        return {sid: float(sim.compute_virtual_wait(user, s, time_now_min))
               for sid, s in spklus.items()}


def main():
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, honest_estwait=True)
    forecaster = OracleForecaster()
    sim = tr._fresh_sim()
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
                               "forecaster": "OracleForecaster (compute_virtual_wait, no ML)",
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

            new_sessions = sim.logs[prev_sim_logs_len:]
            prev_sim_logs_len = len(sim.logs)
            for s in new_sessions:
                srec = {"day": day, **s}
                session_log.append(srec)
                flog.write(json.dumps({"event": "session", **srec}) + "\n")

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

    overall = {
        "seed": SEED, "dataset": DATASET, "horizon_days": len(day_records),
        "n_decisions": len(decision_log), "n_sessions": len(session_log),
        "reward_first%d_days_mean" % n_report: _seg_mean("mean_reward", slice(0, n_report)),
        "reward_last%d_days_mean" % n_report: _seg_mean("mean_reward", slice(-n_report, None)),
        "acceptance_first%d_days_mean" % n_report: _seg_mean("acceptance_rate", slice(0, n_report)),
        "acceptance_last%d_days_mean" % n_report: _seg_mean("acceptance_rate", slice(-n_report, None)),
        "trust_first%d_days_mean" % n_report: _seg_mean("mean_trust_allusers", slice(0, n_report)),
        "trust_last%d_days_mean" % n_report: _seg_mean("mean_trust_allusers", slice(-n_report, None)),
        "entropy_first%d_mean" % n_report: _seg_mean("entropy_final", slice(0, n_report)),
        "entropy_last%d_mean" % n_report: _seg_mean("entropy_final", slice(-n_report, None)),
        "explained_var_mean": float(np.mean([d["explained_var"] for d in day_records])),
        "total_served_final": day_records[-1]["total_served_cum"] if day_records else 0,
        "mean_flock_penalty_mean": float(np.mean([d["mean_flock_penalty"] for d in day_records])),
    }

    sessions_by_user = defaultdict(list)
    for s in session_log:
        sessions_by_user[s["user"]].append(s)
    n_up = n_down = n_flat = 0
    for uid, recs in sessions_by_user.items():
        t0, t1 = recs[0]["trust_after"], recs[-1]["trust_after"]
        if t1 > t0 + 0.01:
            n_up += 1
        elif t1 < t0 - 0.01:
            n_down += 1
        else:
            n_flat += 1
    individual = {"n_users_with_sessions": len(sessions_by_user),
                 "trust_naik": n_up, "trust_turun": n_down, "trust_flat": n_flat}

    pred = np.array([s["est_wait"] for s in session_log])
    act = np.array([s["wait_time"] for s in session_log])
    err = pred - act
    forecaster_perf = {
        "n": int(len(pred)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_pred_minus_act": float(np.mean(err)),
        "corr": float(np.corrcoef(pred, act)[0, 1]) if len(pred) > 1 else None,
        "mean_actual_wait": float(act.mean()),
        "mean_pred_wait": float(pred.mean()),
    }

    summary = {"overall": overall, "individual": individual, "forecaster": forecaster_perf,
              "day_records": day_records}
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 1. PERFORMA KESELURUHAN (Oracle, no-ML forecaster) ===")
    print(json.dumps(overall, indent=2))
    print("\n=== 2. PERFORMA INDIVIDU ===")
    print(json.dumps(individual, indent=2))
    print("\n=== 3. AKURASI ORACLE (vs realisasi) ===")
    print(json.dumps(forecaster_perf, indent=2))
    print(f"\n[INFO] Log rinci -> {OUT_JSONL}")
    print(f"[INFO] Ringkasan -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
