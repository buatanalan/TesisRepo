import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Uji horizon 30 hari, satu seed, agen HPPO DENGAN forecaster TERLATIH (Mode B):
LearnedForecaster (MLP PyTorch) di-pretrain offline dari trajektori baseline GreedyAgent
di dataset yang sama (train_mode_b), lalu DIBEKUKAN (freeze) sebelum policy dilatih --
selaras training.train_mode_b (gradient forecaster & policy tidak dibagi -> tak ada
non-stationarity antar-pembelajar RL).

Perbandingan dengan run_test_30d_single_seed.py (FormulaForecaster, tak terlatih):
skrip ini pakai dataset & seed yang SAMA (scenario_dataset.json, seed=0, 30 hari) supaya
selisih hasil murni disebabkan forecaster, bukan variasi acak lain.

Mengukur 3 level performa di akhir run:
  1. KESELURUHAN (agregat 30 hari): reward, acceptance, trust rata-rata, gini, served,
     herding, MAE/bias forecaster -- kurva 5-hari-awal vs 5-hari-akhir (indikator belajar).
  2. INDIVIDU (per user): distribusi arah perubahan trust (naik/turun/flat), acceptance
     paruh-awal vs paruh-akhir riwayat tiap user.
  3. FORECASTER: (a) in-sample fit MAE saat pretraining offline (Mode B), (b) akurasi
     realized (est_wait vs wait_time aktual) per hari selama 30 hari online -- dibandingkan
     terhadap baseline FormulaForecaster dari run sebelumnya bila filenya ada.
"""
import json
from collections import defaultdict

import numpy as np
import torch

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import LearnedForecaster, collect_forecast_dataset
from marl_spklu.rl.training import _fresh_sim
from marl_spklu.agents.greedy_agent import GreedyAgent

SEED = 0
DATASET = "scenario_dataset.json"
CHUNK = 96
N_UPDATES = 30
PRETRAIN_STEPS = 2880   # horizon penuh dataset -> forecaster lihat pola sepanjang 30 hari

OUT_JSONL = "test_30d_seed0_learnedfc_log.jsonl"
OUT_SUMMARY = "test_30d_seed0_learnedfc_summary.json"


def pretrain_forecaster():
    """Mode B, tahap 1: kumpulkan (X, y) dari rollout GreedyAgent, fit LearnedForecaster,
    kembalikan (forecaster beku, in-sample MAE) untuk pelaporan performa forecaster."""
    sim = _fresh_sim(DATASET)
    X, y = collect_forecast_dataset(sim, min(PRETRAIN_STEPS, sim.max_steps), agent=GreedyAgent())
    forecaster = LearnedForecaster().fit(X, y)

    with torch.no_grad():
        pred = forecaster.model(torch.tensor(X, dtype=torch.float32)).numpy()
    err = pred - y
    fit_stats = {
        "n_pairs": int(X.shape[0]),
        "mae_in_sample": float(np.mean(np.abs(err))),
        "rmse_in_sample": float(np.sqrt(np.mean(err ** 2))),
        "bias_in_sample": float(np.mean(err)),
        "corr_in_sample": float(np.corrcoef(pred, y)[0, 1]) if len(y) > 1 else float("nan"),
        "mean_y": float(np.mean(y)),
        "mean_pred": float(np.mean(pred)),
    }
    return forecaster, fit_stats


def main():
    print("=== Tahap 1: pretrain LearnedForecaster offline (Mode B, baseline GreedyAgent) ===")
    forecaster, fit_stats = pretrain_forecaster()
    print("Fit forecaster (in-sample):", json.dumps(fit_stats, indent=2))

    print("\n=== Tahap 2: latih policy HPPO 30 hari (continuing task) dengan forecaster BEKU ===")
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, honest_estwait=True)
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
                               "forecaster": "LearnedForecaster(Mode B, frozen)",
                               "fit_stats": fit_stats, "n_spklu": len(sids),
                               "n_users": len(sim.users), "chunk": CHUNK,
                               "n_updates": N_UPDATES}) + "\n")

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

    # ================= ANALISIS AKHIR =================
    n_report = min(5, len(day_records))

    def _seg_mean(key, sl):
        vals = [d[key] for d in day_records[sl]]
        return float(np.mean(vals)) if vals else None

    # ---- 1. Performa keseluruhan ----
    overall = {
        "seed": SEED, "dataset": DATASET, "horizon_days": len(day_records),
        "n_decisions": len(decision_log), "n_sessions": len(session_log),
        "reward_first%d_days_mean" % n_report: _seg_mean("mean_reward", slice(0, n_report)),
        "reward_last%d_days_mean" % n_report: _seg_mean("mean_reward", slice(-n_report, None)),
        "acceptance_first%d_days_mean" % n_report: _seg_mean("acceptance_rate", slice(0, n_report)),
        "acceptance_last%d_days_mean" % n_report: _seg_mean("acceptance_rate", slice(-n_report, None)),
        "trust_first%d_days_mean" % n_report: _seg_mean("mean_trust_allusers", slice(0, n_report)),
        "trust_last%d_days_mean" % n_report: _seg_mean("mean_trust_allusers", slice(-n_report, None)),
        "explained_var_mean": float(np.mean([d["explained_var"] for d in day_records])),
        "entropy_first%d_mean" % n_report: _seg_mean("entropy_final", slice(0, n_report)),
        "entropy_last%d_mean" % n_report: _seg_mean("entropy_final", slice(-n_report, None)),
        "total_served_final": day_records[-1]["total_served_cum"] if day_records else 0,
    }

    # ---- 2. Performa individu (per user) ----
    sessions_by_user = defaultdict(list)
    for s in session_log:
        sessions_by_user[s["user"]].append(s)

    n_up = n_down = n_flat = 0
    delta_trust_all = []
    for uid, recs in sessions_by_user.items():
        t0, t1 = recs[0]["trust_after"], recs[-1]["trust_after"]
        delta_trust_all.append(t1 - t0)
        if t1 > t0 + 0.01:
            n_up += 1
        elif t1 < t0 - 0.01:
            n_down += 1
        else:
            n_flat += 1

    complied_updaters = []
    for uid, recs in sessions_by_user.items():
        comp = [r for r in recs if r["complied"]]
        if len(comp) >= 2:
            complied_updaters.append(comp[-1]["trust_after"] - comp[0]["trust_after"])

    acc_first, acc_last = [], []
    for uid, recs in sessions_by_user.items():
        if len(recs) >= 4:
            half = len(recs) // 2
            acc_first.append(np.mean([r["complied"] for r in recs[:half]]))
            acc_last.append(np.mean([r["complied"] for r in recs[half:]]))

    individual = {
        "n_users_with_sessions": len(sessions_by_user),
        "trust_naik": n_up, "trust_turun": n_down, "trust_flat": n_flat,
        "mean_delta_trust_per_user": float(np.mean(delta_trust_all)) if delta_trust_all else None,
        "n_users_ge2_complied_sessions": len(complied_updaters),
        "mean_delta_trust_among_complied_updaters": float(np.mean(complied_updaters)) if complied_updaters else None,
        "n_users_ge4_sessions": len(acc_first),
        "acceptance_first_half_mean": float(np.mean(acc_first)) if acc_first else None,
        "acceptance_last_half_mean": float(np.mean(acc_last)) if acc_last else None,
    }

    # ---- 3. Performa forecaster (realized, online, per hari) ----
    by_day_sessions = defaultdict(list)
    for s in session_log:
        by_day_sessions[s["day"]].append(s)

    fc_daily = []
    all_pred, all_act = [], []
    for day in sorted(by_day_sessions):
        recs = by_day_sessions[day]
        pred = np.array([r["est_wait"] for r in recs])
        act = np.array([r["wait_time"] for r in recs])
        all_pred.append(pred); all_act.append(act)
        err = pred - act
        corr = float(np.corrcoef(pred, act)[0, 1]) if len(pred) > 1 else None
        fc_daily.append({
            "day": day, "n": len(recs),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias_pred_minus_act": float(np.mean(err)),
            "corr": corr,
            "underpromise_rate": float(np.mean(act > pred)),
        })
    pred_all = np.concatenate(all_pred); act_all = np.concatenate(all_act)
    err_all = pred_all - act_all
    forecaster_perf = {
        "offline_fit_stats": fit_stats,
        "online_n": int(len(pred_all)),
        "online_mae": float(np.mean(np.abs(err_all))),
        "online_rmse": float(np.sqrt(np.mean(err_all ** 2))),
        "online_bias_pred_minus_act": float(np.mean(err_all)),
        "online_corr": float(np.corrcoef(pred_all, act_all)[0, 1]),
        "online_mean_actual_wait": float(act_all.mean()),
        "online_mean_pred_wait": float(pred_all.mean()),
        "online_mae_first%d_days_mean" % n_report: float(np.mean([d["mae"] for d in fc_daily[:n_report]])),
        "online_mae_last%d_days_mean" % n_report: float(np.mean([d["mae"] for d in fc_daily[-n_report:]])),
        "daily": fc_daily,
    }

    summary = {
        "overall": overall,
        "individual": individual,
        "forecaster": forecaster_perf,
        "day_records": day_records,
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 1. PERFORMA KESELURUHAN ===")
    print(json.dumps(overall, indent=2))
    print("\n=== 2. PERFORMA INDIVIDU (per user) ===")
    print(json.dumps(individual, indent=2))
    print("\n=== 3. PERFORMA FORECASTER ===")
    print(json.dumps({k: v for k, v in forecaster_perf.items() if k != "daily"}, indent=2))
    print(f"\n[INFO] Log rinci -> {OUT_JSONL}")
    print(f"[INFO] Ringkasan -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
