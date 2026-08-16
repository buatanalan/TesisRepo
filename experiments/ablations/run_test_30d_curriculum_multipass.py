import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Versi MULTI-PASS dari run_test_30d_curriculum.py -- sebelumnya cuma 1 pass (30 hari,
sekali lintas dataset). Di sini N_PASSES=5 (150 hari total) supaya policy & kurikulum
fairness punya lebih banyak waktu berkembang.

Beda penting dari versi 1-pass: skedul kurikulum kini dibentangkan di seluruh 150 hari
(global_day/149), BUKAN reset ke 0 tiap pass -- makna "curriculum" yang sebenarnya adalah
menaikkan kesulitan bertahap sepanjang TOTAL training, bukan berulang tiap lintasan dataset.

Di batas tiap pass (per 30 hari): sim direset (state user/trust segar), TAPI:
  - Policy PPO TIDAK direset (bobot terus dibawa lintas pass, seperti TorchContinuingTrainer
    multi-pass & run_test_30d_x10_joint.py sebelumnya).
  - init_trust (utk varian yg pakai) di-reapply tiap pass baru (representasi kohort user
    baru yang di-bootstrap trust-nya, konsisten dgn semantik intervensi "trust awal tinggi").

4 varian sama seperti sebelumnya, kini x5 pass:
  1. baseline            : trust=0.5, fairness statis penuh
  2. high_trust_static   : trust=1.0, fairness statis penuh sejak awal
  3. fairness_off_static : trust=1.0, fairness OFF selamanya
  4. curriculum          : trust=1.0, fairness naik LINEAR 0->penuh sepanjang 150 hari (global)
"""
import json

import numpy as np

from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator

DATASET = "scenario_dataset_60d.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 60
N_PASSES = 10
TOTAL_DAYS = DAYS_PER_PASS * N_PASSES   # 150
FULL_ALPHA_GINI = 0.5
FULL_ALPHA_FLOCK = 0.3


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def run_variant(name, init_trust, fairness_schedule):
    """fairness_schedule(global_day) -> (alpha_gini, alpha_flock), global_day in [0, TOTAL_DAYS)."""
    print(f"\n{'=' * 70}\n=== VARIAN: {name} (init_trust={init_trust}, {N_PASSES} pass x "
         f"{DAYS_PER_PASS} hari = {TOTAL_DAYS} hari) ===\n{'=' * 70}")
    rc = RewardCalculator(alpha_gini=0.0, alpha_flock=0.0)
    tr = TorchContinuingTrainer(DATASET, k=3, rollout_steps=CHUNK, seed=SEED,
                                verbose=False, reward_calc=rc, honest_estwait=True)
    forecaster = FormulaForecaster()

    day_records = []
    pass_records = []
    global_day = 0

    for pass_idx in range(N_PASSES):
        sim = tr._fresh_sim()
        if init_trust is not None:
            for u in sim.users:
                u.trust = float(init_trust)
        agent = RLRolloutAgent(tr.policy, sim, tr.rc, forecaster, tr.k, honest_estwait=tr.honest_estwait)

        step = 0
        for day_in_pass in range(DAYS_PER_PASS):
            ag, af = fairness_schedule(day_in_pass)
            tr.rc.alpha_gini = ag
            tr.rc.alpha_flock = af

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

            if resolved:
                stats = tr.ppo.update(resolved)
                rewards = np.array([t.reward for t in resolved])
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                trust_vals = [u.trust for u in sim.users]
                day_rec = {
                    "global_day": global_day, "pass": pass_idx, "day_in_pass": day_in_pass,
                    "alpha_gini": ag, "alpha_flock": af,
                    "mean_reward": float(rewards.mean()),
                    "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                    "mean_trust_allusers": float(np.mean(trust_vals)),
                    "gini_served_cum": _gini(served),
                    "total_served_cum": int(served.sum()),
                    "entropy_final": stats.get("entropy_final", 0.0),
                }
                day_records.append(day_rec)
                print(f"  [Pass {pass_idx+1}/{N_PASSES} Hari {day_in_pass+1:02d}/{DAYS_PER_PASS} "
                     f"(#{global_day+1:03d}/{TOTAL_DAYS})] a_gini={ag:.3f} a_flock={af:.3f} "
                     f"accept={day_rec['acceptance_rate']:.2f} trust={day_rec['mean_trust_allusers']:.3f} "
                     f"gini_served={day_rec['gini_served_cum']:.3f} served={day_rec['total_served_cum']}")

            global_day += 1
            if boundary:
                break
            agent.transitions = pending

        recs_this_pass = [d for d in day_records if d["pass"] == pass_idx]
        served_pass = np.array([s.total_served for s in sim.spklus.values()], float)
        pass_rec = {
            "pass": pass_idx,
            "acceptance_mean": float(np.mean([d["acceptance_rate"] for d in recs_this_pass])),
            "trust_end": recs_this_pass[-1]["mean_trust_allusers"] if recs_this_pass else None,
            "gini_served_end": _gini(served_pass),
            "herding_events_cum": sim.herding_events,
        }
        pass_records.append(pass_rec)
        print(f"  --- Pass {pass_idx+1} selesai: accept_mean={pass_rec['acceptance_mean']:.3f} "
             f"trust_end={pass_rec['trust_end']:.3f} gini_end={pass_rec['gini_served_end']:.3f} ---")

    n_report = min(5, len(day_records))
    result = {
        "variant": name, "init_trust": init_trust, "total_days": TOTAL_DAYS,
        "acceptance_first%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[:n_report]])),
        "acceptance_last%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[-n_report:]])),
        "trust_last%d" % n_report: float(np.mean([d["mean_trust_allusers"] for d in day_records[-n_report:]])),
        "gini_served_final": pass_records[-1]["gini_served_end"],
        "herding_events_final_pass": pass_records[-1]["herding_events_cum"],
        "reward_last%d" % n_report: float(np.mean([d["mean_reward"] for d in day_records[-n_report:]])),
        "entropy_last%d" % n_report: float(np.mean([d["entropy_final"] for d in day_records[-n_report:]])),
        "pass_records": pass_records,
        "day_records": day_records,
    }
    return result


def main():
    variants = {
        "baseline": (0.5, lambda gday: (FULL_ALPHA_GINI, FULL_ALPHA_FLOCK)),
        "high_trust_static": (1.0, lambda gday: (FULL_ALPHA_GINI, FULL_ALPHA_FLOCK)),
        "fairness_off_static": (1.0, lambda gday: (0.0, 0.0)),
        # Kurikulum PER-PASS: bobot fairness naik 0->penuh SEPANJANG SATU PASS (day_in_pass),
        # lalu RESET ke 0 di awal tiap pass baru -- bukan naik kumulatif lintas 600 hari
        # spt versi sebelumnya (yg keliru: day_in_pass diganti global_day, seharusnya
        # sebaliknya).
        "curriculum": (1.0, lambda day_in_pass: (FULL_ALPHA_GINI * day_in_pass / (DAYS_PER_PASS - 1),
                                                 FULL_ALPHA_FLOCK * day_in_pass / (DAYS_PER_PASS - 1))),
    }

    ONLY = ["curriculum"]   # 3 varian lain tak berubah dari run sebelumnya (konstan thd day_in_pass)
    all_results = {}
    for name, (init_trust, sched) in variants.items():
        if name not in ONLY:
            continue
        all_results[name] = run_variant(name, init_trust, sched)

    with open(f"test_curriculum_perpass_newdataset_{DAYS_PER_PASS}dx{N_PASSES}pass_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'=' * 100}\n=== TABEL PERBANDINGAN AKHIR ({TOTAL_DAYS} hari / {N_PASSES} pass) ===\n{'=' * 100}")
    cols = ["acceptance_last5", "trust_last5", "gini_served_final", "herding_events_final_pass",
           "reward_last5", "entropy_last5"]
    header = f"{'variant':<22}" + "".join(f"{c:>20}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        row = f"{name:<22}"
        for c in cols:
            v = r.get(c)
            row += f"{v:>20.4f}" if isinstance(v, float) else f"{str(v):>20}"
        print(row)

    print("\n=== ACCEPTANCE & GINI PER-PASS (progresi lintas 5 pass) ===")
    for name, r in all_results.items():
        accs = [p["acceptance_mean"] for p in r["pass_records"]]
        ginis = [p["gini_served_end"] for p in r["pass_records"]]
        print(f"{name:<22} accept/pass={['%.3f'%a for a in accs]}  gini/pass={['%.3f'%g for g in ginis]}")

    print(f"\n[INFO] Ringkasan lengkap -> test_curriculum_perpass_newdataset_{DAYS_PER_PASS}dx{N_PASSES}pass_summary.json")


if __name__ == "__main__":
    main()
