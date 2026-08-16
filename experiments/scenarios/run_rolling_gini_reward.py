import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Perbaikan REWARD: ganti gini_instant (Gini atas utilisasi SESAAT per-step, terbukti
secara struktural selalu tinggi saat sistem sepi -- §7.2 LAPORAN) dengan Gini atas RATA-RATA
utilisasi JENDELA-BERGULIR 24 jam (96 step). Reward jadi mengukur "seberapa merata stasiun
dipakai selama sehari terakhir", bukan snapshot sesaat yang menyesatkan.

Suku flocking TIDAK diubah (memang inheren sesaat: banyak rekomendasi serempak dalam 1 window).
Actor TETAP lokal, forecaster = FormulaForecaster (dipertahankan). TIDAK edit marl_spklu/.

2 varian x (10 pass x 30 hari), dibandingkan dgn hasil instant-Gini yg SUDAH ada
(test_enhanced_critic_longhorizon_summary.json):
  - instant-Gini baseline critic  : gini/pass 0.122->0.191 (referensi lama)
  - instant-Gini enhanced critic   : gini/pass 0.123->0.253 (referensi lama, MEMBURUK)
  - rolling_basecritic  [BARU]      : critic baseline + reward rolling-Gini
  - rolling_enhcritic    [BARU]      : critic diperkaya + reward rolling-Gini
"""
import json
import math
from collections import deque

import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.user import UserState
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.ppo import PPOTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator, _gini as _gini_calc

DATASET = "scenario_dataset.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
INIT_TRUST = 1.0
ROLLING_WINDOW = 96   # 24 jam


class RollingGiniMixin:
    """Override on_step_end: pakai Gini atas RATA-RATA utilisasi jendela-bergulir (bukan sesaat).
    Suku flocking tetap sesaat (inheren). Diletakkan sbg mixin agar bisa dikombinasi dgn
    critic baseline maupun diperkaya."""

    def _ensure_window(self):
        if not hasattr(self, "_util_window"):
            self._util_window = deque(maxlen=ROLLING_WINDOW)

    def on_step_end(self, sim, step):
        self._ensure_window()
        cur = np.array([s.get_utilization() for s in self.sim.spklus.values()])
        self._util_window.append(cur)
        utils_rolling = np.mean(np.stack(self._util_window), axis=0)   # per-SPKLU mean atas window

        transitions_this_step = [tr for tr in self.transitions if tr.step == step]
        n_window = len(transitions_this_step)
        rec_counts = {}
        for tr in transitions_this_step:
            rec_counts[tr.chosen_idx] = rec_counts.get(tr.chosen_idx, 0) + 1
        for tr in transitions_this_step:
            if not tr.resolved and tr.flock_penalty == 0.0:
                n_same = rec_counts.get(tr.chosen_idx, 0)
                penalty = self.rc.flocking_penalty(n_same, n_window)
                tr.flock_penalty = penalty
                # global_reward memakai utils_rolling (BUKAN sesaat) utk suku Gini.
                tr.reward += self.rc.global_reward(utils_rolling, n_same=n_same, n_window=n_window)


class EnhancedCriticMixin:
    def _build_critic_obs(self):
        utilisasi = np.array([self.sim.spklus[s].get_utilization() for s in self.sids])
        antrean = np.array([self.sim.spklus[s].get_queue_length() for s in self.sids]) / self.q_scale
        gini_u = _gini_calc(utilisasi)
        remaining = []
        for s in self.sids:
            times = [ev["remaining_time"] for c in self.sim.spklus[s].charging.values() for ev in c]
            remaining.append(float(np.mean(times)) if times else 0.0)
        remaining = np.array(remaining) / self.wait_scale
        traveling = np.zeros(self.N)
        for u in self.sim.users:
            if u.state == UserState.TRAVELING and u.target_spklu in self.sid_to_idx:
                traveling[self.sid_to_idx[u.target_spklu]] += 1
        traveling = traveling / self.q_scale
        time_now = self.sim.current_step * self.sim.dt_minutes
        hour = (time_now / 60.0) % 24.0
        dow = (time_now / (60.0 * 24.0)) % 7.0
        temporal = np.array([math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
                            math.sin(2 * math.pi * dow / 7.0), math.cos(2 * math.pi * dow / 7.0)])
        return np.concatenate([utilisasi, antrean, [gini_u], remaining, traveling, temporal]).astype(np.float32)


class RollingBaseCriticAgent(RollingGiniMixin, RLRolloutAgent):
    pass


class RollingEnhCriticAgent(RollingGiniMixin, EnhancedCriticMixin, RLRolloutAgent):
    pass


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def build_sim():
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)
    for u in sim.users:
        u.trust = INIT_TRUST
    return sim


def run_variant(name, agent_cls, critic_obs_dim_fn):
    print(f"\n{'=' * 70}\n=== VARIAN: {name} ({N_PASSES} pass x {DAYS_PER_PASS} hari) ===\n{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    N = len(build_sim().spklus)
    policy = HPPOPolicy(4 + 4 * N, critic_obs_dim_fn(N), N, delta_max=10.0)
    ppo = PPOTrainer(policy)
    rc = RewardCalculator()
    forecaster = FormulaForecaster()

    day_records = []
    pass_records = []
    for pass_idx in range(N_PASSES):
        sim = build_sim()
        agent = agent_cls(policy, sim, rc, forecaster, k=3, honest_estwait=True)
        step = 0
        for day_in_pass in range(DAYS_PER_PASS):
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
                stats = ppo.update(resolved)
                rewards = np.array([t.reward for t in resolved])
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                day_records.append({
                    "pass": pass_idx, "day_in_pass": day_in_pass,
                    "mean_reward": float(rewards.mean()),
                    "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                    "gini_served_cum": _gini(served),
                    "explained_var": stats.get("explained_var", 0.0),
                })
            if boundary:
                break
            agent.transitions = pending

        recs_this = [d for d in day_records if d["pass"] == pass_idx]
        served_pass = np.array([s.total_served for s in sim.spklus.values()], float)
        pr = {
            "pass": pass_idx,
            "acceptance_mean": float(np.mean([d["acceptance_rate"] for d in recs_this])),
            "gini_end": _gini(served_pass),
            "explained_var_mean": float(np.mean([d["explained_var"] for d in recs_this])),
            "herding_events": sim.herding_events,
        }
        pass_records.append(pr)
        print(f"  [Pass {pass_idx+1:02d}/{N_PASSES}] accept={pr['acceptance_mean']:.3f} "
             f"gini_end={pr['gini_end']:.3f} EV={pr['explained_var_mean']:+.3f} herding={pr['herding_events']}")

    return {
        "variant": name,
        "acceptance_per_pass": [p["acceptance_mean"] for p in pass_records],
        "gini_per_pass": [p["gini_end"] for p in pass_records],
        "ev_per_pass": [p["explained_var_mean"] for p in pass_records],
        "herding_pass10": pass_records[-1]["herding_events"],
        "pass_records": pass_records,
    }


def main():
    results = {
        "rolling_basecritic": run_variant("rolling_basecritic", RollingBaseCriticAgent, lambda N: 2 * N + 1),
        "rolling_enhcritic": run_variant("rolling_enhcritic", RollingEnhCriticAgent, lambda N: 4 * N + 5),
    }
    with open("test_rolling_gini_reward_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 100}\n=== REWARD rolling-Gini @ 300 HARI (10 pass) ===\n{'=' * 100}")
    print("(Referensi instant-Gini dari run sebelumnya: baseline gini 0.122->0.191, "
         "enhanced gini 0.123->0.253)")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  acceptance/pass: {['%.3f'%a for a in r['acceptance_per_pass']]}")
        print(f"  gini/pass      : {['%.3f'%g for g in r['gini_per_pass']]}")
        print(f"  expl_var/pass  : {['%.2f'%e for e in r['ev_per_pass']]}")
        print(f"  gini pass1->pass10: {r['gini_per_pass'][0]:.3f}->{r['gini_per_pass'][-1]:.3f}  "
             f"herding_pass10={r['herding_pass10']}")
    print("\n[INFO] Ringkasan lengkap -> test_rolling_gini_reward_summary.json")


if __name__ == "__main__":
    main()
