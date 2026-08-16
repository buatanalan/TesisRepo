import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Uji integrasi 3 stack (isolasi kontribusi forecaster vs critic) di dataset BARU:

  A_formula_basecritic  : FormulaForecaster + critic baseline (2N+1)  -- stack LAMA, referensi
  B_learned_basecritic   : LearnedForecaster(Mode B) + critic baseline -- efek FORECASTER saja
  C_learned_enhcritic      : LearnedForecaster(Mode B) + critic DIPERKAYA (4N+5) -- kedua perbaikan

Kebijakan akses informasi (disepakati user):
  - FORECASTER: sadar-TEMPORAL (extract_features sudah punya sin/cos jam & hari), BUTA traveling-EV
    (info niat perjalanan pribadi tak boleh diintip layanan penduga-wait publik). LearnedForecaster
    memenuhi ini apa adanya -- tak ada fitur traveling-EV di dalamnya.
  - CRITIC (hanya training, CTDE): boleh lihat traveling-EV + waktu-tersedia-slot + temporal (celah
    yg sudah diidentifikasi). EnhancedCriticAgent._build_critic_obs.
  - ACTOR: TETAP LOKAL, tak disentuh (_build_obs warisan RLRolloutAgent apa adanya).

Forecaster (LearnedForecaster) DIBEKUKAN setelah pretraining offline (Mode B) -- konsisten
gradien terpisah RL vs supervised. TIDAK mengedit marl_spklu/.
"""
import json
import math

import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.user import UserState
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.ppo import PPOTrainer
from marl_spklu.rl.rollout import RLRolloutAgent
from marl_spklu.rl.forecaster import FormulaForecaster, LearnedForecaster, collect_forecast_dataset
from marl_spklu.rl.rewards import RewardCalculator, _gini as _gini_calc
from marl_spklu.agents.greedy_agent import GreedyAgent

DATASET = "scenario_dataset.json"
SEED = 0
CHUNK = 96
N_UPDATES = 30
INIT_TRUST = 1.0


class EnhancedCriticAgent(RLRolloutAgent):
    """Critic diperkaya: + waktu-tersedia-slot, + EV-traveling, + temporal. Actor TAK diubah."""

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


def pretrain_learned_forecaster():
    """Mode B: kumpulkan (X,y) dari trajektori GreedyAgent, fit, bekukan. X TIDAK punya
    traveling-EV (extract_features buta itu -- sesuai kebijakan); y = compute_virtual_wait
    (label boleh mencerminkan realita penuh, termasuk traveling-EV -> forecaster memang akan
    sistematis under-predict, diterima sbg realistis)."""
    sim = build_sim()
    X, y = collect_forecast_dataset(sim, min(2880, sim.max_steps), agent=GreedyAgent())
    fc = LearnedForecaster().fit(X, y)
    with torch.no_grad():
        pred = fc.model(torch.tensor(np.asarray(X, np.float32))).numpy()
    err = pred - np.asarray(y)
    print(f"  [pretrain LearnedForecaster] n={len(y)} in-sample MAE={np.mean(np.abs(err)):.2f} "
         f"bias={np.mean(err):.2f}")
    return fc


def run_variant(name, forecaster_factory, agent_cls, critic_obs_dim_fn):
    print(f"\n{'=' * 70}\n=== VARIAN: {name} ===\n{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    sim0 = build_sim()
    N = len(sim0.spklus)
    obs_dim = 4 + 4 * N
    critic_obs_dim = critic_obs_dim_fn(N)
    policy = HPPOPolicy(obs_dim, critic_obs_dim, N, delta_max=10.0)
    ppo = PPOTrainer(policy)
    rc = RewardCalculator()
    forecaster = forecaster_factory()
    sim = build_sim()
    agent = agent_cls(policy, sim, rc, forecaster, k=3, honest_estwait=True)
    print(f"N={N} obs_dim={obs_dim} critic_obs_dim={critic_obs_dim}")

    day_records = []
    session_log = []
    prev_len = 0
    step = 0
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

        new_sessions = sim.logs[prev_len:]
        prev_len = len(sim.logs)
        for s in new_sessions:
            session_log.append(s)

        if resolved:
            stats = ppo.update(resolved)
            rewards = np.array([t.reward for t in resolved])
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            trust_vals = [u.trust for u in sim.users]
            day_rec = {
                "day": day, "mean_reward": float(rewards.mean()),
                "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                "mean_trust_allusers": float(np.mean(trust_vals)),
                "gini_served_cum": _gini(served), "total_served_cum": int(served.sum()),
                "explained_var": stats.get("explained_var", 0.0),
                "v_loss": stats.get("v_loss", 0.0),
            }
            day_records.append(day_rec)
            print(f"  [Hari {day + 1:02d}/{N_UPDATES}] R={day_rec['mean_reward']:+.4f} "
                 f"accept={day_rec['acceptance_rate']:.2f} trust={day_rec['mean_trust_allusers']:.3f} "
                 f"EV={day_rec['explained_var']:+.3f} gini={day_rec['gini_served_cum']:.3f} "
                 f"served={day_rec['total_served_cum']}")

        if boundary:
            break
        agent.transitions = pending

    n_report = min(5, len(day_records))
    served_final = np.array([s.total_served for s in sim.spklus.values()], float)
    pred = np.array([s["est_wait"] for s in session_log])
    act = np.array([s["wait_time"] for s in session_log])
    err = pred - act
    return {
        "variant": name, "critic_obs_dim": critic_obs_dim,
        "acceptance_last%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[-n_report:]])),
        "trust_last%d" % n_report: float(np.mean([d["mean_trust_allusers"] for d in day_records[-n_report:]])),
        "gini_served_final": _gini(served_final),
        "reward_last%d" % n_report: float(np.mean([d["mean_reward"] for d in day_records[-n_report:]])),
        "explained_var_mean": float(np.mean([d["explained_var"] for d in day_records])),
        "explained_var_last%d" % n_report: float(np.mean([d["explained_var"] for d in day_records[-n_report:]])),
        "forecaster_realized_mae": float(np.mean(np.abs(err))) if len(err) else None,
        "forecaster_realized_bias": float(np.mean(err)) if len(err) else None,
        "mean_actual_wait": float(act.mean()) if len(act) else None,
        "day_records": day_records,
    }


def main():
    print("=== Pretrain LearnedForecaster (Mode B, sekali; dipakai ulang utk B & C) ===")
    shared_learned = pretrain_learned_forecaster()

    results = {}
    results["A_formula_basecritic"] = run_variant(
        "A_formula_basecritic", lambda: FormulaForecaster(), RLRolloutAgent, lambda N: 2 * N + 1)
    results["B_learned_basecritic"] = run_variant(
        "B_learned_basecritic", lambda: shared_learned, RLRolloutAgent, lambda N: 2 * N + 1)
    results["C_learned_enhcritic"] = run_variant(
        "C_learned_enhcritic", lambda: shared_learned, EnhancedCriticAgent, lambda N: 4 * N + 5)

    with open("test_integrated_stack_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 110}\n=== PERBANDINGAN 3 STACK (dataset baru, trust=1.0, 30 hari) ===\n{'=' * 110}")
    cols = ["forecaster_realized_mae", "forecaster_realized_bias", "explained_var_mean",
           "explained_var_last5", "acceptance_last5", "gini_served_final", "reward_last5"]
    header = f"{'stack':<24}" + "".join(f"{c:>22}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        row = f"{name:<24}"
        for c in cols:
            v = r.get(c)
            row += f"{v:>22.4f}" if isinstance(v, float) else f"{str(v):>22}"
        print(row)
    print("\n[INFO] Ringkasan lengkap -> test_integrated_stack_summary.json")


if __name__ == "__main__":
    main()
