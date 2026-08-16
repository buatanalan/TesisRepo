import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Uji perbaikan celah critic (3 prioritas teratas, lihat diskusi sebelumnya):
  1. Waktu-tersedia slot   -- rata2 remaining_time EV yg SEDANG charging per SPKLU (celah:
     critic cuma tahu rasio utilisasi 0-1, tak tahu KAPAN slot terisi akan kosong).
  2. EV traveling           -- jumlah EV menuju tiap SPKLU yg belum tiba (celah sama dgn
     forecaster yg sudah didiagnosis sebelumnya, kini utk critic).
  3. Konteks temporal          -- sin/cos jam & hari (celah: critic buta waktu sepenuhnya,
     padahal demand berpola 2-puncak jelas di dataset baru).

TIDAK mengedit marl_spklu/rl/rollout.py atau policy.py -- EnhancedCriticAgent men-subclass
RLRolloutAgent (cuma override _build_critic_obs), dan HPPOPolicy dipakai APA ADANYA (generik
thd critic_obs_dim, tinggal diberi angka lebih besar saat konstruksi).

Baseline critic_obs: 2N+1 dim (utilisasi, antrean, gini) -- ASLI, tak diubah.
Enhanced critic_obs : 4N+5 dim (+ remaining_time, + n_traveling, + 4 fitur temporal).

Kedua varian dilatih pada kondisi IDENTIK (dataset baru, trust=1.0 statis, populasi &
kapasitas ASLI) -- satu2nya beda adalah info yg dilihat critic. Metrik utama: explained_variance
(seberapa baik critic menaksir return) & metrik perilaku standar (acceptance/gini/reward).
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
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator, _gini as _gini_calc

DATASET = "scenario_dataset.json"
SEED = 0
CHUNK = 96
N_UPDATES = 30


class EnhancedCriticAgent(RLRolloutAgent):
    """Subclass RLRolloutAgent -- HANYA _build_critic_obs yg berbeda. Semua lain (obs actor,
    reward hooks, dst.) reuse APA ADANYA dari kelas induk."""

    def _build_critic_obs(self):
        utilisasi = np.array([self.sim.spklus[s].get_utilization() for s in self.sids])
        antrean = np.array([self.sim.spklus[s].get_queue_length() for s in self.sids]) / self.q_scale
        gini_utilisasi = _gini_calc(utilisasi)

        # --- Perbaikan #1: waktu-tersedia slot (rata2 remaining_time yg SEDANG charging) ---
        remaining = []
        for s in self.sids:
            spklu = self.sim.spklus[s]
            times = [ev["remaining_time"] for c in spklu.charging.values() for ev in c]
            remaining.append(float(np.mean(times)) if times else 0.0)
        remaining = np.array(remaining) / self.wait_scale

        # --- Perbaikan #2: EV traveling menuju tiap SPKLU (belum tiba) ---
        traveling_counts = np.zeros(self.N)
        for u in self.sim.users:
            if u.state == UserState.TRAVELING and u.target_spklu in self.sid_to_idx:
                traveling_counts[self.sid_to_idx[u.target_spklu]] += 1
        traveling_counts = traveling_counts / self.q_scale

        # --- Perbaikan #3: konteks temporal (jam & hari) ---
        time_now = self.sim.current_step * self.sim.dt_minutes
        hour = (time_now / 60.0) % 24.0
        dow = (time_now / (60.0 * 24.0)) % 7.0
        temporal = np.array([
            math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * dow / 7.0), math.cos(2 * math.pi * dow / 7.0),
        ])

        return np.concatenate([
            utilisasi, antrean, [gini_utilisasi], remaining, traveling_counts, temporal,
        ]).astype(np.float32)


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def build_sim(willingness_ratio=5.0):
    """Kondisi REALISTIS: populasi & kapasitas ASLI dataset baru, cuma trust di-bootstrap 1.0
    (pemenang tunggal terbukti dari sweep sebelumnya) supaya sinyal belajar cukup kuat utk
    perbandingan critic yg bermakna."""
    sim = Simulator({}, [], None, user_willingness_radius_km=None,
                    user_willingness_ratio=willingness_ratio)
    sim.load_from_dataset(DATASET)
    for u in sim.users:
        u.trust = 1.0
    return sim


def run_variant(name, agent_cls, critic_obs_dim_fn):
    print(f"\n{'=' * 70}\n=== VARIAN: {name} ===\n{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    sim0 = build_sim()
    N = len(sim0.spklus)
    obs_dim = 4 + 4 * N
    critic_obs_dim = critic_obs_dim_fn(N)
    print(f"N_SPKLU={N}  obs_dim={obs_dim}  critic_obs_dim={critic_obs_dim}")

    policy = HPPOPolicy(obs_dim, critic_obs_dim, N, delta_max=10.0)
    ppo = PPOTrainer(policy)
    rc = RewardCalculator()
    forecaster = FormulaForecaster()
    sim = build_sim()
    agent = agent_cls(policy, sim, rc, forecaster, k=3, honest_estwait=True)

    day_records = []
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

        if resolved:
            stats = ppo.update(resolved)
            rewards = np.array([t.reward for t in resolved])
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            day_rec = {
                "day": day,
                "mean_reward": float(rewards.mean()),
                "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                "gini_served_cum": _gini(served),
                "total_served_cum": int(served.sum()),
                "explained_var": stats.get("explained_var", 0.0),
                "approx_kl": stats.get("approx_kl", 0.0),
                "entropy_final": stats.get("entropy_final", 0.0),
                "v_loss": stats.get("v_loss", 0.0),
            }
            day_records.append(day_rec)
            print(f"  [Hari {day + 1:02d}/{N_UPDATES}] R={day_rec['mean_reward']:+.4f} "
                 f"accept={day_rec['acceptance_rate']:.2f} EV={day_rec['explained_var']:+.3f} "
                 f"v_loss={day_rec['v_loss']:.3f} KL={day_rec['approx_kl']:.4f} "
                 f"gini_served={day_rec['gini_served_cum']:.3f} served={day_rec['total_served_cum']}")

        if boundary:
            break
        agent.transitions = pending

    n_report = min(5, len(day_records))
    served_final = np.array([s.total_served for s in sim.spklus.values()], float)
    result = {
        "variant": name, "critic_obs_dim": critic_obs_dim,
        "acceptance_last%d" % n_report: float(np.mean([d["acceptance_rate"] for d in day_records[-n_report:]])),
        "gini_served_final": _gini(served_final),
        "reward_last%d" % n_report: float(np.mean([d["mean_reward"] for d in day_records[-n_report:]])),
        "explained_var_mean": float(np.mean([d["explained_var"] for d in day_records])),
        "explained_var_last%d" % n_report: float(np.mean([d["explained_var"] for d in day_records[-n_report:]])),
        "v_loss_mean": float(np.mean([d["v_loss"] for d in day_records])),
        "day_records": day_records,
    }
    return result


def main():
    results = {
        "baseline_critic": run_variant("baseline_critic", RLRolloutAgent, lambda N: 2 * N + 1),
        "enhanced_critic": run_variant("enhanced_critic", EnhancedCriticAgent, lambda N: 4 * N + 5),
    }
    with open("test_enhanced_critic_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 100}\n=== PERBANDINGAN: critic lama vs critic diperkaya ===\n{'=' * 100}")
    cols = ["explained_var_mean", "explained_var_last5", "v_loss_mean",
           "acceptance_last5", "gini_served_final", "reward_last5"]
    header = f"{'varian':<20}" + "".join(f"{c:>20}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        row = f"{name:<20}"
        for c in cols:
            v = r.get(c)
            row += f"{v:>20.4f}" if isinstance(v, float) else f"{str(v):>20}"
        print(row)

    print("\n[INFO] Ringkasan lengkap -> test_enhanced_critic_summary.json")


if __name__ == "__main__":
    main()
