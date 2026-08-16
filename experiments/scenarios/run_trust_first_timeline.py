import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Eksperimen TRUST-FIRST TIMELINE (menyerang batasan §9 LAPORAN_KONFIGURASI_REWARD.md:
acceptance/trust mentok ~0.60 dan belum menyaingi greedy S1).

Hipotesis user: pada konfigurasi rekomendasi, trust di-RESET tiap pass (30 hari) sehingga tak
pernah melewati asimtot ~0.60 yang tercapai dalam satu pass. Kalau trust dibiarkan
TERAKUMULASI lintas-pass (timeline kontinu) dan objektif Gini DITUNDA sampai trust cukup tinggi
(bangun kepercayaan dulu, ratakan belakangan), acceptance & trust mungkin menembus plateau.

Tiga varian, konfigurasi rekomendasi IDENTIK (ent_coef=0.3, dataset 5x, Formula, rolling-Gini,
critic baseline, actor lokal, bobot individual patokan). HANYA berbeda pada penanganan trust:

  1. reset_baseline : trust RESET ke 0.5 tiap pass; lambda=0.25 tetap.  (= konfigurasi rekomendasi, acuan)
  2. continuous     : trust DIBAWA lintas-pass (mulai 0.5 di pass-0); lambda=0.25 tetap.  (isolasi efek kontinuitas)
  3. trust_first    : trust DIBAWA lintas-pass; lambda=0 (Gini MATI) sampai mean-trust >= THRESHOLD,
                      lalu lambda=0.25 (Gini HIDUP).  (usulan penuh: bangun trust dulu, ratakan kemudian)

Trust dibawa lewat dict {user_id: trust} yang direstor setelah build_sim (kejadian kedatangan
di-reset dari dataset tiap pass, tapi trust user dipertahankan). NOL edit ke marl_spklu/.
"""
import json
import random

import numpy as np
import torch

from marl_spklu.env.simulator import Simulator
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.ppo import PPOTrainer
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import RewardCalculator

from run_rolling_gini_reward import RollingBaseCriticAgent

DATASET = "scenario_dataset_5x.json"
SEED = 0
CHUNK = 96
DAYS_PER_PASS = 30
N_PASSES = 10
ENT_COEF = 0.3
LAMBDA = 0.25
BASE_ALPHA_GINI = 0.5
BASE_ALPHA_FLOCK = 0.3
INIT_TRUST = 0.5
TRUST_THRESHOLD = 0.65   # trust_first: Gini menyala saat mean-trust menembus nilai ini


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def build_sim(trust_carry=None):
    sim = Simulator({}, [], None, user_willingness_radius_km=None, user_willingness_ratio=5.0)
    sim.load_from_dataset(DATASET)
    for u in sim.users:
        if trust_carry is not None and u.user_id in trust_carry:
            u.trust = trust_carry[u.user_id]
        else:
            u.trust = INIT_TRUST
    return sim


def run_variant(name, carry_trust, staged):
    """carry_trust: bawa trust lintas-pass?  staged: mulai lambda=0 lalu naik saat trust>=threshold?"""
    print(f"\n{'=' * 70}\n=== VARIAN: {name}  (carry_trust={carry_trust} staged={staged}) ===\n{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    N = len(build_sim().spklus)
    policy = HPPOPolicy(4 + 4 * N, 2 * N + 1, N, delta_max=10.0)
    ppo = PPOTrainer(policy, ent_coef=ENT_COEF)
    forecaster = FormulaForecaster()

    # rc dipakai ulang; alpha_gini/alpha_flock diubah per-pass (mutable) utk staging.
    rc = RewardCalculator(alpha_wait=1.0, beta_prox=0.1, alpha_honesty=1.0,
                          alpha_gini=BASE_ALPHA_GINI * LAMBDA, alpha_flock=BASE_ALPHA_FLOCK * LAMBDA)

    trust_carry = None
    gini_active = not staged     # non-staged: Gini aktif sejak awal
    switch_pass = None
    pass_records = []
    for pass_idx in range(N_PASSES):
        # Tentukan lambda efektif pass ini (staging berdasar trust dari akhir pass sebelumnya).
        if staged and not gini_active and trust_carry is not None:
            if float(np.mean(list(trust_carry.values()))) >= TRUST_THRESHOLD:
                gini_active = True
                switch_pass = pass_idx
        lam = LAMBDA if gini_active else 0.0
        rc.alpha_gini = BASE_ALPHA_GINI * lam
        rc.alpha_flock = BASE_ALPHA_FLOCK * lam

        sim = build_sim(trust_carry if carry_trust else None)
        agent = RollingBaseCriticAgent(policy, sim, rc, forecaster, k=3, honest_estwait=True)
        step = 0
        accepts = []
        for _ in range(DAYS_PER_PASS):
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
                ppo.update(resolved)
                accepts.append(float(np.mean([t.complied for t in resolved])))
            if boundary:
                break
            agent.transitions = pending

        # Simpan trust utk pass berikutnya.
        trust_carry = {u.user_id: float(u.trust) for u in sim.users}
        served = np.array([s.total_served for s in sim.spklus.values()], float)
        waits = [L["wait_time"] for L in sim.logs]
        pr = {
            "pass": pass_idx,
            "lambda": lam,
            "acceptance": float(np.mean(accepts)),
            "trust_end": float(np.mean([u.trust for u in sim.users])),
            "gini_served": _gini(served),
            "mean_wait": float(np.mean(waits)) if waits else 0.0,
            "herding": sim.herding_events,
        }
        pass_records.append(pr)
        flag = " <-- Gini ON" if (switch_pass is not None and pass_idx == switch_pass) else ""
        print(f"  [Pass {pass_idx+1:02d}] lam={lam:.2f} accept={pr['acceptance']:.3f} "
             f"trust={pr['trust_end']:.3f} gini={pr['gini_served']:.3f} wait={pr['mean_wait']:.1f}{flag}")

    last5 = pass_records[-5:]
    return {
        "name": name, "carry_trust": carry_trust, "staged": staged, "switch_pass": switch_pass,
        "acceptance": float(np.mean([p["acceptance"] for p in last5])),
        "trust": float(np.mean([p["trust_end"] for p in last5])),
        "gini": float(np.mean([p["gini_served"] for p in last5])),
        "wait": float(np.mean([p["mean_wait"] for p in last5])),
        "herding": float(np.mean([p["herding"] for p in last5])),
        "trust_per_pass": [p["trust_end"] for p in pass_records],
        "acceptance_per_pass": [p["acceptance"] for p in pass_records],
        "gini_per_pass": [p["gini_served"] for p in pass_records],
    }


def main():
    results = [
        run_variant("reset_baseline", carry_trust=False, staged=False),
        run_variant("continuous",     carry_trust=True,  staged=False),
        run_variant("trust_first",    carry_trust=True,  staged=True),
    ]
    with open("test_trust_first_timeline_summary.json", "w") as f:
        json.dump({"config": {"dataset": DATASET, "ent_coef": ENT_COEF, "lambda": LAMBDA,
                              "threshold": TRUST_THRESHOLD, "n_passes": N_PASSES,
                              "days_per_pass": DAYS_PER_PASS, "seed": SEED},
                  "variants": results}, f, indent=2)

    print(f"\n\n{'=' * 92}\n=== TRUST-FIRST TIMELINE (300 hari, konfig rekomendasi, HANYA beda penanganan trust) ===\n{'=' * 92}")
    print(f"{'varian':<16}{'accept':>9}{'trust':>8}{'gini':>8}{'wait':>8}{'herding':>9}{'switch':>8}")
    print("-" * 66)
    for r in results:
        sw = "-" if r["switch_pass"] is None else f"p{r['switch_pass']+1}"
        print(f"{r['name']:<16}{r['acceptance']:>9.3f}{r['trust']:>8.3f}{r['gini']:>8.3f}"
             f"{r['wait']:>8.1f}{r['herding']:>9.0f}{sw:>8}")
    print("\n--- trust/pass (apakah menembus plateau ~0.60?) ---")
    for r in results:
        print(f"  {r['name']:<16}: {['%.3f'%t for t in r['trust_per_pass']]}")
    print("--- acceptance/pass ---")
    for r in results:
        print(f"  {r['name']:<16}: {['%.3f'%a for a in r['acceptance_per_pass']]}")
    print("\n[INFO] Ringkasan -> test_trust_first_timeline_summary.json")


if __name__ == "__main__":
    main()
