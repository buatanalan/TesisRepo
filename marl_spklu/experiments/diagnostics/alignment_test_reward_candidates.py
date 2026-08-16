"""Uji keselarasan 5 kandidat reward pemerataan (usulan "perbandingan lebih setara dgn
greedy_queue") -- metodologi SAMA dengan yang dipakai memvalidasi EquityRewardCalculator:
jalankan beberapa kebijakan aturan-tetap (rentang Gini rendah..tinggi), rekam kuantitas
per-keputusan yang sama dipakai reward PDQN (utilisasi saat keputusan, indeks kandidat
feasible, a_hat, default, dipilih), hitung tiap kandidat reward per-keputusan, lalu
korelasikan rata-rata reward per skenario terhadap gini_mean skenario itu. Reward yang
BAIK harus punya korelasi NEGATIF kuat (reward tinggi -> Gini rendah).

Kandidat (lihat diskusi laporan / percakapan):
  1. marginal_gini : gini(utils bila DEFAULT yg nambah) - gini(utils bila A_HAT yg nambah)
  2. var_reduction  : var(...) versi (1), pakai variance bukan Gini (lebih murah/mulus)
  3. rank_percentile: seberapa rendah utilisasi a_hat dibanding kandidat feasible lain
  4. target_dev     : -(utils[a_hat] - mean(utils[feasible]))**2
  5. current_prod   : r_rec + r_shift yang SUDAH dipakai EquityRewardCalculator (pembanding)

Pemakaian:
    python -m marl_spklu.experiments.diagnostics.alignment_test_reward_candidates
"""
import contextlib
import random

import numpy as np

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.agents.wait_predictor import DeterministicWaitPredictor
from marl_spklu.env.user import User
from marl_spklu.experiments import harness

DATASET = "scenario_dataset_7d.json"
BUMP = 0.05  # kenaikan utilisasi ternormalisasi hipotetis ~ 1 EV di stasiun kapasitas~20


class RandomAgent:
    def __init__(self):
        self.wait_predictor = DeterministicWaitPredictor()

    def predict_waits(self, spklus):
        return self.wait_predictor.predict(spklus)

    def get_recommendation(self, spklus):
        return [random.choice(sorted(spklus.keys()))]


class AntiGreedyAgent:
    """Kebalikan Greedy-util: sengaja rekomendasikan SPKLU PALING ramai -> Gini terburuk,
    dipakai sbg jangkar ujung-atas rentang Gini utk uji keselarasan."""

    def __init__(self):
        self.wait_predictor = DeterministicWaitPredictor()

    def predict_waits(self, spklus):
        return self.wait_predictor.predict(spklus)

    def _score(self, spklu):
        total_evs = sum(len(q) for q in spklu.queues.values())
        total_evs += sum(len(c) for c in spklu.charging.values())
        total_cap = sum(spklu.capacities.values())
        return total_evs / total_cap if total_cap > 0 else 0.0

    def get_recommendation(self, spklus):
        scores = {sid: self._score(s) for sid, s in spklus.items()}
        max_score = max(scores.values())
        worst = sorted(sid for sid, s in scores.items() if s == max_score)
        return [worst[0]]


@contextlib.contextmanager
def record_decisions(sim, records: list):
    """Patch User.decide_spklu utk merekam snapshot per-keputusan (sama semantik dgn
    PDQNRolloutAgent.get_recommendation/on_decision), berlaku utk AGEN APA PUN karena
    hanya membaca `recommendations` (list ID) yg sudah dihasilkan agen sebelum dipanggil."""
    orig = User.decide_spklu
    sids = list(sim.spklus.keys())
    sid_to_idx = {s: i for i, s in enumerate(sids)}

    def patched(self, recommendations, estimated_waits, spklu_features, speed_kmh=40.0,
               gamma=0.1, soc_percent=50.0, willingness_radius_km=None, willingness_ratio=None,
               queue_lengths=None, balk_ratio=None, f_rec=None):
        utils = np.array([sim.spklus[s].get_utilization() for s in sids])
        a_hat_id = recommendations[0] if recommendations else None
        a_hat_idx = sid_to_idx.get(a_hat_id)
        kwargs = dict(speed_kmh=speed_kmh, gamma=gamma, soc_percent=soc_percent,
                     willingness_radius_km=willingness_radius_km,
                     willingness_ratio=willingness_ratio, queue_lengths=queue_lengths)
        if balk_ratio is not None:
            kwargs["balk_ratio"] = balk_ratio
        if f_rec is not None:
            kwargs["f_rec"] = f_rec
        chosen = orig(self, recommendations, estimated_waits, spklu_features, **kwargs)

        feasible_ids = list(self.last_candidate_ids) if self.last_candidate_ids else sids
        feasible_idx = [sid_to_idx[s] for s in feasible_ids if s in sid_to_idx]
        if not feasible_idx:
            feasible_idx = list(range(len(sids)))
        if self.last_u_pref is not None and len(self.last_u_pref) == len(feasible_ids):
            default_local = int(np.argmax(self.last_u_pref))
            default_idx = sid_to_idx.get(feasible_ids[default_local], feasible_idx[0])
        else:
            default_idx = feasible_idx[0]
        if a_hat_idx is None:
            a_hat_idx = default_idx  # Natural (tanpa rekomendasi): a_hat = default -> r_rec=0
        chosen_idx = sid_to_idx.get(chosen, default_idx)

        records.append(dict(utils=utils, feasible_idx=feasible_idx, a_hat_idx=a_hat_idx,
                            default_idx=default_idx, chosen_idx=chosen_idx))
        return chosen

    User.decide_spklu = patched
    try:
        yield
    finally:
        User.decide_spklu = orig


def candidate_rewards(rec: dict) -> dict:
    u = rec["utils"]
    feas = rec["feasible_idx"]
    a_hat, default, chosen = rec["a_hat_idx"], rec["default_idx"], rec["chosen_idx"]

    u_ahat_bump = u.copy(); u_ahat_bump[a_hat] += BUMP
    u_def_bump = u.copy(); u_def_bump[default] += BUMP

    r_marginal_gini = harness.metrics._gini(u_def_bump) - harness.metrics._gini(u_ahat_bump) \
        if hasattr(harness.metrics, "_gini") else _gini(u_def_bump) - _gini(u_ahat_bump)
    r_var_reduction = float(np.var(u_def_bump) - np.var(u_ahat_bump))

    u_feas = u[feas]
    n_feas = len(feas)
    if n_feas > 1:
        rank = float(np.sum(u_feas > u[a_hat])) / (n_feas - 1)
    else:
        rank = 1.0
    r_rank = rank

    mean_feas = float(u_feas.mean()) if n_feas else float(u[a_hat])
    r_target_dev = -((float(u[a_hat]) - mean_feas) ** 2)

    r_rec = mean_feas - float(u[a_hat])
    r_shift = float(u[default]) - float(u[chosen])
    r_current_prod = r_rec + r_shift

    return dict(marginal_gini=r_marginal_gini, var_reduction=r_var_reduction,
               rank_percentile=r_rank, target_dev=r_target_dev, current_prod=r_current_prod)


def _gini(a):
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a)
    n = a.shape[0]
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


POLICIES = {
    "greedy_util":  lambda: GreedyAgent(mode="utilization"),
    "greedy_queue": lambda: GreedyAgent(mode="queue"),
    "greedy_wait":  lambda: GreedyAgent(mode="wait"),
    "natural":      lambda: None,
    "random":       lambda: RandomAgent(),
    "anti_greedy":  lambda: AntiGreedyAgent(),
}


def main():
    import random as _random
    from marl_spklu.env.simulator import Simulator
    from marl_spklu.env.history_buffer import HistoryBuffer
    from marl_spklu.experiments import metrics as _metrics
    harness.metrics = _metrics  # pastikan referensi tersedia utk _gini fallback di atas

    rows = []
    for name, factory in POLICIES.items():
        seed = 42
        _random.seed(seed)
        np.random.seed(seed)
        sim = Simulator({}, [], None, user_willingness_radius_km=None,
                        user_willingness_ratio=harness.DEFAULT_WILLINGNESS_RATIO)
        sim.load_from_dataset(DATASET)
        max_steps = getattr(sim, "max_steps", max(sim.spawn_schedule) + 1)
        sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=max_steps)
        agent = factory()

        records = []
        with record_decisions(sim, records):
            sim.run(max_steps=max_steps, agent=agent)

        gs = _metrics.gini_series_from_history(sim.history)
        gini_mean = float(gs.mean()) if gs.size else 0.0

        per_decision = [candidate_rewards(r) for r in records]
        agg = {k: float(np.mean([d[k] for d in per_decision])) for k in per_decision[0]} \
            if per_decision else {}
        rows.append((name, gini_mean, agg, len(records)))
        print(f"{name:<14} gini_mean={gini_mean:.4f}  n_dec={len(records)}  "
             + "  ".join(f"{k}={v:+.4f}" for k, v in agg.items()))

    print("\n=== Korelasi Pearson(reward_mean_per_skenario, gini_mean) -- makin negatif makin baik ===")
    ginis = np.array([r[1] for r in rows])
    keys = rows[0][2].keys()
    for k in keys:
        vals = np.array([r[2][k] for r in rows])
        corr = float(np.corrcoef(vals, ginis)[0, 1])
        print(f"  {k:<16} corr = {corr:+.4f}")


if __name__ == "__main__":
    main()
