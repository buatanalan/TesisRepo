"""Harness eksperimen: menjalankan satu skenario (agen tertentu) di atas dataset,
mengekstrak KPI, dan mengagregasi lintas beberapa seed.

Bukan lapisan RL -- ini kerangka pengukuran ringan untuk baseline non-RL (S0/S1/S2)
dan nantinya dipakai ulang oleh skenario RL. Satu 'skenario' = satu agent_factory.

Contoh:
    from marl_spklu.experiments.harness import run_scenario, run_multi_seed, SCENARIOS
    res = run_multi_seed("scenario_dataset.json", SCENARIOS["S1_greedy"], seeds=range(5))
"""
import os
import math
import random
import contextlib

import numpy as np

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.env.user import User, UserState
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.agents.diversified_agent import DiversifiedAgent
from marl_spklu.agents.opsrl_agent import OPSRLAgent
from marl_spklu.experiments import metrics

# Lokasi dataset default (root repo). Perbaikan T1 (Validasi_Generik/LAPORAN_VALIDASI.md):
# "scenario_dataset.json" tak pernah ada di repo (terverifikasi) -- diarahkan ke dataset
# kanonik Klaster 12 yang benar-benar dipakai seluruh jalur kalibrasi (Tier 1-9).
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.normpath(
    os.path.join(_HERE, "..", "..", "scenario_dataset_klaster12.json"))


@contextlib.contextmanager
def force_full_compliance():
    """Patch User.decide_spklu agar SELALU memilih rekomendasi pertama agen (kepatuhan=1),
    mereplikasi asumsi kepatuhan-penuh-eksogen literatur OP-SRL (Baseline 3, §II.6.3/Bab IV
    tesis) -- berbeda dari kepercayaan endogen probabilistik yang dipelajari agen MARL."""
    orig = User.decide_spklu

    def patched(self, recommendations, estimated_waits, spklu_features, speed_kmh=40.0,
               gamma=0.1, soc_percent=50.0, willingness_radius_km=None, willingness_ratio=None,
               queue_lengths=None):
        if not recommendations:
            return orig(self, recommendations, estimated_waits, spklu_features,
                       speed_kmh=speed_kmh, gamma=gamma, soc_percent=soc_percent,
                       willingness_radius_km=willingness_radius_km,
                       willingness_ratio=willingness_ratio, queue_lengths=queue_lengths)
        # Replikasi side-effect wajib decide_spklu (state, target, waktu tempuh) yang
        # normalnya mengikuti hasil sampling probabilistik -- di sini dipaksa = rekomendasi.
        chosen_spklu = recommendations[0]
        self.compliance_history.append(1)
        if len(self.compliance_history) > 30:
            self.compliance_history.pop(0)
        self.last_rec_complied = True
        self.est_wait_presented = float(estimated_waits.get(chosen_spklu, 0.0))
        self.target_spklu = chosen_spklu
        self.prev_spklu = chosen_spklu
        target_loc = spklu_features[chosen_spklu].get('loc', (0.0, 0.0))
        dist_km = math.dist(self.location, target_loc)
        self.remaining_travel_time = (dist_km / speed_kmh) * 60.0
        self.state = UserState.TRAVELING
        return chosen_spklu

    User.decide_spklu = patched
    try:
        yield
    finally:
        User.decide_spklu = orig


# Katalog skenario baseline: nama -> (agent_factory, force_compliance).
# agent_factory: callable -> agent | None. force_compliance: True utk baseline yang
# mengasumsikan kepatuhan penuh eksogen (Baseline 3 / OP-SRL).
SCENARIOS = {
    "S0_no_intervention": (None, False),
    "S1_greedy":          (lambda: GreedyAgent(), False),
    "S2_dchoices":        (lambda: DiversifiedAgent(mode="d_choices", d=2), False),
    "S2_proportional":    (lambda: DiversifiedAgent(mode="proportional", tau=1.0), False),
    "S3_opsrl":           (lambda: OPSRLAgent(), True),
}


def _unpack_scenario(entry):
    """Terima entry lama (agent_factory saja) atau baru (agent_factory, force_compliance)."""
    if isinstance(entry, tuple):
        return entry
    return entry, False


# Perbaikan T1 (Validasi_Generik/LAPORAN_VALIDASI.md): default lama (5.0) BERBEDA dari
# jalur kalibrasi Tier 6/9 (willingness_ratio=None, dipakai utk baseline_wait_mean, gamma,
# ambang trust). Beda skala wait terukur ~2,4-2,85x -- eksperimen lewat harness diam-diam
# memakai ambang trust yg salah skala relatif thd wait di sana. Disamakan ke None (tanpa
# batas jangkauan relatif) supaya SEMUA eksperimen memakai konfigurasi yg sama dgn
# kalibrasi. Klaster 12 berdiameter <5 km -> batas jangkauan relatif jarang mengikat
# scr fisik di sini; lihat V2.2 (LAPORAN_VALIDASI.md) utk arah sensitivitasnya.
DEFAULT_WILLINGNESS_RATIO = None
DEFAULT_WILLINGNESS_RADIUS_KM = None


def run_scenario(dataset_path=DEFAULT_DATASET, agent_factory=None, seed=42, max_steps=None,
                 willingness_ratio=DEFAULT_WILLINGNESS_RATIO,
                 willingness_radius_km=DEFAULT_WILLINGNESS_RADIUS_KM,
                 force_compliance=False):
    """Jalankan satu run (satu seed) dan kembalikan dict KPI lengkap.
    Default eksperimen memakai batas jangkauan rasio-5 (agen range-aware).
    force_compliance=True: pengguna SELALU mengikuti rekomendasi (Baseline 3/OP-SRL)."""
    random.seed(seed)
    np.random.seed(seed)

    sim = Simulator({}, [], None,
                    user_willingness_radius_km=willingness_radius_km,
                    user_willingness_ratio=willingness_ratio)
    sim.load_from_dataset(dataset_path)
    if max_steps is None:
        max_steps = getattr(sim, "max_steps", max(sim.spawn_schedule) + 1)

    # window_size = max_steps -> util_history kronologis penuh tanpa wraparound.
    sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=max_steps)
    agent = agent_factory() if agent_factory else None
    # Agen yang butuh akses ke `sim` (mis. InferenceAgent RL, yang memetakan sids -> idx
    # observasi) baru bisa dikonstruksi penuh setelah `sim` ada -- lihat _DeferredMarlAgent.
    if agent is not None and hasattr(agent, "bind_to_sim"):
        agent.bind_to_sim(sim)
    if force_compliance:
        with force_full_compliance():
            sim.run(max_steps=max_steps, agent=agent)
    else:
        sim.run(max_steps=max_steps, agent=agent)

    gs = metrics.gini_series_from_history(sim.history)
    n_stations = len(sim.spklus)
    return {
        # Kelas 2 -- Dinamika
        "gini_final": float(gs[-1]) if gs.size else 0.0,
        "gini_final_smoothed": metrics.gini_final_smoothed(gs),
        "gini_mean": float(gs.mean()) if gs.size else 0.0,
        "gini_slope": metrics.slope(gs),
        "auc_gini": metrics.auc_gini(gs),
        "time_to_neg_slope_day": metrics.time_to_negative_slope(gs),
        "acceptance_overall": metrics.overall_acceptance_rate(sim.users),
        "acceptance_by_tercile": metrics.acceptance_rate_by_tercile(sim.users),
        "trust": metrics.trust_stats(sim.users),
        # Kelas 3 -- Koordinasi
        "herding_index": metrics.herding_index(sim.rec_distribution_log),
        "herding_events": sim.herding_events,
        "flocking_index": metrics.flocking_index(sim.rec_distribution_log),
        "rec_entropy_norm": metrics.recommendation_entropy(sim.rec_distribution_log, n_stations),
        # Kelas 1 -- Level
        "wait": metrics.wait_stats(sim.logs),
        "wait_by_compliance": metrics.wait_stats_by_compliance(sim.logs),
        "served": metrics.served_stats(sim.spklus),
    }


# KPI skalar yang diagregasi mean +/- std lintas seed.
_SCALAR_KEYS = [
    "gini_final", "gini_final_smoothed", "gini_mean", "gini_slope", "auc_gini",
    "acceptance_overall", "herding_index", "herding_events", "flocking_index",
    "rec_entropy_norm",
]


def run_multi_seed(dataset_path=DEFAULT_DATASET, agent_factory=None, seeds=range(5),
                   max_steps=None, willingness_ratio=DEFAULT_WILLINGNESS_RATIO,
                   willingness_radius_km=DEFAULT_WILLINGNESS_RADIUS_KM,
                   force_compliance=False):
    """Jalankan satu skenario lintas beberapa seed; agregasi KPI skalar (mean, std)."""
    seeds = list(seeds)
    runs = [run_scenario(dataset_path, agent_factory, seed=s, max_steps=max_steps,
                         willingness_ratio=willingness_ratio,
                         willingness_radius_km=willingness_radius_km,
                         force_compliance=force_compliance)
            for s in seeds]

    agg = {"n_seeds": len(seeds)}
    for k in _SCALAR_KEYS:
        vals = np.array([r[k] for r in runs], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    # time-to-neg-slope: bisa None -> laporkan jumlah seed yang mencapainya + median hari
    tts = [r["time_to_neg_slope_day"] for r in runs]
    reached = [t for t in tts if t is not None]
    agg["time_to_neg_slope"] = {
        "n_reached": len(reached), "n_seeds": len(seeds),
        "median_day": float(np.median(reached)) if reached else None,
    }
    # wait & served & trust: ambil rata-rata mean antar-seed (ringkas)
    agg["wait_mean"] = float(np.mean([r["wait"]["mean"] for r in runs]))
    agg["wait_p90"] = float(np.mean([r["wait"]["p90"] for r in runs]))
    agg["frac_inactive"] = float(np.mean([r["served"]["frac_inactive"] for r in runs]))
    agg["trust_mean"] = float(np.mean([r["trust"].get("mean", 0.5) for r in runs]))
    agg["trust_frac_below_0.5"] = float(np.mean([r["trust"].get("frac_below_0.5", 0.0) for r in runs]))

    # Objektif 2 (III.2.2, para. 484): wait_mean(patuh) tak boleh >20% lebih buruk dari
    # wait_mean(default). Agregasi lintas seed: rata-rata mean tiap grup, lalu rasio dari
    # rata-rata itu (bukan rata-rata rasio per-seed -- lebih stabil saat n kecil per seed).
    wc_complied_means = [r["wait_by_compliance"]["complied"]["mean"] for r in runs
                         if r["wait_by_compliance"]["complied"]["n"] > 0]
    wc_default_means = [r["wait_by_compliance"]["default"]["mean"] for r in runs
                        if r["wait_by_compliance"]["default"]["n"] > 0]
    wait_complied_mean = float(np.mean(wc_complied_means)) if wc_complied_means else None
    wait_default_mean = float(np.mean(wc_default_means)) if wc_default_means else None
    wait_compliance_ratio = (wait_complied_mean / wait_default_mean
                             if wait_complied_mean is not None and wait_default_mean
                             else None)
    agg["wait_by_compliance"] = {
        "complied_mean": wait_complied_mean, "default_mean": wait_default_mean,
        "ratio": wait_compliance_ratio,
        "criterion_20pct_ok": (wait_compliance_ratio <= 1.2)
                              if wait_compliance_ratio is not None else None,
    }
    agg["_runs"] = runs
    return agg


def compare_scenarios(dataset_path=DEFAULT_DATASET, scenario_names=None, seeds=range(5),
                      max_steps=None, willingness_ratio=DEFAULT_WILLINGNESS_RATIO,
                      willingness_radius_km=DEFAULT_WILLINGNESS_RADIUS_KM,
                      extra_scenarios=None):
    """Jalankan beberapa skenario dan kembalikan dict {nama: agregat}.
    extra_scenarios: dict tambahan {nama: (agent_factory, force_compliance)} yang digabung
    dengan katalog SCENARIOS (mis. skenario RL yang butuh policy terlatih saat runtime)."""
    catalog = dict(SCENARIOS)
    if extra_scenarios:
        catalog.update(extra_scenarios)
    if scenario_names is None:
        scenario_names = list(catalog.keys())
    results = {}
    for name in scenario_names:
        agent_factory, force_compliance = _unpack_scenario(catalog[name])
        results[name] = run_multi_seed(dataset_path, agent_factory, seeds=seeds,
                                       max_steps=max_steps,
                                       willingness_ratio=willingness_ratio,
                                       willingness_radius_km=willingness_radius_km,
                                       force_compliance=force_compliance)
    return results


def format_comparison(results) -> str:
    """Tabel teks ringkas perbandingan skenario (untuk konsol / laporan)."""
    cols = ["gini_slope", "auc_gini", "herding_index", "flocking_index", "rec_entropy_norm",
            "acceptance_overall", "trust_mean", "wait_mean", "frac_inactive"]
    header = f"{'skenario':<20}" + "".join(f"{c:>16}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, agg in results.items():
        row = f"{name:<20}"
        for c in cols:
            v = agg[c]["mean"] if isinstance(agg.get(c), dict) else agg.get(c)
            row += f"{v:>16.5f}" if isinstance(v, float) else f"{str(v):>16}"
        lines.append(row)
    lines.append("")
    lines.append("Objektif 2 (III.2.2): wait_mean(patuh) vs wait_mean(default), ratio<=1.2 -> OK")
    wc_header = f"{'skenario':<20}{'wait_complied':>16}{'wait_default':>16}{'ratio':>10}{'ok?':>6}"
    lines.append(wc_header)
    lines.append("-" * len(wc_header))
    for name, agg in results.items():
        wc = agg.get("wait_by_compliance", {})
        cm, dm, ratio, ok = wc.get("complied_mean"), wc.get("default_mean"), wc.get("ratio"), wc.get("criterion_20pct_ok")
        row = f"{name:<20}"
        row += f"{cm:>16.3f}" if cm is not None else f"{'n/a':>16}"
        row += f"{dm:>16.3f}" if dm is not None else f"{'n/a':>16}"
        row += f"{ratio:>10.3f}" if ratio is not None else f"{'n/a':>10}"
        row += f"{str(ok):>6}" if ok is not None else f"{'n/a':>6}"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Runner baseline S0/S1/S2 (non-RL).")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--seeds", type=int, default=5, help="jumlah seed (0..N-1)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--scenarios", type=str, default=None,
                   help="daftar skenario dipisah koma; default semua di SCENARIOS")
    p.add_argument("--out", type=str, default=None, help="tulis hasil agregat ke JSON")
    args = p.parse_args()

    names = args.scenarios.split(",") if args.scenarios else None
    res = compare_scenarios(args.dataset, names, seeds=range(args.seeds),
                            max_steps=args.max_steps)
    print(format_comparison(res))

    if args.out:
        slim = {n: {k: v for k, v in agg.items() if k != "_runs"} for n, agg in res.items()}
        with open(args.out, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"\n[INFO] hasil agregat ditulis ke {args.out}")
