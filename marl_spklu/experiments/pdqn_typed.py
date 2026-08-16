"""Eksperimen replikasi PENUH PDQN (Lin et al. 2024) pada lingkungan 5 TIPE PREFERENSI.

Menjalankan set ABLASI yang selama ini hilang -- inilah yang menjawab klaim inti paper,
yaitu bahwa modul preferensi (LSTM per-tipe + attention) memberi keunggulan:

  S0  Natural              tanpa intervensi
  S1  Greedy-util          argmin utilisasi ternormalisasi (baseline spesifikasi)
  S1q Greedy-queue         argmin antrean (baseline "minimum queuing" paper)
  D   DQN tanpa preferensi jaringan sama, c_t dipaksa nol (baseline ketiga paper)
  P1  PDQN 1-LSTM          satu LSTM dibagi semua tipe
  P5  PDQN 5-LSTM          satu LSTM per tipe (desain paper)

Perbandingan yang menentukan:
  P5 vs D   -> kontribusi INFORMASI preferensi (arsitektur & kapasitas identik)
  P5 vs P1  -> kontribusi PEMISAHAN per-tipe (klaim eksplisit paper §IV.A bahwa satu
               LSTM untuk semua tipe berkinerja buruk)

Pakai:
    python -m marl_spklu.experiments.pdqn_typed --n-updates 100 --seeds 2
"""
import argparse
import json

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.experiments import harness
from marl_spklu.experiments.ablations import binary_recommendation_mode, constant_trust
from marl_spklu.experiments.pdqn_baseline import train_pdqn
from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent

DEFAULT_DATASET = "scenario_dataset_typed_7d.json"

# (label, kwargs trainer). None = baseline non-RL, ditangani terpisah.
RL_VARIANTS = [
    ("D_dqn_nopref", dict(use_preference=False)),
    ("P1_pdqn_1lstm", dict(n_types=1)),
    ("P5_pdqn_5lstm", dict()),          # n_types dideteksi otomatis dari dataset -> 5
]


def run(dataset, mu_values, n_updates, rollout_steps, seeds, train_seed, stochastic, verbose):
    out = {}
    for mu in mu_values:
        res = {}
        # --- baseline non-RL (tak perlu training) ---
        with binary_recommendation_mode(mu, stochastic), constant_trust(mu):
            res["S0_natural"] = harness.run_multi_seed(dataset, None, seeds=seeds)
            res["S1_greedy_util"] = harness.run_multi_seed(
                dataset, lambda: GreedyAgent(mode="utilization"), seeds=seeds)
            res["S1q_greedy_queue"] = harness.run_multi_seed(
                dataset, lambda: GreedyAgent(mode="queue"), seeds=seeds)

        # --- varian RL ---
        for label, kw in RL_VARIANTS:
            if verbose:
                print(f"\n=== mu={mu} | melatih {label} ===")
            q_net, tr = train_pdqn(dataset, mu, n_updates=n_updates, rollout_steps=rollout_steps,
                                   seed=train_seed, verbose=verbose, stochastic=stochastic, **kw)
            with binary_recommendation_mode(mu, stochastic), constant_trust(mu):
                res[label] = harness.run_multi_seed(
                    dataset, lambda qn=q_net: PDQNInferenceAgent(qn), seeds=seeds)
            res[label]["_n_params"] = sum(p.numel() for p in q_net.parameters())
            res[label]["_n_types"] = tr.n_types
            res[label]["_use_preference"] = tr.use_preference
        out[mu] = res
    return out


def summarize(out):
    lines = []
    for mu, res in out.items():
        lines.append(f"\n=== mu_hat = {mu} ===")
        lines.append(f"{'skenario':<20}{'gini_mean':>18}{'kepatuhan':>11}{'herding':>9}"
                     f"{'entropi':>9}{'wait':>8}")
        lines.append("-" * 75)
        for name, a in res.items():
            g = a["gini_mean"]
            lines.append(f"{name:<20}{g['mean']:>10.4f}+-{g['std']:.4f}"
                         f"{a['acceptance_overall']['mean']:>11.3f}"
                         f"{a['herding_index']['mean']:>9.3f}"
                         f"{a['rec_entropy_norm']['mean']:>9.3f}{a['wait_mean']:>8.1f}")
        d = res.get("D_dqn_nopref", {}).get("gini_mean", {}).get("mean")
        p1 = res.get("P1_pdqn_1lstm", {}).get("gini_mean", {}).get("mean")
        p5 = res.get("P5_pdqn_5lstm", {}).get("gini_mean", {}).get("mean")
        if None not in (d, p1, p5):
            lines.append(f"  P5 vs DQN-tanpa-preferensi : {d - p5:+.4f}  (kontribusi informasi preferensi)")
            lines.append(f"  P5 vs PDQN-1-LSTM          : {p1 - p5:+.4f}  (kontribusi pemisahan per-tipe)")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--n-updates", type=int, default=100)
    p.add_argument("--rollout-steps", type=int, default=96)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--mu", default="0.2,0.5,0.8")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    res = run(a.dataset, [float(x) for x in a.mu.split(",")], a.n_updates, a.rollout_steps,
              range(a.seeds), a.train_seed, a.stochastic, verbose=True)
    print(summarize(res))
    if a.out:
        slim = {str(mu): {n: {k: v for k, v in agg.items() if k != "_runs"}
                          for n, agg in r.items()} for mu, r in res.items()}
        with open(a.out, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"\n[INFO] -> {a.out}")
