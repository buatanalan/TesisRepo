"""Evaluasi greedy_util/greedy_queue dengan metrik kaya IDENTIK metodologi dgn
`_uji_master_ev_ppo_metrik.py` (K.satu_run, K.agg) -- pembanding apple-to-apple utk
hasil MASTER-EV-PPO DGR (gap_ratio).

Pemakaian:
    python _uji_greedy_metrik.py 0,1,2 30d
    python _uji_greedy_metrik.py 0,1,2 90d
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.agents.greedy_agent import GreedyAgent

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])

ARMS = [("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2)),
        ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=2))]


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)

    per_seed = {}
    agregat = {}
    harian = {}
    for lbl, fac in ARMS:
        for mode in ("abs", "signed"):
            for beku in (False, True):
                label = f"{lbl}|{mode}|{'beku' if beku else 'dinamis'}"
                runs = []
                for sd in SEEDS:
                    print(f"  [{label}] seed={sd} ...", flush=True)
                    r = K.satu_run(fac, mode, sd, beku)
                    runs.append(r)
                per_seed[label] = runs
                agregat[label] = K.agg(runs)
                harian[label] = K.agg_harian(runs)
                print(f"  [{label}] gini={agregat[label]['gini']:.4f} "
                     f"wait={agregat[label]['wait']:.1f} trust={agregat[label]['trust']:.3f} "
                     f"acc={agregat[label]['acc']:.3f}", flush=True)

    out = dict(horizon=TAG, seeds=SEEDS, per_seed=per_seed, agregat=agregat, harian=harian)
    nama = f"uji_greedy_metrik_{TAG}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    main()
