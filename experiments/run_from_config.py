import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Runner tunggal utk menjalankan skenario solusi utama dari file YAML eksternal.

Alih-alih mengedit skrip Python atau menghafal flag CLI, parameter (dataset, jumlah
update, seed, dst.) ditulis di file .yaml. Runner ini memuatnya dan memanggil
`run_solusi_utama` (experiments/training/run_solusi_utama.py) satu kali per
kombinasi run x seed yang didefinisikan.

Pemakaian:
    python experiments/run_from_config.py experiments/configs/contoh.yaml
    python experiments/run_from_config.py experiments/configs/contoh.yaml --run only_this_run
    python experiments/run_from_config.py experiments/configs/contoh.yaml --dry-run

Format YAML minimal (lihat experiments/configs/contoh.yaml utk contoh lengkap):

    defaults:
      dataset: scenario_dataset_calibrated_main.json
      updates: 200
      rollout_steps: 96
      k: 3
      eval_seeds: 5
      collect_steps: 500
      out_dir: outputs/results

    runs:
      - name: solusi_utama
        seeds: [0, 1, 2]
      - name: rollout_steps_60
        params:
          rollout_steps: 60
        seeds: [0]

Tiap entri di `runs` mewarisi `defaults`, lalu override lewat `params`. Skrip
menjalankan `run_solusi_utama` sekali per seed dan menyimpan hasil ke
`{out_dir}/{run_name}_seed{seed}.json`.
"""
import argparse
import copy
import json
import time

import yaml

from experiments.training.run_solusi_utama import run_solusi_utama, DATASET_DEFAULT

REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Parameter yg diteruskan langsung ke run_solusi_utama() -- nama field YAML harus
# persis sama dgn ini (kecuali 'dataset', yg dresolusi relatif ke repo root).
_PASSTHROUGH_KEYS = [
    "dataset", "updates", "rollout_steps", "k", "seed", "eval_seeds",
    "max_steps", "collect_steps",
]


def _resolve_path(path):
    if path is None or _os.path.isabs(path):
        return path
    return _os.path.join(REPO_ROOT, path)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", {}) or {}
    runs = cfg.get("runs", [])
    if not runs:
        raise ValueError(f"Config '{config_path}' tidak punya entri 'runs'.")
    return defaults, runs


def build_run_plan(defaults, runs, only_run=None):
    """Gabungkan defaults+params per-run, ekspansi ke satu entri per seed."""
    plan = []
    for run in runs:
        name = run["name"]
        if only_run and name != only_run:
            continue
        merged = copy.deepcopy(defaults)
        merged.update(run.get("params", {}) or {})
        seeds = run.get("seeds", [merged.get("seed", 0)])
        out_dir = merged.pop("out_dir", "outputs/results")
        for seed in seeds:
            entry = {k: v for k, v in merged.items() if k in _PASSTHROUGH_KEYS}
            entry["seed"] = seed
            entry["dataset"] = _resolve_path(entry.get("dataset", DATASET_DEFAULT))
            out_path = _resolve_path(_os.path.join(out_dir, f"{name}_seed{seed}.json"))
            plan.append({"run_name": name, "params": entry, "out_path": out_path})
    if not plan:
        available = [r["name"] for r in runs]
        raise ValueError(f"Tidak ada run cocok dgn '{only_run}'. Run tersedia: {available}")
    return plan


def execute_plan(plan, dry_run=False):
    results = []
    for i, entry in enumerate(plan, 1):
        print(f"\n{'=' * 100}")
        print(f"[{i}/{len(plan)}] run={entry['run_name']} seed={entry['params']['seed']} "
             f"-> {entry['out_path']}")
        print(f"{'=' * 100}")
        if dry_run:
            print(json.dumps(entry["params"], indent=2))
            continue
        _os.makedirs(_os.path.dirname(entry["out_path"]), exist_ok=True)
        t0 = time.time()
        run_solusi_utama(out_path=entry["out_path"], **entry["params"])
        dt = time.time() - t0
        print(f"[SELESAI] {entry['run_name']} seed={entry['params']['seed']} dlm {dt:.1f}s")
        results.append(entry["out_path"])
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="Path ke file .yaml (mis. experiments/configs/contoh.yaml)")
    ap.add_argument("--run", default=None, help="Jalankan hanya satu entri 'runs' by name.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Cetak rencana eksekusi (params tiap run x seed) tanpa training.")
    args = ap.parse_args()

    defaults, runs = load_config(args.config)
    plan = build_run_plan(defaults, runs, only_run=args.run)

    print(f"Rencana eksekusi: {len(plan)} run (dari {len(runs)} entri config x seed).")
    for e in plan:
        print(f"  - {e['run_name']} seed={e['params']['seed']} -> {e['out_path']}")

    execute_plan(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
