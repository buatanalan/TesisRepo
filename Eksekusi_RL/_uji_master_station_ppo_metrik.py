"""Evaluasi MASTER-stasiun+PPO dengan METRIK KAYA yang sama dengan `_uji_konsolidasi.py`
-- pola SAMA PERSIS `_uji_master_ddpg_metrik.py` (lihat docstring berkas itu utk alasan
kenapa terpisah dari loop 19-lengan).

Pemakaian (checkpoint HARUS sudah ada, hasil `_run_master_station_ppo_pipeline.py`):
    python _uji_master_station_ppo_metrik.py 0,1,2 30d
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_station_ppo_policy import (MasterStationPPOPolicy,
                                                      MasterStationPPOInferenceAgent)

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
         if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])
K_REC = 3


def muat_policy(seed):
    ckpt = os.path.join(common.OUTDIR, f"master_station_ppo_seed{seed}.pt")
    assert os.path.exists(ckpt), (
        f"checkpoint tak ditemukan: {ckpt}\n"
        f"latih dulu lewat: python _run_master_station_ppo_pipeline.py --n-train-seed "
        f"{seed+1} ...")
    sim0 = common.fresh_sim(K.DS)
    n_spklu = len(sim0.spklus)
    policy = MasterStationPPOPolicy(n_spklu)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()
    return policy


def fac_dari_policy(policy):
    def fac(sim, _policy=policy):
        return MasterStationPPOInferenceAgent(_policy, sim, k=K_REC)
    return fac


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)

    per_seed = {}
    agregat = {}
    harian = {}
    for mode in ("abs", "signed"):
        for beku in (False, True):
            label = f"MASTER-stasiun-PPO|{mode}|{'beku' if beku else 'dinamis'}"
            runs = []
            for sd in SEEDS:
                policy = muat_policy(sd)
                fac = fac_dari_policy(policy)
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
    common.save_json(out, f"uji_master_station_ppo_metrik_{TAG}.json")
    print(f"\nSAVED -> outputs/uji_master_station_ppo_metrik_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
