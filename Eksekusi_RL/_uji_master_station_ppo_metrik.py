"""Evaluasi MASTER-stasiun+PPO dengan METRIK KAYA yang sama dengan `_uji_konsolidasi.py`
-- pola SAMA PERSIS `_uji_master_ddpg_metrik.py` (lihat docstring berkas itu utk alasan
kenapa terpisah dari loop 19-lengan).

Pemakaian (checkpoint HARUS sudah ada, hasil `_run_master_station_ppo_pipeline.py`):
    python _uji_master_station_ppo_metrik.py 0,1,2 30d                       # tanpa-P
    python _uji_master_station_ppo_metrik.py 0,1,2 30d --pref                # +P (identitas)
    python _uji_master_station_ppo_metrik.py 0,1,2 30d --pref --pref-feature-mode
                                                                              # +P (fitur kaya)
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
                                                      MasterStationPPOPrefPolicy,
                                                      MasterStationPPOInferenceAgent)

_FLAGS = {"--pref", "--pref-feature-mode"}
_argv_bersih = [a for a in _argv_asli if a not in _FLAGS]
PREF = "--pref" in _argv_asli
PREF_FEAT = "--pref-feature-mode" in _argv_asli
if PREF_FEAT and not PREF:
    raise SystemExit("--pref-feature-mode hanya berlaku bersama --pref")
POLICY_CLS = MasterStationPPOPrefPolicy if PREF else MasterStationPPOPolicy
POLICY_KW = {"pref_feature_mode": True} if PREF_FEAT else {}
if PREF_FEAT:
    TAG_ARM = "master_station_ppo_pref_feat"
    LABEL_ARM = "MASTER-stasiun-PPO+P(fitur)"
elif PREF:
    TAG_ARM = "master_station_ppo_pref"
    LABEL_ARM = "MASTER-stasiun-PPO+P"
else:
    TAG_ARM = "master_station_ppo"
    LABEL_ARM = "MASTER-stasiun-PPO"

SEEDS = ([int(s) for s in _argv_bersih[1].replace(" ", "").split(",") if s]
         if len(_argv_bersih) > 1 else [0, 1, 2])
TAG = _argv_bersih[2] if len(_argv_bersih) > 2 else "30d"
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])
K_REC = 3


def muat_policy(seed):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_seed{seed}.pt")
    assert os.path.exists(ckpt), (
        f"checkpoint tak ditemukan: {ckpt}\n"
        f"latih dulu lewat: python _run_master_station_ppo_pipeline.py "
        f"{'--pref ' if PREF else ''}--n-train-seed {seed+1} ...")
    sim0 = common.fresh_sim(K.DS)
    n_spklu = len(sim0.spklus)
    policy = POLICY_CLS(n_spklu, **POLICY_KW)
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
            label = f"{LABEL_ARM}|{mode}|{'beku' if beku else 'dinamis'}"
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
    nama = f"uji_{TAG_ARM}_metrik_{TAG}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    main()
