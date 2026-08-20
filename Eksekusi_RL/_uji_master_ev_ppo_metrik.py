"""Evaluasi MasterEV-PPO dengan METRIK KAYA yang sama dengan `_uji_konsolidasi.py` --
mirror `_uji_master_ev_metrik.py`, checkpoint dari `_run_master_ev_ppo_pipeline.py`.

Pemakaian:
    python _uji_master_ev_ppo_metrik.py 0,1,2 30d [tag_arm]
    tag_arm opsional -- default "master_ev_ppo". Utk varian +P (lihat `--pref`/
    `--pref-feature-mode` di `_run_master_ev_ppo_pipeline.py`), berikan eksplisit:
    "master_ev_ppo_pref" atau "master_ev_ppo_pref_feat".
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOPolicy, MasterEVPPOPrefPolicy,
                                                MasterEVPPOInferenceAgent)
from marl_spklu.rl.forecaster import FormulaForecaster

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_ev_ppo"
# PERBAIKAN: pencocokan SUBSTRING (bukan `==` eksak) -- tag_arm bisa dapat akhiran
# horizon (mis. "master_ev_ppo_pref_feat_90d"), pencocokan eksak lama membuat
# pref_feature_mode salah terdeteksi False -> dim LSTM riwayat salah (12 bukan 10) ->
# RuntimeError size-mismatch saat load_state_dict.
_is_pref_feat = "pref_feat" in TAG_ARM
_is_pref = "pref" in TAG_ARM
LABEL_ARM = (f"MASTER-EV-PPO+P(fitur)[{TAG_ARM}]" if _is_pref_feat
            else f"MASTER-EV-PPO+P[{TAG_ARM}]" if _is_pref
            else f"MASTER-EV-PPO[{TAG_ARM}]")
POLICY_CLS = MasterEVPPOPrefPolicy if _is_pref else MasterEVPPOPolicy
POLICY_KW = dict(pref_feature_mode=True) if _is_pref_feat else dict()
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])
K_REC = 3


def muat_policy(seed, n_spklu):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    assert os.path.exists(ckpt), (
        f"checkpoint tak ditemukan: {ckpt}\n"
        f"latih dulu lewat: python _run_master_ev_ppo_pipeline.py --n-train-seed {seed+1} ...")
    pol = POLICY_CLS(n_spklu, **POLICY_KW)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


def fac_dari_policy(policy):
    def fac(sim, _pol=policy):
        return MasterEVPPOInferenceAgent(_pol, sim, FormulaForecaster(), k=K_REC)
    return fac


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)
    sim0 = common.fresh_sim(K.DS)
    n_spklu = len(sim0.spklus)

    per_seed = {}
    agregat = {}
    harian = {}
    for mode in ("abs", "signed"):
        for beku in (False, True):
            label = f"{LABEL_ARM}|{mode}|{'beku' if beku else 'dinamis'}"
            runs = []
            for sd in SEEDS:
                policy = muat_policy(sd, n_spklu)
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
