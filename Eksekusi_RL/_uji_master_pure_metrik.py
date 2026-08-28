"""Evaluasi MASTER benar-benar murni (`master_pure_policy.py`/`master_pure_trainer.py`)
dgn METRIK KAYA yg sama dgn `_uji_konsolidasi.py` -- mirror `_uji_master_ev_ppo_metrik.py`.
Arsitektur aktor (`MasterPureActor`) IDENTIK lintas 3 tag (`specialist0_wait`,
`specialist1_gini`, `dgr`) -- hanya proses latihnya yg beda, jadi tak perlu deteksi
varian arsitektur spt skrip MasterEVPPO.

Pemakaian:
    python _uji_master_pure_metrik.py 0,1,2 30d master_pure_dgr
    python _uji_master_pure_metrik.py 0,1,2 30d master_pure_specialist0_wait
    python _uji_master_pure_metrik.py 0,1,2 30d master_pure_specialist1_gini
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_pure_policy import MasterPureActor
from marl_spklu.rl.master_pure_trainer import MasterPureInferenceAgent
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER
from marl_spklu.rl.forecaster import FormulaForecaster

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_pure_dgr"
LABEL_ARM = f"MASTER-PURE[{TAG_ARM}]"
K_REC = 3
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])


def muat_policy(seed):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    assert os.path.exists(ckpt), (
        f"checkpoint tak ditemukan: {ckpt}\n"
        f"latih dulu lewat: python _run_master_pure_pipeline.py --n-train-seed {seed+1} ...")
    pol = MasterPureActor(STATION_FEAT_DIM_MASTER)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


def fac_dari_policy(policy):
    def fac(sim, _pol=policy):
        agent = MasterPureInferenceAgent(_pol, forecaster=FormulaForecaster(), k=K_REC)
        agent.bind_to_sim(sim)
        return agent
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
