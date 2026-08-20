"""Evaluasi MasterEV dengan METRIK KAYA yang sama dengan `_uji_konsolidasi.py` --
mirror `_uji_master_ddpg_metrik.py`, lihat docstring di sana utk alasan terpisah
dari loop 19-lengan.

Pemakaian (checkpoint HARUS sudah ada, hasil `_run_master_ev_pipeline.py`):
    python _uji_master_ev_metrik.py 0,1,2 30d [tag_arm]
    tag_arm opsional -- default "master_ev" (preset seimbang4x+gap_ratio baku). Utk
    varian ablasi (mis. "master_ev_default" atau "master_ev_default_fixed", lihat
    akhiran tag di `_run_master_ev_pipeline.py`), berikan eksplisit sbg argumen ke-3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_ev_policy import MasterEVActor
from marl_spklu.rl.master_ev_trainer import MasterEVInferenceAgent
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER_EV

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_ev"
LABEL_ARM = "MASTER-EV" if TAG_ARM == "master_ev" else f"MASTER-EV[{TAG_ARM}]"
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])
K_REC = 3


def muat_actor(seed):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    assert os.path.exists(ckpt), (
        f"checkpoint tak ditemukan: {ckpt}\n"
        f"latih dulu lewat: python _run_master_ev_pipeline.py --n-train-seed {seed+1} ...")
    actor = MasterEVActor(STATION_FEAT_DIM_MASTER_EV)
    actor.load_state_dict(torch.load(ckpt, map_location="cpu"))
    actor.eval()
    return actor


def fac_dari_actor(actor):
    def fac(sim, _actor=actor):
        agent = MasterEVInferenceAgent(_actor, k=K_REC)
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
                actor = muat_actor(sd)
                fac = fac_dari_actor(actor)
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
