"""Evaluasi lengan PURE3 (mode aliran-murni wait/gini/acceptance) dgn metrik kaya sama
`_uji_konsolidasi.py` -- mirror `_uji_master_pure_hybrid_ppo_metrik.py`, TERMASUK pola
LATIH-3/EVAL-10 (checkpoint dipakai BERGILIR `seed % len(checkpoint)`, agregasi per-
checkpoint bukan mean mentah 10 run -- lih. `ckpt_untuk`). Mendukung DUA tulang punggung
lewat argumen `--backbone`:
    ppo  -> `MasterPurePPOActor`/`MasterPurePPOInferenceAgent`  (master_pure_ppo_trainer.py)
    ddpg -> `MasterPureActor`/`MasterPureInferenceAgent`        (master_pure_trainer.py)

Dipakai utk membandingkan backbone PPO vs DDPG (Tabel VI.1/VI.2) PADA SKEMA REWARD YANG
SAMA dgn lengan Hybrid (3 aliran murni), menutup inkonsistensi metodologis lama (Tabel
VI.1/VI.2 sebelumnya dilatih dgn skema 2-aliran berbeda dari seluruh hasil ablasi utama).

Pemakaian:
    python _uji_master_pure_ppo_pure3_metrik.py 0,1,2 30d master_pure_ppo_pure3_dgr ppo
    python _uji_master_pure_ppo_pure3_metrik.py 0,1,2 30d master_pure_ddpg_pure3_dgr ddpg
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER
from marl_spklu.rl.forecaster import FormulaForecaster

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_pure_ppo_pure3_dgr"
BACKBONE = sys.argv[4] if len(sys.argv) > 4 else "ppo"
assert BACKBONE in ("ppo", "ddpg"), f"--backbone tak dikenal: {BACKBONE!r} (pilih ppo/ddpg)"
LABEL_ARM = f"MASTER-{'PPO' if BACKBONE == 'ppo' else 'DDPG'}-PURE3[{TAG_ARM}]"
K_REC = 3
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])

if BACKBONE == "ppo":
    from marl_spklu.rl.master_pure_ppo_policy import MasterPurePPOActor as _Actor
    from marl_spklu.rl.master_pure_ppo_trainer import MasterPurePPOInferenceAgent as _InferAgent
else:
    from marl_spklu.rl.master_pure_policy import MasterPureActor as _Actor
    from marl_spklu.rl.master_pure_trainer import MasterPureInferenceAgent as _InferAgent


def _checkpoint_tersedia():
    """Daftar indeks seed checkpoint yang BENAR-BENAR ada, terurut -- pola PERSIS sama
    `_uji_master_pure_hybrid_ppo_metrik.py::_checkpoint_tersedia`."""
    ada = []
    i = 0
    while True:
        if not os.path.exists(os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{i}.pt")):
            break
        ada.append(i)
        i += 1
    return ada


_CKPT = _checkpoint_tersedia()


def ckpt_untuk(seed):
    """Pola LATIH 3 SEED / EVAL 10 SEED: checkpoint dipakai BERGILIR (`seed % jumlah_
    checkpoint`), bukan jatuh semua ke seed 0 -- lih. catatan statistik lengkap di
    `_uji_master_pure_hybrid_ppo_metrik.py::ckpt_untuk` (simpangan baku yang layak
    dilaporkan = simpangan baku RERATA PER-CHECKPOINT, bukan 10 run mentah)."""
    assert _CKPT, (
        f"tak ada checkpoint sama sekali utk {TAG_ARM}\n"
        f"latih dulu: python _run_master_{'pure_ppo' if BACKBONE == 'ppo' else 'ddpg'}"
        f"_pure3_pipeline.py --n-train-seed 3 ...")
    return _CKPT[seed % len(_CKPT)]


def muat_policy(seed):
    c = ckpt_untuk(seed)
    if c != seed:
        print(f"  [bergilir] seed eval={seed} -> checkpoint seed{c} "
             f"(simulasi tetap seed={seed}; {len(_CKPT)} checkpoint tersedia)", flush=True)
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{c}.pt")
    pol = _Actor(STATION_FEAT_DIM_MASTER)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


def fac_dari_policy(policy):
    def fac(sim, _pol=policy):
        agent = _InferAgent(_pol, forecaster=FormulaForecaster(), k=K_REC)
        agent.bind_to_sim(sim)
        return agent
    return fac


def main():
    print(f"backbone={BACKBONE} horizon={TAG} ({K.DS})", flush=True)
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

    # `ckpt_per_seed` WAJIB ikut tersimpan -- lih. catatan statistik di `ckpt_untuk`.
    out = dict(horizon=TAG, backbone=BACKBONE, seeds=SEEDS, per_seed=per_seed,
               agregat=agregat, harian=harian,
               ckpt_per_seed={int(s): int(ckpt_untuk(s)) for s in SEEDS},
               n_checkpoint=len(_CKPT))
    nama = f"uji_{TAG_ARM}_metrik_{TAG}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    main()
