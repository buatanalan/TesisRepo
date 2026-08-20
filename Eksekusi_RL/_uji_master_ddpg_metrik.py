"""Evaluasi MASTER-DDPG dengan METRIK KAYA yang sama dengan `_uji_konsolidasi.py` --
TANPA menggabungkannya ke loop 19-lengan. Metode diuji satu per satu; kekayaan metriknya
tetap sama (141 skalar + deret harian + ringkasan pengguna/stasiun).

Kenapa terpisah dari `_uji_konsolidasi.py`, bukan ditambahkan sebagai lengan ke-20:
MASTER-DDPG punya bentuk kebijakan yang sama sekali berbeda (aktor per-stasiun +
observasi §3.1 murni, `marl_spklu/rl/master_paper_obs.py`) -- BUKAN `HPPOPolicy` yang
dimuat lewat `registry.py::bangun_kebijakan`. Memaksanya masuk skema `pol(stem, seed)`
lama akan menuntut cabang khusus di tengah loop yang sudah padat.

Fungsi metrik (`satu_run`, `agg`, `agg_harian`, `ringkas_stasiun`, `gini`, `mode_trust`)
DIIMPOR LANGSUNG dari `_uji_konsolidasi.py`, bukan diduplikasi -- satu sumber kebenaran
untuk definisi tiap metrik. `sys.argv` disamarkan sementara sebelum impor krn modul itu
mem-parse argv di level modul (utk SEEDS/TAG/DS-nya sendiri) -- lihat blok impor.

Pemakaian (checkpoint HARUS sudah ada, hasil `_run_master_ddpg_pipeline.py`):
    python _uji_master_ddpg_metrik.py 0,1,2 30d            # §3.1 murni
    python _uji_master_ddpg_metrik.py 0,1,2 30d --ev-obs   # +state permintaan
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

# --- impor _uji_konsolidasi dgn argv disamarkan (lihat docstring) ---
_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_ddpg_policy import MasterStationActor
from marl_spklu.rl.master_ddpg_trainer import MasterDDPGInferenceAgent
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER, STATION_FEAT_DIM_MASTER_EV

_argv_bersih = [a for a in _argv_asli if a != "--ev-obs"]
EV_OBS = "--ev-obs" in _argv_asli
STATION_FEAT_DIM = STATION_FEAT_DIM_MASTER_EV if EV_OBS else STATION_FEAT_DIM_MASTER
TAG_ARM = "master_ddpg_ev" if EV_OBS else "master_ddpg"
LABEL_ARM = "MASTER-DDPG+EV" if EV_OBS else "MASTER-DDPG"

SEEDS = ([int(s) for s in _argv_bersih[1].replace(" ", "").split(",") if s]
         if len(_argv_bersih) > 1 else [0, 1, 2])
TAG = _argv_bersih[2] if len(_argv_bersih) > 2 else "30d"
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])   # ikut regime 4x, BUKAN DATASET_KANONIK
K_REC = 3   # langit-langit rekomendasi -- samakan dgn --k pipeline pelatihan


def muat_actor(seed):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    assert os.path.exists(ckpt), (
        f"checkpoint tak ditemukan: {ckpt}\n"
        f"latih dulu lewat: python _run_master_ddpg_pipeline.py "
        f"{'--ev-obs ' if EV_OBS else ''}--n-train-seed {seed+1} ...")
    actor = MasterStationActor(STATION_FEAT_DIM)
    actor.load_state_dict(torch.load(ckpt, map_location="cpu"))
    actor.eval()
    return actor


def fac_dari_actor(actor):
    """fac(sim) -> agen terikat -- kontrak yang sama dipakai `K.satu_run`."""
    def fac(sim, _actor=actor):
        agent = MasterDDPGInferenceAgent(_actor, k=K_REC, use_ev_obs=EV_OBS)
        agent.bind_to_sim(sim)
        return agent
    return fac


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}  ev_obs={EV_OBS}", flush=True)

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
