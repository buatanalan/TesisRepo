"""Faktorial 2x2 (metode x aturan-trust) pada trust DINAMIS, horizon dapat dipilih.

  abs    -- aturan berlaku sekarang, |actual - est| (Pemodelan_Variasi_Distribusi.md 7.2)
  signed -- hanya keterlambatan (actual > est) menghukum; over-estimasi netral

Faktor aturan-trust HARUS berpasangan: tanpa lengan "abs" pada horizon yang SAMA, tiap
perbedaan tak dapat diatribusikan -- bisa krn horizon, bisa krn aturan.

Pemakaian:
    python _uji_aturan_trust.py                 # seed 0, horizon 90d (default lama)
    python _uji_aturan_trust.py 0,1,2 30d       # 3 seed, horizon 30 hari
    python _uji_aturan_trust.py 1,2,3,4 90d     # seed tambahan, horizon 90 hari

Stem keluaran memuat horizon (`hppo_30d_abs`, `hppo_90d_abs`, ...) sehingga hasil kedua
horizon tak pernah saling menimpa.

CATATAN daya statistik: dgn 3 seed per lengan, uji Mann-Whitney dua-sisi berbatas KERAS
pada p = 0,10 (hanya C(6,3)=20 susunan) -- pemisahan sempurna sekalipun tak bisa mencapai
p<0,05. Perlu >=4 seed per lengan bila signifikansi hendak dilaporkan.
"""
import sys, os, time, json, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common
import marl_spklu.env.user as U
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.master_policy import (MasterPolicy, MasterPolicyEqCap,
                                         MasterPrefPolicy)
import _tahap2_jalankan as T2

# Horizon yang tersedia. Kuncinya ikut jadi bagian nama berkas keluaran.
HORIZON = {
    "30d": "scenario_dataset_klaster12_4x.json",
    "90d": "scenario_dataset_klaster12_4x_90d.json",
}

# argv[1] = daftar seed, argv[2] = horizon. Default dipertahankan persis seperti run
# pertama (seed 0, 90d) supaya hasil lama tetap dapat direproduksi tanpa argumen.
def _argv_seeds(default=(0,)):
    if len(sys.argv) > 1:
        return [int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
    return list(default)

def _argv_horizon(default="90d"):
    h = sys.argv[2] if len(sys.argv) > 2 else default
    if h not in HORIZON:
        raise SystemExit(f"horizon tak dikenal: {h} (pilih {list(HORIZON)})")
    return h

SEEDS = _argv_seeds()
SEED = SEEDS[0]          # dipakai _uji_aturan_trust_eval.py
TAG_HORIZON = _argv_horizon()
DATASET = HORIZON[TAG_HORIZON]

# Nama lama dipertahankan sbg alias supaya rujukan lama tak patah.
DATASET_90D = HORIZON["90d"]


@contextlib.contextmanager
def mode_trust(mode):
    """Alihkan zona penalti update_trust. Dibaca sbg global modul saat update_trust
    dipanggil, jadi menimpa atribut modul sudah cukup (tak perlu menambal kelas)."""
    orig = U.TRUST_PENALTY_MODE
    U.TRUST_PENALTY_MODE = mode
    try:
        yield
    finally:
        U.TRUST_PENALTY_MODE = orig


def nama_lengan(metode, mode, tag_horizon=None):
    """Stem keluaran: metode + horizon + aturan. Horizon WAJIB masuk nama -- tanpa itu
    run 30d akan menimpa checkpoint 90d yang sudah ada."""
    return f"{metode}_{tag_horizon or TAG_HORIZON}_{'abs' if mode == 'abs' else 'sgn'}"


def main():
    T2.DATASET = DATASET                  # runner membaca ini saat `jalankan` dipanggil
    ds_path = os.path.join(common.ROOT, DATASET)
    assert os.path.exists(ds_path), (
        f"dataset {TAG_HORIZON} tak ditemukan: {ds_path}\n"
        f"(dataset harus ada di root repo; cek apakah sudah ter-commit ke git)")

    jobs = []
    for mode in ("abs", "signed"):
        # Lengan lama (tetap): H-PPO = MASTER + adaptasi riwayat; P-PPO = + modul preferensi
        jobs.append((nama_lengan("hppo", mode), None, mode))
        jobs.append((nama_lengan("pppo", mode), PPPOPolicy, mode))
        # Lengan MASTER (2026-08-18): baseline tanpa encoder riwayat per-pengguna, yang
        # selama ini TIDAK ADA -- tanpanya kontribusi adaptasi riwayat tak dapat
        # diatribusikan, krn H-PPO pun sudah memuat encoder turunan PDQN.
        jobs.append((nama_lengan("master", mode), MasterPolicy, mode))
        jobs.append((nama_lengan("mastereq", mode), MasterPolicyEqCap, mode))
        jobs.append((nama_lengan("masterp", mode), MasterPrefPolicy, mode))

    t_all = time.time()
    print(f"horizon={TAG_HORIZON} ({DATASET})", flush=True)
    print(f"seed yang dijalankan: {SEEDS}", flush=True)
    for sd in SEEDS:
        for nama, pcls, mode in jobs:
            print(f"\n{'='*64}\n[{nama}] seed={sd} aturan-trust={mode} "
                  f"dataset={DATASET}\n{'='*64}", flush=True)
            t0 = time.time()
            with mode_trust(mode):
                T2.jalankan(nama, sd, policy_cls=pcls, trust_dinamis=True)
            # Aturan yang dipakai saat training HARUS ikut tercatat: checkpoint tak bisa
            # ditafsirkan tanpa mengetahui lingkungan yang melatihnya.
            mp = os.path.join(common.OUTDIR, f"t2_{nama}_seed{sd}_meta.json")
            if os.path.exists(mp):
                m = json.load(open(mp))
                m["trust_penalty_mode"] = mode
                m["dataset"] = DATASET
                m["horizon"] = TAG_HORIZON
                json.dump(m, open(mp, "w"), indent=1)
            print(f"[{nama}] seed={sd} selesai {(time.time()-t0)/60:.1f} mnt", flush=True)

    print(f"\nSEMUA TRAINING SELESAI dalam {(time.time()-t_all)/60:.1f} mnt", flush=True)


if __name__ == "__main__":
    main()
