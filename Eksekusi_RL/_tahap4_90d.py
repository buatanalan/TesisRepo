"""Tahap 4 -- horizon 90 HARI, trust DINAMIS, faktorial 2x2 (metode x aturan-trust).

Motivasi: pada horizon 30 hari trust menurun LINEAR tanpa melandai (uji_erosi_trust_90d),
sehingga titik akhirnya bergantung horizon dan bukan sifat kebijakan. 90 hari dipakai agar
lintasan trust sempat mendekati rezim tunaknya.

Faktor 2 (aturan trust) WAJIB disertakan: tanpa lengan "abs" di 90 hari, tiap perbedaan
thd hasil 30 hari tak dapat diatribusikan -- bisa krn horizon, bisa krn aturan.

  abs    -- aturan berlaku sekarang (Pemodelan_Variasi_Distribusi.md 7.2)
  signed -- hanya keterlambatan (actual > est) menghukum; over-estimasi netral

1 seed (=0): ini uji EKSPLORATIF horizon, bukan klaim inferensial. Klaim per-metode tetap
bersandar pada Tahap 3 (3 seed). Jangan laporkan p-value dari lengan ini.
"""
import sys, os, time, json, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common
import marl_spklu.env.user as U
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
import _tahap2_jalankan as T2

DATASET_90D = "scenario_dataset_klaster12_4x_90d.json"
SEED = 0


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


def main():
    T2.DATASET = DATASET_90D              # runner membaca ini saat `jalankan` dipanggil
    ds_path = os.path.join(common.ROOT, DATASET_90D)
    assert os.path.exists(ds_path), f"dataset 90d tak ditemukan: {ds_path}"

    jobs = []
    for mode in ("abs", "signed"):
        tag = "abs" if mode == "abs" else "sgn"
        jobs.append((f"hppo_90d_{tag}", None, mode))
        jobs.append((f"pppo_90d_{tag}", PPPOPolicy, mode))

    t_all = time.time()
    for nama, pcls, mode in jobs:
        print(f"\n{'='*64}\n[{nama}] aturan-trust={mode} dataset={DATASET_90D}\n{'='*64}",
              flush=True)
        t0 = time.time()
        with mode_trust(mode):
            T2.jalankan(nama, SEED, policy_cls=pcls, trust_dinamis=True)
        # Aturan yang dipakai saat training HARUS ikut tercatat: checkpoint tak bisa
        # ditafsirkan tanpa mengetahui lingkungan yang melatihnya.
        mp = os.path.join(common.OUTDIR, f"t2_{nama}_seed{SEED}_meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            m["trust_penalty_mode"] = mode
            m["dataset"] = DATASET_90D
            json.dump(m, open(mp, "w"), indent=1)
        print(f"[{nama}] selesai {(time.time()-t0)/60:.1f} mnt", flush=True)

    print(f"\nSEMUA TRAINING SELESAI dalam {(time.time()-t_all)/60:.1f} mnt", flush=True)


if __name__ == "__main__":
    main()
