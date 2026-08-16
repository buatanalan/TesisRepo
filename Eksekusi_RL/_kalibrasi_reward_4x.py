"""Kalibrasi ulang bobot reward pada REZIM 4x (yang dibekukan Tahap 1).

Pemicu: `gabungan` (dipakai seluruh Tahap 2) ternyata punya varians-individual hanya 2,5%
-- kanal individualnya praktis mati karena `beta_prox` dinaikkan 4,5x padahal `prox`
struktural hampir konstan (rasio mean/std 6,53). Tahap 0 SUDAH memperingatkan hal ini.

Temuan kedua saat pengukuran: std mentah gini:flock = 1:159 pada rezim 4x. Dengan rasio
bobot preset lama (4,62:1), suku GINI -- objektif pemerataan itu sendiri -- menyumbang
~0,1% varians kanal global. Praktis tak menghasilkan gradien.

Skrip ini menghitung bobot dari std MENTAH terukur, dua opsi:
  A. jaga rasio bobot gini:flock preset lama (perubahan minimal)
  B. seimbangkan VARIANS gini vs flock di dalam kanal global (objektif pemerataan
     benar-benar hadir di gradien)

`beta_prox` DIPERTAHANKAN 0,1 (suku minor tetap) sesuai keputusan Tahap 0 -- menaikkannya
hanya menambah offset konstan, bukan sinyal.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import common
from marl_spklu.rl.rewards import RewardCalculator

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
BETA_PROX = 0.1          # suku minor tetap (keputusan Tahap 0)
TARGET_STD = 0.15        # skala target tiap kanal (mengikuti Tahap 0)


def std_mentah(seed=0):
    rc = RewardCalculator(alpha_wait=1.0, beta_prox=1.0, alpha_gini=1.0,
                          alpha_flock=1.0, use_delta_gini=True)
    d = common.reward_five_way(rc, dataset_path=DS, seed=seed, k=2)
    return {k: d[k]["std"] for k in ("prox", "gini", "flock", "improvement")}


def bobot(s, opsi, target=TARGET_STD):
    """Kembalikan dict bobot agar std kanal individual ~= std kanal global ~= target."""
    # Kanal individual dipikul `improvement` (prox sengaja minor & hampir konstan).
    alpha_wait = target / s["improvement"]
    if opsi == "A":       # jaga rasio bobot gini:flock lama (0,5785 : 0,1252 = 4,62)
        r = 0.5785 / 0.1252
        # std_global = w_f * sqrt((r*s_gini)^2 + s_flock^2)
        w_f = target / np.hypot(r * s["gini"], s["flock"])
        alpha_gini, alpha_flock = r * w_f, w_f
    elif opsi == "B":     # varians gini == varians flock di dalam kanal global
        per = target / np.sqrt(2.0)
        alpha_gini, alpha_flock = per / s["gini"], per / s["flock"]
    else:
        raise ValueError(opsi)
    return dict(alpha_wait=float(alpha_wait), beta_prox=BETA_PROX,
                alpha_gini=float(alpha_gini), alpha_flock=float(alpha_flock),
                use_delta_gini=True)


def verifikasi(kw, label, seed=0):
    d = common.reward_five_way(RewardCalculator(**kw), dataset_path=DS, seed=seed, k=2)
    sp, sg = d["prox"]["std"], d["gini"]["std"]
    sf, si = d["flock"]["std"], d["improvement"]["std"]
    var_ind = sp**2 + si**2
    var_glo = sg**2 + sf**2
    frac = var_ind / max(var_ind + var_glo, 1e-12)
    print(f"  {label}")
    print(f"    prox std={sp:.4f}  improvement std={si:.4f}  -> kanal individual {np.sqrt(var_ind):.4f}")
    print(f"    gini std={sg:.4f}  flock std={sf:.4f}       -> kanal global     {np.sqrt(var_glo):.4f}")
    print(f"    varians individual = {frac*100:.1f}%   |  kontribusi gini dlm global = "
          f"{sg**2/max(var_glo,1e-12)*100:.1f}%")
    return dict(kw=kw, frac_individual=float(frac),
                frac_gini_dlm_global=float(sg**2 / max(var_glo, 1e-12)))


def main():
    s = std_mentah()
    print("std MENTAH rezim 4x:", {k: round(v, 4) for k, v in s.items()}, "\n")

    hasil = {}
    print("=== PEMBANDING (konfigurasi yang sedang dipakai) ===")
    hasil["gabungan_lama"] = verifikasi(
        dict(alpha_wait=0.0, beta_prox=0.4467, alpha_gini=0.3672,
             alpha_flock=0.0795, use_delta_gini=True), "gabungan (Tahap 2 sekarang)")
    hasil["seimbang_lama"] = verifikasi(
        dict(alpha_wait=0.0897, beta_prox=0.1, alpha_gini=0.5785,
             alpha_flock=0.1252, use_delta_gini=True), "seimbang (Tahap 0)")

    print("\n=== KANDIDAT BARU (dikalibrasi pada rezim 4x) ===")
    for opsi in ("A", "B"):
        kw = bobot(s, opsi)
        print(f"  [Opsi {opsi}] bobot: alpha_wait={kw['alpha_wait']:.4f} "
              f"beta_prox={kw['beta_prox']:.3f} alpha_gini={kw['alpha_gini']:.4f} "
              f"alpha_flock={kw['alpha_flock']:.4f}")
        hasil[f"opsi_{opsi}"] = verifikasi(kw, f"seimbang4x_{opsi}")

    common.save_json(dict(std_mentah=s, hasil=hasil), "kalibrasi_reward_4x.json")
    print("\nSAVED -> kalibrasi_reward_4x.json")


if __name__ == "__main__":
    main()
