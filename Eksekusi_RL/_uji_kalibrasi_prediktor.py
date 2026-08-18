"""E0 -- Apakah erosi trust dapat dihentikan dengan MENGKALIBRASI ULANG prediktor?

HIPOTESIS (biner, dapat jatuh ke dua arah):
  Kesenjangan janji-pengalaman dapat diurai menjadi BIAS (komponen sistematis, dapat
  dihapus dgn menggeser prediktor sejauh konstanta c) dan RAGAM (sebaran sisa, tak
  tersentuh pergeseran).
    - Bila didominasi BIAS   -> ada c != 0 yang menurunkan rasio Sum(beta)/Sum(alpha).
    - Bila didominasi RAGAM  -> tiap c != 0 justru mendorong prediksi yang tadinya TEPAT
                                keluar dari zona bukti positif, sehingga Sum(alpha) turun
                                lebih cepat daripada Sum(beta), dan argmin_c = 0.

Uji: sapu c pada rentang lebar, cari argmin rasio. Hasilnya menentukan apakah seluruh
kelas solusi "perbaiki prediktornya" terbuka atau tertutup.

CATATAN REPRODUKTIBILITAS: versi pertama analisis ini dijalankan dari direktori sementara
dan tidak pernah disimpan -- temuannya sempat tidak dapat diverifikasi. Skrip ini
menggantikannya, memakai checkpoint 200-iterasi yang berlaku.

    python _uji_kalibrasi_prediktor.py            # seed 0,1,2 (default)
    python _uji_kalibrasi_prediktor.py 0,1,2,3,4
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.rollout import InferenceAgent
from marl_spklu.env.user import (DELTAW_TOL_LOW as LO, DELTAW_TOL_HIGH as HI,
                                 TRUST_EPS_ALPHA as EA, TRUST_EPS_BETA as EB)

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
         if len(sys.argv) > 1 else [0, 1, 2])
SAPUAN = np.arange(-90.0, 30.5, 0.5)      # menit; cukup lebar utk memuat ekor kedua arah


class VW(ForecasterBase):
    def predict(self, sp, t=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {k: 0.0 for k in sp}
        return {k: float(sim.compute_virtual_wait(user, v, t)) for k, v in sp.items()}


def pol(stem, seed):
    ck = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}.pt")
    mp = os.path.join(common.OUTDIR, f"t2_{stem}_seed{seed}_meta.json")
    if not (os.path.exists(ck) and os.path.exists(mp)):
        return None
    m = json.load(open(mp))
    c = PPPOPolicy if m.get("policy_cls") == "PPPOPolicy" else HPPOPolicy
    kw = dict(n_critics=m.get("n_critics", 1))
    if c is PPPOPolicy:
        kw.update(pref_d_lstm=m.get("pref_d_lstm", 64), pref_d_attn=m.get("pref_d_attn", 64))
    p = c(m["obs_dim"], m["critic_obs_dim"], m["N"], **kw)
    p.load_state_dict(torch.load(ck)); p.eval()
    return lambda sim, pp=p: InferenceAgent(pp, sim, VW(), k=2, epsilon=0.0, threshold=0.20)


def galat(fac, seed):
    """Kumpulkan galat BERTANDA (actual - est) untuk trip yang MEMATUHI rekomendasi --
    syarat yang sama dgn User.update_trust."""
    sim = common.fresh_sim(DS)
    random.seed(seed); np.random.seed(seed)
    sim.run(max_steps=sim.max_steps, agent=fac(sim))
    return np.array([l["wait_time"] - l["est_wait"] for l in sim.logs if l.get("complied")],
                    float)


def rasio(d):
    """Sum(beta)/Sum(alpha) di bawah aturan trust yang berlaku (zona penalti ABSOLUT).
    Meniru User.update_trust persis: reward bila |d| <= LO, penalti bila |d| >= HI."""
    a = np.abs(d)
    rew = a <= LO
    pen = a >= HI
    sa = float(np.sum(EA * (1.0 - a[rew] / LO)))
    sb = float(np.sum(EB * (a[pen] / HI)))
    return sb / max(sa, 1e-9), sa, sb, float(rew.mean()), float(pen.mean())


def main():
    ARMS = [("greedy_util", lambda sim: GreedyAgent(mode="utilization", top_k=2), None),
            ("H-PPO", None, "hppo_30d_abs"),
            ("P-PPO", None, "pppo_30d_abs")]

    out = {"seeds": SEEDS, "sapuan": [float(SAPUAN[0]), float(SAPUAN[-1])], "lengan": {}}
    print(f"seed: {SEEDS}   sapuan c: {SAPUAN[0]:+.0f}..{SAPUAN[-1]:+.0f} menit\n")
    print("%-12s %8s %9s %9s %9s %9s %9s" % (
        "lengan", "n", "c*", "rasio@c*", "rasio@0", "perbaikan", "%tepat"))
    print("-" * 70)

    for lbl, fac, stem in ARMS:
        DS_ALL = []
        for sd in SEEDS:
            f = fac if stem is None else pol(stem, sd)
            if f is None:
                continue
            DS_ALL.append(galat(f, sd))
        if not DS_ALL:
            print("%-12s (checkpoint belum ada)" % lbl); continue
        d = np.concatenate(DS_ALL)

        kurva = [rasio(d - c)[0] for c in SAPUAN]
        i = int(np.argmin(kurva))
        c_star = float(SAPUAN[i])
        r0, sa0, sb0, tepat0, hukum0 = rasio(d)
        r_best = float(kurva[i])

        dd = d - np.median(d)
        rec = dict(
            n=int(d.size), c_star=c_star, rasio_c_star=r_best, rasio_c0=float(r0),
            perbaikan=float(r0 - r_best), sum_alpha=sa0, sum_beta=sb0,
            frac_tepat=tepat0, frac_hukum=hukum0,
            median_galat=float(np.median(d)), std_galat=float(d.std()),
            iqr_galat=float(np.percentile(d, 75) - np.percentile(d, 25)),
            p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)),
            frac_telat=float(np.mean(d >= HI)), frac_cepat=float(np.mean(-d >= HI)),
            kurva_c=[float(x) for x in SAPUAN], kurva_rasio=[float(x) for x in kurva])
        out["lengan"][lbl] = rec

        print("%-12s %8d %+9.1f %9.3f %9.3f %+9.3f %8.1f%%" % (
            lbl, d.size, c_star, r_best, r0, r_best - r0, 100 * tepat0))

    print()
    print("=== bentuk sebaran galat (menentukan bias vs ragam) ===")
    print("%-12s %9s %9s %9s %9s %9s" % ("lengan", "median", "IQR", "sd", "p10", "p90"))
    print("-" * 62)
    for lbl, r in out["lengan"].items():
        print("%-12s %9.2f %9.2f %9.2f %9.2f %9.2f" % (
            lbl, r["median_galat"], r["iqr_galat"], r["std_galat"], r["p10"], r["p90"]))

    print()
    print("=== arah ekor (dasar argumen aturan trust asimetris) ===")
    print("%-12s %12s %12s" % ("lengan", "%terlambat", "%terlalu cepat"))
    print("-" * 38)
    for lbl, r in out["lengan"].items():
        print("%-12s %11.1f%% %11.1f%%" % (lbl, 100 * r["frac_telat"], 100 * r["frac_cepat"]))

    # Kesimpulan HANYA sah bila seluruh lengan yang direncanakan benar-benar berjalan.
    # (Versi pertama menarik kesimpulan "semua lengan" padahal 2 dari 3 dilewati karena
    # checkpoint-nya tak ada -- `all()` atas himpunan kosong/parsial bernilai True.)
    diharapkan = {lbl for lbl, _, _ in ARMS}
    hilang = sorted(diharapkan - set(out["lengan"]))
    if hilang:
        out["kesimpulan"] = (f"TIDAK LENGKAP -- lengan tak terevaluasi: {hilang}. "
                             "Checkpoint .pt tidak tersedia; kesimpulan ditahan.")
    else:
        semua_nol = all(abs(r["c_star"]) < 1e-9 for r in out["lengan"].values())
        out["kesimpulan"] = (
            f"c*=0 pada {len(out['lengan'])}/{len(diharapkan)} lengan -- galat didominasi "
            "RAGAM, kalibrasi ulang prediktor TIDAK dapat menghentikan erosi"
            if semua_nol else
            "ditemukan c* != 0 -- galat memuat BIAS yang dapat dikoreksi")
    out["lengan_hilang"] = hilang
    print()
    print("KESIMPULAN:", out["kesimpulan"])
    common.save_json(out, "uji_kalibrasi_prediktor.json")
    print("SAVED -> outputs/uji_kalibrasi_prediktor.json")


if __name__ == "__main__":
    main()
