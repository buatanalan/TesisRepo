"""Ablasi suku reward Prox -- memisahkan DUA penjelasan kegagalan modul P:

  (1) "sudah diwakili"    -- suku Prox (rewards.py::decision_reward, aktif di SEMUA
                             lengan) sudah mengajarkan "rekomendasikan yang mirip
                             dgn yang dipilih" langsung dari fitur SAAT INI. Matikan
                             Prox -> P SEHARUSNYA mulai membantu (kanal penggantinya
                             hilang).
  (2) "tak ada sisa sinyal" -- model pilihan pengguna (`Model_Simulasi_Inti.md §3.1`,
                             suku `soc_urgency`) menjadikan kepatuhan fungsi keadaan
                             SESAAT, bukan sifat historis stabil. Matikan Prox -> P
                             TETAP tak membantu, krn memang tak ada yg bisa digali
                             riwayat preferensi.

4 kondisi (2x2: dengan/tanpa P x Prox aktif/mati), reward LAIN semua sama
(RewardCalculator baku, HANYA beta_prox berubah 0.1->0.0) -- satu faktor diisolasi.

Pemakaian:
    python _uji_ablasi_prox_P.py 0,1,2 30d
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common
from marl_spklu.rl.master_station_ppo_policy import (MasterStationPPOTrainer,
                                                      MasterStationPPOPolicy,
                                                      MasterStationPPOPrefPolicy,
                                                      MasterStationPPOInferenceAgent)
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import FormulaForecaster

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
         if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
DS = os.path.join(common.ROOT, K.HORIZON[TAG])
N_UPDATES = 300
ROLLOUT_STEPS = 96
K_REC = 3
MODE = "signed"

KONDISI = [
    ("noP_proxOn", MasterStationPPOPolicy, {}, 0.1),
    ("P_proxOn", MasterStationPPOPrefPolicy, {"pref_feature_mode": True}, 0.1),
    ("noP_proxOff", MasterStationPPOPolicy, {}, 0.0),
    ("P_proxOff", MasterStationPPOPrefPolicy, {"pref_feature_mode": True}, 0.0),
]


def latih(tag, cls, kw, beta_prox, seed):
    rc = RewardCalculator(beta_prox=beta_prox)
    tr = MasterStationPPOTrainer(DS, rollout_steps=ROLLOUT_STEPS, seed=seed, verbose=False,
                                 policy_cls=cls, policy_kw=kw, reward_calc=rc)
    pol = tr.train(FormulaForecaster(), n_updates=N_UPDATES)
    ckpt = os.path.join(common.OUTDIR, f"ablasi_prox_{tag}_seed{seed}.pt")
    torch.save(pol.state_dict(), ckpt)
    return ckpt


def muat(tag, cls, kw, seed):
    ckpt = os.path.join(common.OUTDIR, f"ablasi_prox_{tag}_seed{seed}.pt")
    sim0 = common.fresh_sim(DS)
    n_spklu = len(sim0.spklus)
    pol = cls(n_spklu, **kw)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


def fac_dari_policy(policy):
    def fac(sim, _p=policy):
        return MasterStationPPOInferenceAgent(_p, sim, k=K_REC)
    return fac


def main():
    print(f"horizon={TAG} ({DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)

    # --- latih (resumable: skip bila checkpoint sudah ada) ---
    for tag, cls, kw, beta_prox in KONDISI:
        for sd in SEEDS:
            ckpt = os.path.join(common.OUTDIR, f"ablasi_prox_{tag}_seed{sd}.pt")
            if os.path.exists(ckpt):
                print(f"  [{tag}] seed={sd} -- SKIP (checkpoint ada)", flush=True)
                continue
            print(f"  [{tag}] seed={sd} -- latih...", flush=True)
            latih(tag, cls, kw, beta_prox, sd)
            print(f"  [{tag}] seed={sd} -- selesai", flush=True)

    # --- evaluasi metrik-kaya ---
    agregat = {}
    for tag, cls, kw, _ in KONDISI:
        runs = []
        for sd in SEEDS:
            pol = muat(tag, cls, kw, sd)
            fac = fac_dari_policy(pol)
            r = K.satu_run(fac, MODE, sd, beku=False)
            runs.append(r)
        agregat[tag] = K.agg(runs)
        print(f"  [{tag}] acc={agregat[tag]['acc']:.3f} gini={agregat[tag]['gini']:.3f} "
             f"wait={agregat[tag]['wait']:.1f}", flush=True)

    d_acc_on = agregat["P_proxOn"]["acc"] - agregat["noP_proxOn"]["acc"]
    d_acc_off = agregat["P_proxOff"]["acc"] - agregat["noP_proxOff"]["acc"]
    print(f"\nd_acc (Prox AKTIF)  : {d_acc_on:+.3f}", flush=True)
    print(f"d_acc (Prox MATI)   : {d_acc_off:+.3f}", flush=True)
    if d_acc_off > d_acc_on + 0.02:
        print("-> KONSISTEN dgn 'sudah diwakili': P mulai membantu setelah Prox dimatikan",
             flush=True)
    else:
        print("-> KONSISTEN dgn 'tak ada sisa sinyal': P tetap tak membantu", flush=True)

    out = dict(horizon=TAG, seeds=SEEDS, mode=MODE, agregat=agregat,
              d_acc_prox_on=d_acc_on, d_acc_prox_off=d_acc_off)
    common.save_json(out, f"uji_ablasi_prox_P_{TAG}.json")
    print(f"\nSAVED -> outputs/uji_ablasi_prox_P_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
