"""Sapuan tingkat KEPATUHAN (trust dibekukan pada beberapa nilai, meniru sumbu `p_check`
PDQN §V-B) -- menguji apakah manfaat modul preferensi TERKONSENTRASI pada rezim
ketidakpatuhan MENENGAH (bukti PDQN Gbr. 10b), bukan seragam di semua rezim.

LATAR
-----
Tiga pengujian P sebelumnya (MASTER, MASTER-bid, MasterStationPPO x2 representasi)
semua dijalankan pada trust DINAMIS ALAMI lingkungan (berkembang ~0,41-0,49) -- rezim
kepatuhan TINGGI. Tak satu pun menunjukkan manfaat acceptance. PDQN sendiri MEMBUKTIKAN
(bukan berasumsi) manfaat preferensi tidak seragam: terbesar pada `p_check` menengah,
kecil/hilang pada kepatuhan tinggi. Skrip ini menguji apakah pola yang sama muncul di
sini -- dgn `constant_trust_shadow(value)` yg MEMBEKUKAN trust efektif yg dipakai
keputusan (User.trust_effective) ke nilai tertentu, PERSIS peran `p_check` di PDQN, sambil
`User.trust` asli (dari trust_alpha/trust_beta) tetap berkembang sbg DIAGNOSTIK murni
(tak memengaruhi keputusan) -- lihat `marl_spklu/experiments/ablations.py`.

TIDAK BUTUH LATIH ULANG -- checkpoint sudah ada (`master_station_ppo_seed*.pt`,
`master_station_ppo_pref_feat_seed*.pt`). Murni evaluasi ulang pada trust yang dibekukan
berbeda-beda.

Pemakaian:
    python _uji_sapuan_kepatuhan_P.py 0,1,2 30d
    python _uji_sapuan_kepatuhan_P.py 0,1,2 30d 0.2,0.35,0.5,0.65,0.8   # sapuan kustom
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_station_ppo_policy import (MasterStationPPOPolicy,
                                                      MasterStationPPOPrefPolicy,
                                                      MasterStationPPOInferenceAgent)

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
         if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
SAPUAN = ([float(s) for s in sys.argv[3].replace(" ", "").split(",") if s]
         if len(sys.argv) > 3 else [0.2, 0.35, 0.5, 0.65, 0.8])
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])
K_REC = 3
MODE = "signed"   # aturan trust baku hasil Tahap 4 -- tak relevan langsung (trust
                   # dibekukan), tapi tetap harus konsisten dgn seluruh pipeline lain


def muat(tag, cls, seed, **kw):
    ckpt = os.path.join(common.OUTDIR, f"{tag}_seed{seed}.pt")
    assert os.path.exists(ckpt), f"checkpoint tak ditemukan: {ckpt}"
    sim0 = common.fresh_sim(K.DS)
    n_spklu = len(sim0.spklus)
    policy = cls(n_spklu, **kw)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()
    return policy


def fac_dari_policy(policy):
    def fac(sim, _policy=policy):
        return MasterStationPPOInferenceAgent(_policy, sim, k=K_REC)
    return fac


def satu_run_pada_trust(fac, seed, value):
    """Reuse K.satu_run APA ADANYA -- timpa TRUST_BEKU modul sementara ke `value`,
    `beku=True` memaksanya memakai constant_trust_shadow(value=TRUST_BEKU)."""
    lama = K.TRUST_BEKU
    K.TRUST_BEKU = float(value)
    try:
        return K.satu_run(fac, MODE, seed, beku=True)
    finally:
        K.TRUST_BEKU = lama


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)
    print(f"sapuan trust (meniru p_check PDQN): {SAPUAN}", flush=True)

    hasil = {"noP": {}, "P": {}}
    for value in SAPUAN:
        for label, tag, cls, kw in [
                ("noP", "master_station_ppo", MasterStationPPOPolicy, {}),
                ("P", "master_station_ppo_pref_feat", MasterStationPPOPrefPolicy,
                 {"pref_feature_mode": True})]:
            runs = []
            for sd in SEEDS:
                policy = muat(tag, cls, sd, **kw)
                fac = fac_dari_policy(policy)
                r = satu_run_pada_trust(fac, sd, value)
                runs.append(r)
            agg = K.agg(runs)
            hasil[label][value] = agg
            print(f"  trust={value:.2f} [{label:4s}] acc={agg['acc']:.3f} "
                 f"wait={agg['wait']:6.1f} gini={agg['gini']:.3f}", flush=True)

    print(f"\n{'trust':>7s}{'acc_noP':>9s}{'acc_P':>9s}{'d_acc':>8s}"
         f"{'wait_noP':>10s}{'wait_P':>9s}{'d_wait':>8s}"
         f"{'gini_noP':>10s}{'gini_P':>9s}{'d_gini':>8s}", flush=True)
    for value in SAPUAN:
        a, p = hasil["noP"][value], hasil["P"][value]
        print(f"{value:7.2f}{a['acc']:9.3f}{p['acc']:9.3f}{p['acc']-a['acc']:+8.3f}"
             f"{a['wait']:10.1f}{p['wait']:9.1f}{p['wait']-a['wait']:+8.1f}"
             f"{a['gini']:10.3f}{p['gini']:9.3f}{p['gini']-a['gini']:+8.3f}", flush=True)

    out = dict(horizon=TAG, seeds=SEEDS, sapuan=SAPUAN, mode=MODE,
              noP={str(v): hasil["noP"][v] for v in SAPUAN},
              P={str(v): hasil["P"][v] for v in SAPUAN})
    common.save_json(out, f"uji_sapuan_kepatuhan_P_{TAG}.json")
    print(f"\nSAVED -> outputs/uji_sapuan_kepatuhan_P_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
