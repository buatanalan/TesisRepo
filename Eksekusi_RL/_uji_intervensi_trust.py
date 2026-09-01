"""Intervensi terkontrol pada trust (Uji B, 2026-09-01) -- menjawab kritik: "jika trust
pengguna DI-RESET ke nilai lain di tengah simulasi, apakah perilaku rekomendasi berubah
SESUAI PREDIKSI mekanisme P=(1-T)*P_pref + T*P_rec?"

Prediksi mekanisme, dinyatakan SEBELUM melihat hasil (falsifiable):
  - Pengguna yang trust-nya DIPAKSA TURUN -> bobot P_rec mengecil -> kepatuhan (`patuh`)
    dan `geser_p` (pergeseran probabilitas akibat rekomendasi) TURUN pasca-intervensi,
    relatif thd sebelum intervensi DAN relatif thd kelompok kontrol yg tak diintervensi.
  - Pengguna yang trust-nya DIPAKSA NAIK -> sebaliknya, NAIK.
  - Kelompok KONTROL (tak diintervensi, ukuran sama, dipilih acak bersamaan) dipakai
    membedakan efek intervensi dari tren alami sepanjang simulasi (mis. erosi trust
    populasi umum, §5.3.2 draf Bab V).

Mekanisme intervensi: menyetel LANGSUNG `trust_alpha`/`trust_beta` pengguna terpilih pd
STEP tertentu (pola sama `ablations.initial_trust`, tapi diterapkan MID-SIMULASI pd
objek User yg sudah berjalan, bukan saat __init__). `update_trust()` TETAP aktif
sesudahnya (trust dinamis, bukan dibekukan) -- yg diuji adalah RESPONS thd titik
berangkat baru, bukan efek pembekuan (`constant_trust_shadow` sudah menjawab itu).

Dijalankan dgn men-drive `sim.step_once()` manual (bukan `sim.run()`) supaya intervensi
bisa disisipkan tepat di STEP yg diinginkan -- lihat `Simulator.run()`:
    for step in range(max_steps): self.step_once(step, agent=agent)

Pemakaian (server):
    python _uji_intervensi_trust.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
import re as _re
from marl_spklu.rl.master_pure_hybrid_policy import MasterHybridPPOActor
from marl_spklu.rl.master_pure_hybrid_trainer import MasterHybridPPOInferenceAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.training import _fresh_sim as _fresh_sim_common

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "90d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3"
K_REC = 3
HARI_INTERVENSI = 45          # titik tengah horizon 90 hari -- cukup riwayat SEBELUM utk
                              # baseline pra-intervensi, cukup sisa SESUDAH utk observasi
N_PER_KELOMPOK = 200          # ukuran tiap kelompok (turun/naik/kontrol)
TRUST_RENDAH, TRUST_TINGGI = 0.1, 0.9
JENDELA_HARI = 15             # bandingkan H±JENDELA_HARI thd hari intervensi

HORIZON = {"30d": "scenario_dataset_klaster12_4x.json",
           "90d": "scenario_dataset_klaster12_4x_90d.json",
           "90d6x": "scenario_dataset_klaster12_6x_90d.json"}
DS = os.path.join(common.ROOT, HORIZON[TAG])

_m_histk = _re.search(r"_histK(\d+)", TAG_ARM)
ACTOR_KW = dict(vec_dim=8, bid_hidden=16, pref_d_lstm=8, pref_d_attn=8, station_attn_dim=8,
                pref_feature_mode="_preffeat" in TAG_ARM,
                pref_pair_outcome="_pairout" in TAG_ARM,
                use_station_attn="_noattn" not in TAG_ARM,
                pref_hist_k=(int(_m_histk.group(1)) if _m_histk else None),
                station_feat_dim=(10 if "_evobs" in TAG_ARM else 7))


def _checkpoint_tersedia():
    ada = []
    i = 0
    while os.path.exists(os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{i}.pt")):
        ada.append(i); i += 1
    return ada


_CKPT = _checkpoint_tersedia()
assert _CKPT, f"tak ada checkpoint utk {TAG_ARM}"


def satu_run(seed):
    ckpt_idx = _CKPT[seed % len(_CKPT)]
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{ckpt_idx}.pt")
    n_spklu = len(_fresh_sim_common(DS).spklus)
    pol = MasterHybridPPOActor(n_spklu, **ACTOR_KW)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()

    sim = common.fresh_sim(DS, rekam_deret=False)
    random.seed(seed); np.random.seed(seed)
    agent = MasterHybridPPOInferenceAgent(pol, forecaster=FormulaForecaster(), k=K_REC)
    agent.bind_to_sim(sim)

    step_intervensi = int(HARI_INTERVENSI * 1440 / sim.dt_minutes)
    kelompok = {"turun": [], "naik": [], "kontrol": []}
    _terjadwal = False

    for step in range(sim.max_steps):
        if step == step_intervensi and not _terjadwal:
            # Pengguna yg SUDAH aktif (pernah spawn >=1x) & belum DONE -- supaya efeknya
            # bisa diamati pd keputusan berikutnya. `sim.users` berisi seluruh populasi;
            # saring yg trust_alpha+trust_beta > 2 (>=1 pembaruan trust pernah terjadi,
            # proksi "pernah aktif" tanpa perlu menelusuri decision_log).
            aktif = [u for u in sim.users if (u.trust_alpha + u.trust_beta) > 2.0]
            random.Random(1000 + seed).shuffle(aktif)
            n = min(N_PER_KELOMPOK, len(aktif) // 3)
            kelompok["turun"] = aktif[:n]
            kelompok["naik"] = aktif[n:2 * n]
            kelompok["kontrol"] = aktif[2 * n:3 * n]
            for u in kelompok["turun"]:
                u.trust_alpha, u.trust_beta = TRUST_RENDAH, 1.0 - TRUST_RENDAH
            for u in kelompok["naik"]:
                u.trust_alpha, u.trust_beta = TRUST_TINGGI, 1.0 - TRUST_TINGGI
            # kontrol: TIDAK disentuh -- trust asli dibiarkan berjalan
            print(f"  [seed={seed}] intervensi @ hari {HARI_INTERVENSI} (step {step}): "
                 f"turun={len(kelompok['turun'])} naik={len(kelompok['naik'])} "
                 f"kontrol={len(kelompok['kontrol'])} dari {len(aktif)} pengguna aktif",
                 flush=True)
            _terjadwal = True
        sim.step_once(step, agent=agent)

    return sim, {k: [u.user_id for u in v] for k, v in kelompok.items()}, step_intervensi


def _ringkas(decision_log, uid_set, step_intervensi, jendela_step):
    """`patuh` & `geser_p` rerata pd jendela SEBELUM vs SESUDAH intervensi, utk
    subset user_id tertentu."""
    pra = [e for e in decision_log if e["user_id"] in uid_set
          and step_intervensi - jendela_step <= e["step"] < step_intervensi]
    pasca = [e for e in decision_log if e["user_id"] in uid_set
            and step_intervensi <= e["step"] < step_intervensi + jendela_step]
    def agg(rows):
        if not rows:
            return dict(n=0, patuh=float("nan"), geser_p=float("nan"))
        patuh = np.mean([r["patuh"] for r in rows])
        gp = [r.get("geser_p") for r in rows if r.get("geser_p") is not None
             and not (isinstance(r.get("geser_p"), float) and np.isnan(r["geser_p"]))]
        return dict(n=len(rows), patuh=float(patuh),
                   geser_p=float(np.mean(gp)) if gp else float("nan"))
    return agg(pra), agg(pasca)


def main():
    print(f"lengan={TAG_ARM}  horizon={TAG} ({DS})", flush=True)
    print(f"hari intervensi={HARI_INTERVENSI}  n/kelompok={N_PER_KELOMPOK}  "
         f"trust turun->{TRUST_RENDAH} naik->{TRUST_TINGGI}\n", flush=True)

    hasil = {"turun": [], "naik": [], "kontrol": []}
    for sd in SEEDS:
        sim, kelompok, step_iv = satu_run(sd)
        jendela_step = int(JENDELA_HARI * 1440 / sim.dt_minutes)
        for nm, uids in kelompok.items():
            pra, pasca = _ringkas(sim.decision_log, set(uids), step_iv, jendela_step)
            hasil[nm].append((pra, pasca))
            print(f"  [seed={sd}] {nm:8s} PRA  n={pra['n']:4d} patuh={pra['patuh']:.3f} "
                 f"geser_p={pra['geser_p']:+.3f}", flush=True)
            print(f"  [seed={sd}] {nm:8s} PASCA n={pasca['n']:4d} patuh={pasca['patuh']:.3f} "
                 f"geser_p={pasca['geser_p']:+.3f}", flush=True)

    print("\n" + "=" * 70)
    print("RINGKASAN (rerata lintas seed, delta = PASCA - PRA)")
    print("=" * 70)
    print(f"{'kelompok':10s}{'d_patuh':>10s}{'d_geser_p':>12s}  {'prediksi'}")
    prediksi = {"turun": "keduanya TURUN", "naik": "keduanya NAIK",
               "kontrol": "keduanya ~0 (tren alami)"}
    ringkas = {}
    for nm, pasangan in hasil.items():
        d_patuh = np.mean([p["patuh"] - a["patuh"] for a, p in pasangan])
        d_gp = np.mean([p["geser_p"] - a["geser_p"] for a, p in pasangan])
        ringkas[nm] = (d_patuh, d_gp)
        print(f"{nm:10s}{d_patuh:+10.3f}{d_gp:+12.3f}  {prediksi[nm]}")

    print()
    cocok = (ringkas["turun"][0] < 0 and ringkas["naik"][0] > 0
            and ringkas["turun"][0] < ringkas["kontrol"][0] < ringkas["naik"][0])
    print(f"Prediksi mekanisme {'TERKONFIRMASI' if cocok else 'TIDAK terkonfirmasi'} "
         f"pd arah delta kepatuhan (turun < kontrol < naik).")

    out = os.path.join(common.OUTDIR, f"intervensi_trust_{TAG_ARM}.json")
    common.save_json({"hasil_per_seed": {k: [(a, p) for a, p in v]
                                        for k, v in hasil.items()},
                      "ringkasan": {k: {"d_patuh": v[0], "d_geser_p": v[1]}
                                   for k, v in ringkas.items()},
                      "hari_intervensi": HARI_INTERVENSI,
                      "trust_rendah": TRUST_RENDAH, "trust_tinggi": TRUST_TINGGI},
                     f"intervensi_trust_{TAG_ARM}.json")
    print(f"\nSAVED -> outputs/intervensi_trust_{TAG_ARM}.json")


if __name__ == "__main__":
    main()
