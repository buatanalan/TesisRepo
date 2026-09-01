"""Ekstraksi bobot attention antar-stasiun (Uji A', 2026-09-01) -- menjawab kritik:
klaim Bab IV bahwa attention membuat representasi tiap kandidat "melihat" kandidat lain
yang hadir pada permintaan yang sama, sehingga agen sadar akan persaingan dan mengurangi
*flocking*, belum diuji langsung pada MEKANISMENYA (bobot attention itu sendiri) --
hanya pada performa agregat (Gini, acc, wait) dan proksi tak-langsung (herding_index).

Tiga sinyal yang diuji, sesuai kritik:
  Sinyal 1 -- korelasi bobot attention[i->j] dgn utilisasi kandidat j: NEGATIF berarti
             agen "mengalah" dari kandidat yang sudah ramai (bobot mengecil).
  Sinyal 2 -- collision rate: proporsi rekomendasi yang berbagi stasiun tujuan dgn
             rekomendasi LAIN pada langkah yang sama, dibanding lengan tanpa attention.
             TIDAK butuh checkpoint (dihitung dari `rec_distribution_log`, tersedia utk
             SEMUA agen termasuk greedy) -- dijalankan terpisah dari ekstraksi bobot.
  Sinyal 3 -- rasio varians ANTAR-komposisi vs DALAM-komposisi: bobot yang benar2
             peka konteks berubah bermakna ketika himpunan kandidat feasible berubah,
             bukan konstan per-stasiun terlepas siapa pesaingnya.

Mekanisme sama dgn `_ekstrak_representasi_laten.py`: monkeypatch `forward()` milik
kelas modul (di sini `SmallStationAttention`), tangkap tensor `weights` (softmax
attention, sudah dihitung eksplisit di baris scores->softmax), gabungkan dgn identitas
stasiun (`self.sids`, urutan TETAP -- lihat `MasterHybridPPORolloutAgent.get_recommendation`,
kolom/baris attention tak pernah diacak ulang per keputusan) dan utilisasi tiap kandidat
SAAT keputusan itu (dari `sim.spklus[sid].get_utilization()`).

CATATAN: hanya lengan dgn `use_station_attn=True` yang punya sesuatu utk ditangkap --
MASTER (`_noattn`) MELEWATI modul ini sepenuhnya (`forward()` tak pernah dipanggil).
Jalankan pada P-MASTER (`_pg0.1_pure3`) dan/atau MASTER+Atensi (`_pure3`, tanpa "noattn")
utk Sinyal 1 & 3. Sinyal 2 (collision rate) dihitung utk SEMUA lengan tanpa kecuali,
termasuk yg tak punya attention -- itulah pembandingnya.

Pemakaian (server):
    # Sinyal 1 & 3 -- perlu checkpoint attention aktif
    python _ekstrak_attensi_stasiun.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3
    python _ekstrak_attensi_stasiun.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pure3

    # Sinyal 2 -- collision rate, jalan utk SEMUA lengan (RL maupun greedy)
    python _ekstrak_attensi_stasiun.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3 --collision-saja
    python _ekstrak_attensi_stasiun.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3 --collision-saja
    python _ekstrak_attensi_stasiun.py 0,1,2 90d greedy --collision-saja
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common
import re as _re
from marl_spklu.rl.master_pure_hybrid_policy import MasterHybridPPOActor, SmallStationAttention
from marl_spklu.rl.master_pure_hybrid_trainer import MasterHybridPPOInferenceAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.training import _fresh_sim as _fresh_sim_common

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "90d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3"
COLLISION_SAJA = "--collision-saja" in sys.argv
K_REC = 3

HORIZON = {"30d": "scenario_dataset_klaster12_4x.json",
          "90d": "scenario_dataset_klaster12_4x_90d.json",
          "90d6x": "scenario_dataset_klaster12_6x_90d.json"}
DS = os.path.join(common.ROOT, HORIZON[TAG])

_ADALAH_GREEDY = TAG_ARM == "greedy"
if _ADALAH_GREEDY:
    # Isolasi sys.argv sebelum impor -- modul itu mem-parsing argv-nya SENDIRI di level
    # modul (pola sama proteksi `_uji_konsolidasi` di dalam modul itu sendiri); argv
    # skrip ini (SEEDS/TAG/"greedy"/--collision-saja) TIDAK cocok layout yang diharapkan.
    _argv_asli_G = sys.argv
    sys.argv = ["_uji_greedy_setara_metrik.py", "0", "30d", "3"]
    import _uji_greedy_setara_metrik as G
    sys.argv = _argv_asli_G
if not _ADALAH_GREEDY:
    _m_histk = _re.search(r"_histK(\d+)", TAG_ARM)
    ACTOR_KW = dict(vec_dim=8, bid_hidden=16, pref_d_lstm=8, pref_d_attn=8, station_attn_dim=8,
                    pref_feature_mode="_preffeat" in TAG_ARM,
                    pref_pair_outcome="_pairout" in TAG_ARM,
                    use_station_attn="_noattn" not in TAG_ARM,
                    pref_hist_k=(int(_m_histk.group(1)) if _m_histk else None),
                    station_feat_dim=(10 if "_evobs" in TAG_ARM else 7))
    PUNYA_ATENSI = ACTOR_KW["use_station_attn"]
else:
    PUNYA_ATENSI = False


def _checkpoint_tersedia():
    if _ADALAH_GREEDY:
        return [0]  # dummy -- greedy tak punya checkpoint, tapi tetap "1 lengan"
    ada = []
    i = 0
    while os.path.exists(os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{i}.pt")):
        ada.append(i); i += 1
    return ada


_CKPT = _checkpoint_tersedia()
assert _CKPT, f"tak ada checkpoint utk {TAG_ARM}"


# --------------------------------------------------------------- penangkap attensi
class _PenangkapAtensi:
    """Sama pola `_ekstrak_representasi_laten.py::_Penangkap`, tapi menangkap
    `SmallStationAttention.forward` -- perlu menghitung ULANG `weights` di dalam
    patch (kode aslinya tak mengembalikan `weights`, hanya `vec + gate*attended`)
    supaya perilaku forward TAK BERUBAH (harus identik bit-demi-bit dgn evaluasi
    normal), sekaligus merekamnya."""

    def __init__(self):
        self.sim = None
        self.rows_w, self.rows_mask, self.rows_util, self.rows_uid, self.rows_step = \
            [], [], [], [], []
        self._asli = SmallStationAttention.forward

    def pasang(self):
        _cap = self

        def _forward_tertangkap(diri, vec, mask):
            q, k, v = diri.q(vec), diri.k(vec), diri.v(vec)
            scores = torch.einsum("bnd,bmd->bnm", q, k) / (diri.d_attn ** 0.5)
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            attended = diri.out(torch.einsum("bnm,bmd->bnd", weights, v))
            gate = torch.sigmoid(diri.gate_raw)
            out = vec + gate * attended
            if _cap.sim is not None:
                u = getattr(_cap.sim, "_current_spawn_user", None)
                if u is not None:
                    sids = getattr(_cap, "sids", None)
                    util = ([_cap.sim.spklus[s].get_utilization() for s in sids]
                           if sids else [])
                    _cap.rows_w.append(weights.detach().cpu().numpy()[0])   # (N,N)
                    _cap.rows_mask.append(mask.detach().cpu().numpy()[0])   # (N,)
                    _cap.rows_util.append(np.array(util, dtype=np.float32))
                    _cap.rows_uid.append(u.user_id)
                    _cap.rows_step.append(int(_cap.sim.current_step))
            return out

        SmallStationAttention.forward = _forward_tertangkap

    def lepas(self):
        SmallStationAttention.forward = self._asli


def satu_run(seed):
    n_spklu = len(_fresh_sim_common(DS).spklus)
    cap = _PenangkapAtensi() if (PUNYA_ATENSI and not COLLISION_SAJA) else None

    if _ADALAH_GREEDY:
        fac = lambda sim: G.GreedySetara(mode="queue", top_k=K_REC,
                                        wait_predictor=G.AdapterFormula())
        sids = None
    else:
        ckpt_idx = _CKPT[seed % len(_CKPT)]
        ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{ckpt_idx}.pt")
        pol = MasterHybridPPOActor(n_spklu, **ACTOR_KW)
        pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
        pol.eval()

        def fac(sim, _pol=pol):
            agent = MasterHybridPPOInferenceAgent(_pol, forecaster=FormulaForecaster(), k=K_REC)
            agent.bind_to_sim(sim)
            if cap is not None:
                cap.sids = agent._roll.sids   # urutan stasiun TETAP, dipakai memetakan indeks->id
            return agent

    if cap is not None:
        cap.pasang()
    try:
        sim = common.fresh_sim(DS, rekam_deret=False)
        if cap is not None:
            cap.sim = sim
        random.seed(seed); np.random.seed(seed)
        agent = fac(sim)
        sim.run(max_steps=sim.max_steps, agent=agent)
    finally:
        if cap is not None:
            cap.lepas()

    # --- Sinyal 2: collision rate dari rec_distribution_log (SELALU dihitung) ---
    total_rec, total_tabrak = 0, 0
    for row in sim.rec_distribution_log:
        for sid, c in row["counts"].items():
            total_rec += c
            if c >= 2:
                total_tabrak += c  # SEMUA EV yg berbagi stasiun tsb, bukan cuma kelebihannya
    collision_rate = total_tabrak / total_rec if total_rec else float("nan")

    hasil_atensi = None
    if cap is not None and cap.rows_w:
        hasil_atensi = (np.array(cap.rows_w, dtype=np.float32),
                       np.array(cap.rows_mask, dtype=bool),
                       np.array(cap.rows_util, dtype=np.float32),
                       np.array(cap.rows_uid, dtype='U16'),
                       np.array(cap.rows_step, dtype=np.int32),
                       cap.sids)
    return collision_rate, hasil_atensi


def main():
    print(f"lengan={TAG_ARM}  horizon={TAG} ({DS})  punya_atensi={PUNYA_ATENSI}", flush=True)
    print(f"mode={'collision-saja' if COLLISION_SAJA else 'penuh (bobot + collision)'}\n",
         flush=True)

    kol = []
    Ws, Ms, Us, Uids, Ss, sids_ref = [], [], [], [], [], None
    for sd in SEEDS:
        cr, hasil = satu_run(sd)
        kol.append(cr)
        print(f"  [seed={sd}] collision_rate={cr:.4f}"
             + (f"  atensi_tertangkap={len(hasil[0])}" if hasil else ""), flush=True)
        if hasil is not None:
            w, m, u, uid, st, sids = hasil
            Ws.append(w); Ms.append(m); Us.append(u); Uids.append(uid); Ss.append(st)
            sids_ref = sids

    common.save_json({"tag_arm": TAG_ARM, "seeds": SEEDS,
                      "collision_rate_per_seed": kol,
                      "collision_rate_mean": float(np.mean(kol))},
                     f"collision_{TAG_ARM}.json")
    print(f"\nSAVED -> outputs/collision_{TAG_ARM}.json  "
         f"(rerata={np.mean(kol):.4f})", flush=True)

    if Ws:
        out = os.path.join(common.OUTDIR, f"attensi_{TAG_ARM}.npz")
        np.savez_compressed(out, w=np.concatenate(Ws), mask=np.concatenate(Ms),
                            util=np.concatenate(Us), user_id=np.concatenate(Uids),
                            step=np.concatenate(Ss), sids=np.array(sids_ref, dtype='U16'))
        total = sum(len(w) for w in Ws)
        print(f"SAVED -> outputs/attensi_{TAG_ARM}.npz  (total {total:,} keputusan)",
             flush=True)
    elif PUNYA_ATENSI and not COLLISION_SAJA:
        print("PERINGATAN: lengan punya atensi tapi nol keputusan tertangkap -- periksa.",
             flush=True)


if __name__ == "__main__":
    main()
