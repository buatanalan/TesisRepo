"""Ekstraksi representasi laten `pref_lstm` (Uji A, 2026-09-01) -- menjawab kritik:
"bagaimana Anda tahu LSTM benar-benar belajar memisahkan preferensi dari kepercayaan,
bukan hanya menghafal pola urutan stasiun?"

Menangkap keluaran `_PrefStationBackbone._encode_pref()` (vektor h_n, dim=pref_d_lstm=8)
untuk SETIAP keputusan pengguna sepanjang satu simulasi evaluasi, dipasangkan dengan:
  - `sid_pref`  : stasiun yang akan dipilih pengguna TANPA rekomendasi (kontrafaktual,
                  dari `sim.decision_log`, sudah dihitung `Simulator._jejak_pengaruh_janji`)
  - `trust`     : trust_effective pengguna PADA SAAT keputusan itu

WAJIB dijalankan untuk KEDUA lengan (bukan cuma P-MASTER) -- MASTER (gate~0, modul P
behaviorally "mati" menurut §5.7 draf Bab V) adalah PEMBANDING NEGATIF: bila representasi
P-MASTER menunjukkan struktur sementara MASTER tidak, itu bukti struktur muncul dari
pelatihan dgn modul aktif, bukan artefak arsitektur LSTM semata. Tanpa pembanding ini,
argumen tetap rentan disanggah "LSTM mana pun begitu saja".

Pemakaian (di server, checkpoint actor MASTER hanya ada di sana):
    python _ekstrak_representasi_laten.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3
    python _ekstrak_representasi_laten.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3

Keluaran: outputs/representasi_{TAG_ARM}.npz per checkpoint (digabung lintas seed),
berisi array h (N,8), sid_pref (N,) string, trust (N,) float, user_id (N,) string,
step (N,) int -- siap dianalisis oleh `_analisis_representasi_laten.py`.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, common

import re as _re
from marl_spklu.rl.master_pure_hybrid_policy import MasterHybridPPOActor, _PrefStationBackbone
from marl_spklu.rl.master_pure_hybrid_trainer import MasterHybridPPOInferenceAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.training import _fresh_sim as _fresh_sim_common

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "90d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3"
K_REC = 3

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


# --------------------------------------------------------------- penangkap representasi
class _Penangkap:
    """Monkeypatch `_encode_pref` DI KELAS (bukan instance) -- semua panggilan lewat
    backbone actor yang sedang aktif otomatis tertangkap. `sim` diisi oleh pemanggil
    SEBELUM `sim.run()` supaya identitas pengguna & step bisa dibaca dari
    `sim._current_spawn_user`/`sim.current_step` (diset simulator SEBELUM
    `agent.get_recommendation()` dipanggil -- lihat `simulator.py` blok Hook RL)."""

    def __init__(self):
        self.sim = None
        self.rows_h, self.rows_uid, self.rows_step = [], [], []
        self._asli = _PrefStationBackbone._encode_pref

    def pasang(self):
        _cap = self

        def _encode_pref_tertangkap(diri, pref_hist):
            h_n = _cap._asli(diri, pref_hist)
            if _cap.sim is not None:
                u = getattr(_cap.sim, "_current_spawn_user", None)
                if u is not None:
                    # h_n bisa (1,d) -- B=1 selalu di jalur inferensi per-user
                    # (get_recommendation dipanggil sekali per keputusan, lih.
                    # master_pure_hybrid_trainer.py:124 obs_t shape (1,N)).
                    _cap.rows_h.append(h_n.detach().cpu().numpy()[0])
                    _cap.rows_uid.append(u.user_id)
                    _cap.rows_step.append(int(_cap.sim.current_step))
            return h_n

        _PrefStationBackbone._encode_pref = _encode_pref_tertangkap

    def lepas(self):
        _PrefStationBackbone._encode_pref = self._asli


def satu_run(seed):
    ckpt_idx = _CKPT[seed % len(_CKPT)]
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{ckpt_idx}.pt")
    n_spklu = len(_fresh_sim_common(DS).spklus)
    pol = MasterHybridPPOActor(n_spklu, **ACTOR_KW)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()

    cap = _Penangkap()
    cap.pasang()
    try:
        sim = common.fresh_sim(DS, rekam_deret=False)
        cap.sim = sim
        random.seed(seed); np.random.seed(seed)
        agent = MasterHybridPPOInferenceAgent(pol, forecaster=FormulaForecaster(), k=K_REC)
        agent.bind_to_sim(sim)
        sim.run(max_steps=sim.max_steps, agent=agent)
    finally:
        cap.lepas()

    # --- gabungkan dgn decision_log (sid_pref, trust) via (user_id, step) ---
    idx_log = {(e["user_id"], e["step"]): e for e in sim.decision_log}
    H, sid_pref, trust, uid, step = [], [], [], [], []
    hilang = 0
    for h, u, s in zip(cap.rows_h, cap.rows_uid, cap.rows_step):
        e = idx_log.get((u, s))
        if e is None or "sid_pref" not in e:
            hilang += 1
            continue
        H.append(h); sid_pref.append(e["sid_pref"]); trust.append(e["trust"])
        uid.append(u); step.append(s)
    print(f"  [seed={seed}, checkpoint {ckpt_idx}] tertangkap={len(cap.rows_h)} "
         f"tergabung={len(H)} hilang(tak cocok log)={hilang}", flush=True)
    return (np.array(H, dtype=np.float32), np.array(sid_pref, dtype='U16'),
            np.array(trust, dtype=np.float32), np.array(uid, dtype='U16'),
            np.array(step, dtype=np.int32))


def main():
    print(f"lengan={TAG_ARM}  horizon={TAG} ({DS})", flush=True)
    print(f"checkpoint tersedia: {_CKPT}", flush=True)
    Hs, Ps, Ts, Us, Ss, seeds_arr = [], [], [], [], [], []
    for sd in SEEDS:
        h, p, t, u, s = satu_run(sd)
        Hs.append(h); Ps.append(p); Ts.append(t); Us.append(u); Ss.append(s)
        seeds_arr.append(np.full(len(h), sd, dtype=np.int32))
    out = os.path.join(common.OUTDIR, f"representasi_{TAG_ARM}.npz")
    np.savez_compressed(out, h=np.concatenate(Hs), sid_pref=np.concatenate(Ps),
                        trust=np.concatenate(Ts), user_id=np.concatenate(Us),
                        step=np.concatenate(Ss), seed=np.concatenate(seeds_arr))
    total = sum(len(h) for h in Hs)
    print(f"\nSAVED -> outputs/representasi_{TAG_ARM}.npz  (total {total:,} keputusan)",
         flush=True)


if __name__ == "__main__":
    main()
