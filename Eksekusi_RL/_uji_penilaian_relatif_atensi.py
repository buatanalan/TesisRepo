"""Uji penilaian relatif antar-stasiun (2026-09-14).

Motivasi atensi antar-stasiun: skor sebuah stasiun seharusnya bergantung pada kondisi
stasiun LAIN yang menjadi kandidat pada keputusan yang sama (penilaian relatif), bukan
hanya pada fiturnya sendiri.

Tanpa atensi (MASTER), kepala aktor dibagi per-stasiun: logit_i = f(x_i). Mengubah
fitur stasiun j tidak pernah mengubah logit_i (i != j), dan urutan i vs k tidak pernah
berubah karena stasiun ketiga. Karena itu metrik silang MASTER = 0 PERSIS -- dipakai
sebagai pemeriksaan kebenaran kode, bukan temuan.

Catatan: MASTER+P juga TIDAK sepenuhnya bebas konteks. `PreferenceAttention` memakai
fitur semua stasiun sbg key/value, lalu hasilnya digabung per-stasiun secara
non-linear (LatePrefMerge). Maka pada lengan ber-P, efek silang bisa berasal dari jalur
P maupun jalur atensi. Untuk memisahkannya, metrik 1-3 pada lengan beratensi juga
dihitung ulang dengan `use_station_attn=False` pada BOBOT YANG SAMA.

Metrik (dihitung pada sampel observasi keputusan yang direkam saat evaluasi):
  M1 rasio sensitivitas silang : rerata |d logit_i / d x_j| (j != i) dibagi
                                 rerata |d logit_i / d x_i|, fitur keadaan stasiun
                                 (slots_avail, upcoming_demand, power, eta).
  M2 laju pembalikan peringkat : eta_norm stasiun j dinaikkan DELTA; proporsi pasangan
                                 (i,k), keduanya != j, yang urutan logit-nya berbalik.
  M3 arah respons silang       : perubahan p_j akibat logit stasiun LAIN saja
                                 (p_j[penuh] - p_j[hanya logit_j yang berubah]).
                                 Saat j memburuk, arah yang logis adalah NEGATIF
                                 (stasiun lain menjadi relatif lebih menarik).
  M4 kontribusi atensi         : sigmoid(gate_raw) dan ||gate*attended|| / ||vec||.
  M5 kesesuaian dgn waktu tunggu: proporsi pasangan kandidat yang urutan logit-nya
                                 searah dengan urutan waktu tunggu virtual (termasuk
                                 perjalanan pemohon). DESKRIPTIF, bukan ukuran benar/
                                 salah: kebijakan juga menimbang pemerataan & penerimaan.

Pemakaian (server, dari root repo):
    .venv/bin/python Eksekusi_RL/_uji_penilaian_relatif_atensi.py 90d <TAG_ARM> [langkah_sampel]
"""
import sys, os, random, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re as _re
import numpy as np
import torch
import common
from marl_spklu.rl.master_pure_hybrid_policy import MasterHybridPPOActor
from marl_spklu.rl.master_pure_hybrid_trainer import MasterHybridPPOInferenceAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.training import _fresh_sim as _fresh_sim_common

TAG = sys.argv[1] if len(sys.argv) > 1 else "90d"
TAG_ARM = (sys.argv[2] if len(sys.argv) > 2
           else "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3")
LANGKAH_SAMPEL = int(sys.argv[3]) if len(sys.argv) > 3 else 10
K_REC = 3

HORIZON = {"30d": "scenario_dataset_klaster12_4x.json",
           "90d": "scenario_dataset_klaster12_4x_90d.json",
           "90d6x": "scenario_dataset_klaster12_6x_90d.json"}
DS = os.path.join(common.ROOT, HORIZON[TAG])

# Indeks fitur build_station_obs (FEATURE_NAMES_MASTER):
# 0 index_norm, 1 time_sin, 2 time_cos, 3 slots_avail_norm, 4 upcoming_demand_norm,
# 5 power_norm, 6 eta_norm. Hanya fitur KEADAAN stasiun (3-6) yang dipakai M1 --
# index & waktu bukan kondisi yang relevan utk penilaian relatif.
FITUR_KEADAAN = [3, 4, 5, 6]
IDX_ETA = 6
DELTA_ETA = 0.5          # +30 menit (skala eta 60 menit)
ETA_MAKS = 5.0           # batas yang sama dgn build_station_obs

_m_histk = _re.search(r"_histK(\d+)", TAG_ARM)
ACTOR_KW = dict(vec_dim=8, bid_hidden=16, pref_d_lstm=8, pref_d_attn=8, station_attn_dim=8,
                pref_feature_mode="_preffeat" in TAG_ARM,
                pref_pair_outcome="_pairout" in TAG_ARM,
                use_station_attn="_noattn" not in TAG_ARM,
                pref_hist_k=(int(_m_histk.group(1)) if _m_histk else None),
                station_feat_dim=(10 if "_evobs" in TAG_ARM else 7))
PUNYA_ATENSI = ACTOR_KW["use_station_attn"]


def _checkpoint_tersedia():
    ada, i = [], 0
    while os.path.exists(os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{i}.pt")):
        ada.append(i); i += 1
    return ada


# ------------------------------------------------------------------ perekaman sampel
class _Perekam:
    """Merekam masukan aktor (station_obs, mask, pref_hist) setiap LANGKAH_SAMPEL
    keputusan, plus waktu tunggu virtual tiap kandidat utk M5. forward asli tetap
    dipanggil, jadi perilaku evaluasi tidak berubah."""

    def __init__(self):
        self.sim, self.sids, self.n_panggil = None, None, 0
        self.sampel = []
        self._asli = MasterHybridPPOActor.forward

    def pasang(self):
        cap = self

        def _forward(diri, station_obs, mask=None, pref_hist=None):
            cap.n_panggil += 1
            if cap.sim is not None and cap.n_panggil % LANGKAH_SAMPEL == 0:
                user = getattr(cap.sim, "_current_spawn_user", None)
                t_now = cap.sim.current_step * cap.sim.dt_minutes
                vw = None
                if user is not None and cap.sids:
                    vw = np.array([cap.sim.compute_virtual_wait(user, cap.sim.spklus[s], t_now)
                                   for s in cap.sids], dtype=np.float64)
                cap.sampel.append(dict(
                    obs=station_obs.detach().clone(),
                    mask=(mask.detach().clone() if mask is not None else None),
                    pref=(pref_hist.detach().clone() if pref_hist is not None else None),
                    vwait=vw))
            return cap._asli(diri, station_obs, mask, pref_hist)

        MasterHybridPPOActor.forward = _forward

    def lepas(self):
        MasterHybridPPOActor.forward = self._asli


def rekam(ckpt_idx, n_spklu):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{ckpt_idx}.pt")
    pol = MasterHybridPPOActor(n_spklu, **ACTOR_KW)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    cap = _Perekam()
    cap.pasang()
    try:
        sim = common.fresh_sim(DS, rekam_deret=False)
        cap.sim = sim
        random.seed(ckpt_idx); np.random.seed(ckpt_idx)
        agent = MasterHybridPPOInferenceAgent(pol, forecaster=FormulaForecaster(), k=K_REC)
        agent.bind_to_sim(sim)
        cap.sids = agent._roll.sids
        sim.run(max_steps=sim.max_steps, agent=agent)
    finally:
        cap.lepas()
    return pol, cap.sampel


# ------------------------------------------------------------------ metrik
def _logit(actor, obs, mask, pref):
    return actor(obs, mask, pref)[0]


def metrik_sensitivitas(actor, sampel):
    """M1, M2, M3 pada satu checkpoint."""
    rasio, flip_n, flip_tot, silang, sendiri = [], 0, 0, [], []
    for s in sampel:
        obs, mask, pref = s["obs"], s["mask"], s["pref"]
        valid = np.where(mask[0].numpy())[0] if mask is not None else np.arange(obs.shape[1])
        if len(valid) < 3:
            continue

        # --- M1: Jacobian logit terhadap fitur keadaan ---
        x = obs.clone().requires_grad_(True)
        lg = _logit(actor, x, mask, pref)
        for i in valid:
            g, = torch.autograd.grad(lg[i], x, retain_graph=True)
            g = g[0][:, FITUR_KEADAAN].abs().sum(dim=1).numpy()   # (N,)
            self_i = g[i]
            lain = [g[j] for j in valid if j != i]
            if self_i > 1e-12:
                rasio.append(float(np.mean(lain) / self_i))

        # --- M2 & M3: naikkan eta stasiun j ---
        with torch.no_grad():
            l0 = _logit(actor, obs, mask, pref).numpy()
            for j in valid:
                xp = obs.clone()
                xp[0, j, IDX_ETA] = min(float(xp[0, j, IDX_ETA]) + DELTA_ETA, ETA_MAKS)
                l1 = _logit(actor, xp, mask, pref).numpy()
                lain = [i for i in valid if i != j]
                for a in range(len(lain)):
                    for b in range(a + 1, len(lain)):
                        i, k = lain[a], lain[b]
                        d0, d1 = l0[i] - l0[k], l1[i] - l1[k]
                        if abs(d0) < 1e-9:
                            continue
                        flip_tot += 1
                        if np.sign(d0) != np.sign(d1):
                            flip_n += 1
                # dekomposisi perubahan p_j
                def _p(v):
                    z = v[valid] - v[valid].max()
                    e = np.exp(z)
                    return e / e.sum()
                pos = list(valid).index(j)
                p_awal = _p(l0)[pos]
                l_own = l0.copy(); l_own[j] = l1[j]
                p_own = _p(l_own)[pos]
                p_full = _p(l1)[pos]
                silang.append(float(p_full - p_own))
                sendiri.append(float(p_own - p_awal))
    silang = np.array(silang); sendiri = np.array(sendiri)
    bermakna = np.abs(silang) > 1e-6
    return dict(
        M1_rasio_silang=float(np.mean(rasio)) if rasio else float("nan"),
        M2_laju_pembalikan=float(flip_n / flip_tot) if flip_tot else float("nan"),
        M3_silang_rerata=float(silang.mean()) if silang.size else float("nan"),
        M3_sendiri_rerata=float(sendiri.mean()) if sendiri.size else float("nan"),
        M3_frac_arah_benar=(float((silang[bermakna] < 0).mean()) if bermakna.any()
                            else float("nan")),
        M3_frac_bermakna=float(bermakna.mean()) if silang.size else float("nan"),
    )


def metrik_kontribusi_atensi(actor, sampel):
    """M4 -- replikasi forward backbone utk mengukur besar residu atensi."""
    bb = actor.backbone
    gate = float(torch.sigmoid(bb.station_attn.gate_raw))
    rasio = []
    with torch.no_grad():
        for s in sampel:
            obs, mask, pref = s["obs"], s["mask"], s["pref"]
            vec = bb.vec_head(obs)
            if pref is not None:
                c = bb._encode_pref(pref)
                ap, _ = bb.pref_attn(obs, c)
                ap = bb.pref_gate * ap
            else:
                ap = torch.zeros(obs.shape[0], bb.pref_d_attn)
            vec = bb.late_merge(vec, ap)
            sa = bb.station_attn
            q, k, v = sa.q(vec), sa.k(vec), sa.v(vec)
            sc = torch.einsum("bnd,bmd->bnm", q, k) / (sa.d_attn ** 0.5)
            sc = sc.masked_fill(~mask.unsqueeze(1), float("-inf"))
            w = torch.nan_to_num(torch.softmax(sc, dim=-1), nan=0.0)
            att = sa.out(torch.einsum("bnm,bmd->bnd", w, v))
            valid = mask[0]
            num = (gate * att[0][valid]).norm(dim=-1)
            den = vec[0][valid].norm(dim=-1).clamp(min=1e-9)
            rasio.append(float((num / den).mean()))
    return dict(M4_gate=gate, M4_rasio_residu=float(np.mean(rasio)) if rasio else float("nan"))


def metrik_kesesuaian_wait(actor, sampel):
    """M5 -- kesesuaian urutan logit dgn waktu tunggu virtual (termasuk perjalanan)."""
    setuju, total = 0, 0
    with torch.no_grad():
        for s in sampel:
            if s["vwait"] is None:
                continue
            l = _logit(actor, s["obs"], s["mask"], s["pref"]).numpy()
            valid = np.where(s["mask"][0].numpy())[0]
            vw = s["vwait"]
            for a in range(len(valid)):
                for b in range(a + 1, len(valid)):
                    i, k = valid[a], valid[b]
                    if abs(vw[i] - vw[k]) < 1e-6 or abs(l[i] - l[k]) < 1e-9:
                        continue
                    total += 1
                    # logit lebih tinggi seharusnya berpasangan dgn tunggu lebih pendek
                    if np.sign(l[i] - l[k]) == np.sign(vw[k] - vw[i]):
                        setuju += 1
    return dict(M5_kesesuaian_wait=float(setuju / total) if total else float("nan"),
                M5_n_pasangan=int(total))


# ------------------------------------------------------------------ utama
def main():
    ckpts = _checkpoint_tersedia()
    assert ckpts, f"tak ada checkpoint utk {TAG_ARM}"
    n_spklu = len(_fresh_sim_common(DS).spklus)
    print(f"lengan={TAG_ARM}  horizon={TAG}  atensi={PUNYA_ATENSI}  "
          f"checkpoint={ckpts}  langkah_sampel={LANGKAH_SAMPEL}", flush=True)

    per_ckpt = []
    for c in ckpts:
        actor, sampel = rekam(c, n_spklu)
        print(f"  [ckpt {c}] sampel terekam: {len(sampel)}", flush=True)
        for p in actor.parameters():
            p.requires_grad_(False)
        hasil = dict(ckpt=c, n_sampel=len(sampel))
        hasil.update(metrik_sensitivitas(actor, sampel))
        hasil.update(metrik_kesesuaian_wait(actor, sampel))
        if PUNYA_ATENSI:
            hasil.update(metrik_kontribusi_atensi(actor, sampel))
            # ablasi atensi pada bobot yang sama -- memisahkan efek silang jalur atensi
            actor.backbone.use_station_attn = False
            tanpa = metrik_sensitivitas(actor, sampel)
            actor.backbone.use_station_attn = True
            hasil.update({f"tanpa_atensi_{k}": v for k, v in tanpa.items()})
        per_ckpt.append(hasil)
        print("    " + "  ".join(f"{k}={v:.4f}" for k, v in hasil.items()
                                if isinstance(v, float)), flush=True)

    kunci = [k for k in per_ckpt[0] if isinstance(per_ckpt[0][k], float)]
    ringkas = {}
    for k in kunci:
        vals = np.array([h[k] for h in per_ckpt], dtype=float)
        ringkas[k] = dict(mean=float(np.nanmean(vals)), sd=float(np.nanstd(vals)))
    out = dict(tag_arm=TAG_ARM, horizon=TAG, punya_atensi=PUNYA_ATENSI,
               delta_eta=DELTA_ETA, fitur_keadaan=FITUR_KEADAAN,
               langkah_sampel=LANGKAH_SAMPEL, per_checkpoint=per_ckpt, ringkas=ringkas)
    nama = f"uji_penilaian_relatif_{TAG_ARM}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    main()
