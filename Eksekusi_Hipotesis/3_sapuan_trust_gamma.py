"""LANGKAH 3 (opsional) -- sapuan evaluasi: kepercayaan awal x gamma.

    python 3_sapuan_trust_gamma.py --lihat
    python 3_sapuan_trust_gamma.py
    python 3_sapuan_trust_gamma.py --tag h6b_utama,h1a_pemerataan_dgr

EVALUASI SAJA -- tidak melatih apa pun. Model yang sudah ada dijalankan ulang di
berbagai kondisi populasi, untuk melihat seberapa jauh keunggulannya bertahan.

Dua sumbu:

  kepercayaan awal   seberapa besar peluang pengguna mengikuti rekomendasi di awal
  gamma              seberapa PEKA pengguna terhadap lama tunggu yang DIJANJIKAN,
                     lewat  P_rec(j) ∝ exp(-gamma · EstWait_j)

gamma dinyatakan dalam satuan yang bisa dibaca: PARUH, yaitu pada berapa menit janji
tunggu daya tarik rekomendasi tinggal separuh. Bawaan sistem setara paruh 12,4 menit.

  paruh  6 mnt -> gamma 0,1155   pengguna sangat pemilih
  paruh 12 mnt -> gamma 0,0578   ~bawaan
  paruh 25 mnt -> gamma 0,0277   cukup sabar
  paruh 50 mnt -> gamma 0,0139   nyaris tak peduli lama tunggu

Kenapa dua sumbu ini bersama, bukan sendiri-sendiri: keduanya masuk ke persamaan yang
sama, `P = (1-T)·P_pref + T·P_rec`. Kepercayaan menentukan SEBERAPA BESAR suara
rekomendasi; gamma menentukan APA ISI suara itu ketika didengar. Menyapu satu tanpa yang
lain tak bisa membedakan "rekomendasi diabaikan" dari "rekomendasi didengar tapi tak
meyakinkan".

CATATAN model yang perlu diketahui: gamma bersifat GLOBAL (sama untuk semua pengguna),
bukan per-individu.
"""
import sys, os, glob, json, random, argparse, datetime, time, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _kakas as K
import common
import torch
from marl_spklu.experiments.ablations import initial_trust
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOPolicy, MasterEVPPOPrefPolicy,
                                                MasterEVPPOInferenceAgent)

STAMP = datetime.date.today().strftime("%Y%m%d")

# Paruh (menit) -> gamma. Ditulis sebagai paruh supaya bermakna saat dilaporkan.
PARUH = [6, 12, 25, 50]
TRUST = [0.3, 0.5, 0.7]


def temukan_ckpt(tag, horizon="30d"):
    """Model seed-median untuk satu lengan. Nama berkas mengikuti `_pipeline_hipotesis`:
    `<tag>__it<XX>[_90d]_<tanggal>_actor_seed<N>.pt`, dengan seed median tercatat di
    berkas `_eval.json` pasangannya."""
    suf = "" if horizon == "30d" else f"_{horizon}"
    ev = sorted(glob.glob(os.path.join(K.OUTDIR, f"{tag}__it05{suf}_*_eval.json")))
    if not ev:
        return None, None
    d = json.load(open(ev[-1], encoding="utf-8"))
    stem = d["tag_stamped"]
    ck = os.path.join(K.OUTDIR, f"{stem}_actor_seed{d['seed_median']}.pt")
    return (ck if os.path.exists(ck) else None), d.get("config", {})


def muat(ckpt, cfg, n_spklu):
    pref = bool(cfg.get("pref"))
    kw = dict(pref_feature_mode=bool(cfg.get("pref_feature_mode"))) if pref else {}
    if cfg.get("no_hist"):
        kw["use_hist"] = False
    pol = (MasterEVPPOPrefPolicy if pref else MasterEVPPOPolicy)(
        n_spklu, n_critics=int(cfg.get("n_critics", 1)), **kw)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


def forecaster(cfg):
    from marl_spklu.rl.forecaster import FormulaForecaster, VirtualWaitForecaster
    return VirtualWaitForecaster() if cfg.get("forecaster") == "vwf" else FormulaForecaster()


def satu_sel(pol, cfg, ds, it, gamma, n_eval, k):
    """Satu sel sapuan. Kedua konteks dipasang SEBELUM `fresh_sim` -- `initial_trust`
    menambal `User.__init__`, jadi harus aktif saat pengguna dibuat."""
    hasil = []
    for s in range(n_eval):
        with K.gamma_pengguna(gamma), initial_trust(value=it):
            sim = common.fresh_sim(ds)
            random.seed(s); np.random.seed(s)
            ag = MasterEVPPOInferenceAgent(pol, sim, forecaster(cfg), k=k)
            sim.run(max_steps=sim.max_steps, agent=ag)
            hasil.append(K.metrik_sim(sim))
    return hasil


def satu_sel_greedy(ds, mode, it, gamma, n_eval):
    hasil = []
    for s in range(n_eval):
        with K.gamma_pengguna(gamma), initial_trust(value=it):
            sim = common.fresh_sim(ds)
            random.seed(s); np.random.seed(s)
            sim.run(max_steps=sim.max_steps, agent=GreedyAgent(mode=mode))
            hasil.append(K.metrik_sim(sim))
    return hasil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="h6b_utama,h1a_pemerataan",
                   help="lengan yang disapu, dipisah koma")
    p.add_argument("--horizon", type=str, default="30d", choices=["30d", "90d"])
    p.add_argument("--n-eval-seed", type=int, default=5)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--paruh", type=str, default=",".join(map(str, PARUH)),
                   help="paruh gamma dalam MENIT, dipisah koma")
    p.add_argument("--trust", type=str, default=",".join(map(str, TRUST)))
    p.add_argument("--greedy", action="store_true",
                   help="ikutkan greedy_util sebagai lantai di tiap sel (menambah waktu)")
    p.add_argument("--lihat", action="store_true")
    args = p.parse_args()

    tags = [t.strip() for t in args.tag.split(",") if t.strip()]
    paruh = [float(x) for x in args.paruh.split(",")]
    trust = [float(x) for x in args.trust.split(",")]
    ds = os.path.join(common.ROOT, K.DATASET_30D if args.horizon == "30d" else K.DATASET_90D)

    K.judul(f"SAPUAN kepercayaan x gamma   ({args.horizon})")
    print(f"  lengan   : {', '.join(tags)}")
    print(f"  trust    : {trust}")
    print(f"  paruh    : {[f'{h:g} mnt (gamma {K.gamma_dari_paruh(h):.4f})' for h in paruh]}")
    n_sel = len(tags) * len(trust) * len(paruh)
    print(f"  sel      : {n_sel} x {args.n_eval_seed} seed = "
          f"{n_sel * args.n_eval_seed} simulasi"
          + (f"  (+greedy: {len(trust)*len(paruh)*args.n_eval_seed})" if args.greedy else ""))

    siap = {}
    print()
    for t in tags:
        ck, cfg = temukan_ckpt(t, args.horizon)
        print(f"  {'OK   ' if ck else 'HILANG'}  {t:<22} "
              f"{os.path.basename(ck) if ck else '-- model tak ditemukan di outputs/'}")
        if ck:
            siap[t] = (ck, cfg)
    K.butuh(siap, "Tak satu pun model ditemukan. Sapuan ini EVALUASI SAJA -- model harus "
                  "sudah ada di outputs/.\n   Jalankan dulu:  python 1_eksperimen.py")

    if args.lihat:
        print("\n(--lihat: tidak ada yang dijalankan)")
        return

    t0, keluar = time.time(), {}
    sim0 = common.fresh_sim(ds)
    n_spklu = len(sim0.spklus)

    for t, (ck, cfg) in siap.items():
        pol = muat(ck, cfg, n_spklu)
        for it, h in itertools.product(trust, paruh):
            g = K.gamma_dari_paruh(h)
            rows = satu_sel(pol, cfg, ds, it, g, args.n_eval_seed, args.k)
            r = {m: float(np.mean([x[m] for x in rows])) for m in rows[0]}
            keluar[f"{t}|it{it:g}|paruh{h:g}"] = dict(
                lengan=t, initial_trust=it, paruh_menit=h, gamma=g, **r)
            print(f"  [{time.time()-t0:6.0f}s] {t:<22} it={it:g} paruh={h:g}mnt  "
                  f"gini={r['gini']:.4f} terima={r['acc']:.3f} "
                  f"tunggu={r['wait']:7.1f} percaya={r['trust']:.3f}", flush=True)

    if args.greedy:
        for it, h in itertools.product(trust, paruh):
            g = K.gamma_dari_paruh(h)
            rows = satu_sel_greedy(ds, "utilization", it, g, args.n_eval_seed)
            r = {m: float(np.mean([x[m] for x in rows])) for m in rows[0]}
            keluar[f"greedy_util|it{it:g}|paruh{h:g}"] = dict(
                lengan="greedy_util", initial_trust=it, paruh_menit=h, gamma=g, **r)
            print(f"  [{time.time()-t0:6.0f}s] {'greedy_util':<22} it={it:g} paruh={h:g}mnt  "
                  f"gini={r['gini']:.4f} terima={r['acc']:.3f} "
                  f"tunggu={r['wait']:7.1f} percaya={r['trust']:.3f}", flush=True)

    # ---------------------------------------------------------------- tabel
    for metrik, judul in (("gini", "GINI"), ("acc", "PENERIMAAN"), ("wait", "WAKTU TUNGGU")):
        K.judul(f"{judul}   (baris = kepercayaan awal, kolom = paruh gamma)")
        for t in list(siap) + (["greedy_util"] if args.greedy else []):
            print(f"\n  {t}")
            print("    trust  " + "".join(f"{h:>10.0f}mnt" for h in paruh))
            for it in trust:
                sel = [keluar.get(f"{t}|it{it:g}|paruh{h:g}", {}).get(metrik) for h in paruh]
                print(f"    {it:<7g}" + "".join(
                    "         -   " if v is None else f"{v:>13.4f}" for v in sel))

    K.simpan(dict(tanggal=STAMP, horizon=args.horizon,
                  config=dict(tags=tags, trust=trust, paruh_menit=paruh,
                              n_eval_seed=args.n_eval_seed, k=args.k,
                              dataset=os.path.basename(ds)),
                  catatan="gamma bersifat GLOBAL (sama semua pengguna), bukan per-individu. "
                          "Sapuan ini EVALUASI SAJA -- model tidak dilatih ulang di tiap sel, "
                          "sehingga yang diukur adalah KETAHANAN kebijakan terhadap "
                          "pergeseran populasi, bukan kinerja optimalnya di tiap kondisi.",
                  sel=keluar),
             f"sapuan_trust_gamma_{args.horizon}_{STAMP}.json")
    print(f"\nSelesai dalam {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
