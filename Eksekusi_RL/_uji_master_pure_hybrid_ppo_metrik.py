"""Evaluasi Master-Hybrid PPO dgn metrik kaya sama `_uji_konsolidasi.py`.

Pemakaian:
    python _uji_master_pure_hybrid_ppo_metrik.py 0,1,2 30d master_hybrid_ppo_dgr
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_pure_hybrid_policy import MasterHybridPPOActor
from marl_spklu.rl.master_pure_hybrid_trainer import MasterHybridPPOInferenceAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.training import _fresh_sim as _fresh_sim_common

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_hybrid_ppo_dgr"
# argumen ke-5 opsional (2026-08-30): "cepat" -> HANYA signed|dinamis (1 kombinasi,
# bukan 4) -- mode paling realistis (signed = sesuai desain reward asli, dinamis =
# trust hidup sungguhan). Baku "" -> perilaku lama (4 kombinasi), TAK berubah utk
# pemanggil lama. Pakai "cepat" utk eksperimen eksploratif murah (mis. uji K=5);
# kembalikan ke default sebelum melaporkan hasil FINAL, krn abs|dinamis pernah
# terbukti membalik urutan gini (Gabungan vs Attn-saja) -- mode itu bisa jadi bukti
# penting, jangan dibuang permanen dari SELURUH pengujian.
MODE_SUBSET = sys.argv[4] if len(sys.argv) > 4 else ""
MODE_COMBOS = ([("signed", False)] if MODE_SUBSET == "cepat"
              else [(m, b) for m in ("abs", "signed") for b in (False, True)])
LABEL_ARM = f"MASTER-HYBRID-PPO[{TAG_ARM}]"
K_REC = 3
# Mode pref DITURUNKAN dari TAG_ARM -- checkpoint & rekonstruksi eval WAJIB sama
# bentuk jaringannya (kelas bug "latih & uji beda mode", berulang di repo ini).
import re as _re
_m_histk = _re.search(r"_histK(\d+)", TAG_ARM)
# `initial_trust` DITURUNKAN dari TAG_ARM (bukan argumen terpisah) -- kebijakan WAJIB
# diuji di lingkungan yang SAMA dgn tempatnya dilatih. Menurunkannya dari tag membuat
# ketidakcocokan latih/uji mustahil terjadi karena lupa meneruskan argumen, kelas bug
# yang sudah berulang di repo ini (pref_hist saat uji, station_feat_dim, dst).
_m_it = _re.search(r"_it([0-9.]+)", TAG_ARM)
INIT_TRUST = float(_m_it.group(1)) if _m_it else None
ACTOR_KW = dict(vec_dim=8, bid_hidden=16, pref_d_lstm=8, pref_d_attn=8, station_attn_dim=8,
                pref_feature_mode="_preffeat" in TAG_ARM,
                pref_pair_outcome="_pairout" in TAG_ARM,
                use_station_attn="_noattn" not in TAG_ARM,
                pref_hist_k=(int(_m_histk.group(1)) if _m_histk else None),
                station_feat_dim=(10 if "_evobs" in TAG_ARM else 7))
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])


def _checkpoint_tersedia():
    """Daftar indeks seed checkpoint yang BENAR-BENAR ada, terurut."""
    ada = []
    i = 0
    while True:
        if not os.path.exists(os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{i}.pt")):
            break
        ada.append(i)
        i += 1
    return ada


_CKPT = _checkpoint_tersedia()


def ckpt_untuk(seed):
    """Pola LATIH 3 SEED / EVAL 10 SEED (2026-08-30): checkpoint dipakai BERGILIR
    (`seed % jumlah_checkpoint`), bukan jatuh semua ke seed 0.

    ALASAN: fallback lama mengarahkan SETIAP seed yang checkpoint-nya tak ada ke seed 0,
    sehingga pd 3 checkpoint x 10 seed eval, checkpoint 0 terpakai 8 dari 10 sampel dan
    agregat didominasi SATU kebijakan. Bergilir membuatnya 4/3/3 -- hampir seimbang.

    CATATAN STATISTIK PENTING: 10 seed eval BUKAN 10 sampel bebas. Ia mengecilkan derau
    LINGKUNGAN pd tiap checkpoint, tetapi jumlah kebijakan bebas tetap sebanyak
    checkpoint yang dilatih (3). Simpangan baku yang layak dilaporkan sbg ketidakpastian
    antar-kebijakan adalah simpangan baku dari RERATA PER-CHECKPOINT, bukan simpangan
    baku ke-10 run mentah (yang akan meremehkan ragam antar-kebijakan)."""
    assert _CKPT, (
        f"tak ada checkpoint sama sekali utk {TAG_ARM}\n"
        f"latih dulu: python _run_master_pure_hybrid_ppo_pipeline.py --n-train-seed 3 ...")
    return _CKPT[seed % len(_CKPT)]


def muat_policy(seed, n_spklu):
    c = ckpt_untuk(seed)
    if c != seed:
        print(f"  [bergilir] seed eval={seed} -> checkpoint seed{c} "
             f"(simulasi tetap seed={seed}; {len(_CKPT)} checkpoint tersedia)", flush=True)
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{c}.pt")
    pol = MasterHybridPPOActor(n_spklu, **ACTOR_KW)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


def fac_dari_policy(policy):
    def fac(sim, _pol=policy):
        agent = MasterHybridPPOInferenceAgent(_pol, forecaster=FormulaForecaster(), k=K_REC)
        agent.bind_to_sim(sim)
        return agent
    return fac


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)
    if INIT_TRUST is not None:
        print(f"lingkungan: initial_trust={INIT_TRUST:g} (diturunkan dari tag)", flush=True)
    n_spklu = len(_fresh_sim_common(K.DS).spklus)

    per_seed = {}
    agregat = {}
    harian = {}
    for mode, beku in MODE_COMBOS:
        label = f"{LABEL_ARM}|{mode}|{'beku' if beku else 'dinamis'}"
        runs = []
        for sd in SEEDS:
            policy = muat_policy(sd, n_spklu)
            fac = fac_dari_policy(policy)
            print(f"  [{label}] seed={sd} ...", flush=True)
            r = K.satu_run(fac, mode, sd, beku)
            runs.append(r)
        per_seed[label] = runs
        agregat[label] = K.agg(runs)
        harian[label] = K.agg_harian(runs)
        print(f"  [{label}] gini={agregat[label]['gini']:.4f} "
             f"wait={agregat[label]['wait']:.1f} trust={agregat[label]['trust']:.3f} "
             f"acc={agregat[label]['acc']:.3f}", flush=True)

    # `ckpt_per_seed` WAJIB ikut tersimpan: pd pola latih-3/eval-10 beberapa seed eval
    # berbagi checkpoint yang sama, sehingga analisis lanjutan HARUS bisa mengelompokkan
    # run menurut kebijakannya (lihat catatan statistik di `ckpt_untuk`). Tanpa peta ini,
    # simpangan baku ke-10 run akan disalahartikan sbg ragam antar-kebijakan.
    out = dict(horizon=TAG, seeds=SEEDS, per_seed=per_seed, agregat=agregat, harian=harian,
               ckpt_per_seed={int(s): int(ckpt_untuk(s)) for s in SEEDS},
               n_checkpoint=len(_CKPT))
    nama = f"uji_{TAG_ARM}_metrik_{TAG}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    # Konteks dibentangkan menutupi SELURUH main(): `K.satu_run` membuat simulator baru
    # tiap run, dan `initial_trust` menambal `User.__init__` -- jadi harus aktif saat
    # setiap simulator dibentuk, bukan sekali di awal.
    import contextlib
    if INIT_TRUST is not None:
        from marl_spklu.experiments.ablations import initial_trust as _initial_trust
        _ctx = _initial_trust(INIT_TRUST)
    else:
        _ctx = contextlib.nullcontext()
    with _ctx:
        main()
