"""Evaluasi MasterEV-PPO dengan METRIK KAYA yang sama dengan `_uji_konsolidasi.py` --
mirror `_uji_master_ev_metrik.py`, checkpoint dari `_run_master_ev_ppo_pipeline.py`.

Pemakaian:
    python _uji_master_ev_ppo_metrik.py 0,1,2 30d [tag_arm]
    tag_arm opsional -- default "master_ev_ppo". Utk varian +P (lihat `--pref`/
    `--pref-feature-mode` di `_run_master_ev_ppo_pipeline.py`), berikan eksplisit:
    "master_ev_ppo_pref" atau "master_ev_ppo_pref_feat".
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOPolicy, MasterEVPPOPrefPolicy,
                                                MasterEVPPOInferenceAgent)
from marl_spklu.rl.forecaster import FormulaForecaster

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "30d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_ev_ppo"
# PERBAIKAN: pencocokan SUBSTRING (bukan `==` eksak) -- tag_arm bisa dapat akhiran
# horizon (mis. "master_ev_ppo_pref_feat_90d"), pencocokan eksak lama membuat
# pref_feature_mode salah terdeteksi False -> dim LSTM riwayat salah (12 bukan 10) ->
# RuntimeError size-mismatch saat load_state_dict.
_is_pref_feat = "pref_feat" in TAG_ARM
_is_pref = "pref" in TAG_ARM
# `vwf`: forecaster HARUS SAMA dgn yg dipakai `_run_master_ev_ppo_pipeline.py --forecaster
# vwf` saat melatih checkpoint ini -- pakai `K.VW` (_uji_konsolidasi.py), BUKAN kelas
# terpisah, supaya identik persis dgn yg dipakai evaluasi H-PPO/P-PPO/MASTER (konvensi
# Tahap 2: "Prediktor EstWait SERAGAM lintas semua metode -- syarat perbandingan setara").
_is_vwf = "vwf" in TAG_ARM
# `n_critics`: HARUS SAMA dgn --n-critics saat melatih -- dim keluaran critic_head
# bergantung ini, mismatch -> RuntimeError size saat load_state_dict (sama kelas bug
# pref_feature_mode sebelumnya). Dideteksi dari akhiran "_K<n>" (lihat _run_master_ev_
# ppo_pipeline.py -- BAKU n_critics=1 tak dapat akhiran).
_m_critics = re.search(r"_K(\d+)(?:_|$)", TAG_ARM)
N_CRITICS = int(_m_critics.group(1)) if _m_critics else 1
# `nohist`: TAK mengubah bentuk jaringan (hist_lstm tetap ada, cuma gerbang runtime) --
# tapi WAJIB cocok dgn saat latih, kalau tidak c_t acak (bobot hist_lstm tak terlatih,
# krn selama training kontribusinya dinolkan) diam-diam ikut mencemari forward() --
# TIDAK memicu RuntimeError spt mismatch dim, jadi harus benar sejak awal.
_is_nohist = "nohist" in TAG_ARM
# `stattn`: MENGUBAH bentuk jaringan (nn.Module baru station_attn.*/station_attn_gate
# ditambahkan ke state_dict) -- WAJIB cocok dgn saat latih, kalau tidak
# load_state_dict RuntimeError "Unexpected key(s)" (lihat commit StationSelfAttention).
_is_stattn = "stattn" in TAG_ARM
# `stattnd<N>`: dimensi Q/K/V/out StationSelfAttention -- MENGUBAH bentuk bobot
# (Linear(N,N) bukan Linear(hidden,hidden)), WAJIB cocok spt N_CRITICS/pref_feature_mode.
_m_stattn_dim = re.search(r"stattnd(\d+)", TAG_ARM)
STATION_ATTN_DIM = int(_m_stattn_dim.group(1)) if _m_stattn_dim else None
# `concatA<N>H<M>`: StationConcatDecisionHead -- MENGUBAH bentuk bobot (menggantikan
# disc_head sepenuhnya), WAJIB cocok spt stattn di atas.
_m_concat = re.search(r"_concatA(\d+)H(\d+)", TAG_ARM)
CONCAT_ATTN_DIM = int(_m_concat.group(1)) if _m_concat else None
CONCAT_MLP_HIDDEN = int(_m_concat.group(2)) if _m_concat else None
_is_concat = _m_concat is not None
# `sepcritH<N>`: head kritik kecil TERPISAH per-aliran DGR (2026-08-28) -- MENGUBAH
# bentuk bobot (critic_heads.* ModuleList menggantikan critic_head tunggal), WAJIB
# cocok spt stattn/concat di atas.
_m_sepcrit = re.search(r"_sepcritH(\d+)", TAG_ARM)
CRITIC_HEAD_SMALL = int(_m_sepcrit.group(1)) if _m_sepcrit else None
_is_sepcrit = _m_sepcrit is not None
# `prefk<N>`: panjang jendela riwayat P -- TAK mengubah bentuk jaringan (LSTM terima
# urutan berapa pun) shg mismatch TAK memicu RuntimeError, harus benar sejak awal
# (sama kelas risiko `nohist`). None (tak ada akhiran) -> baku PDQN_HIST_K=10.
_m_prefk = re.search(r"_prefk(\d+)(?:_|$)", TAG_ARM)
PREF_HIST_K = int(_m_prefk.group(1)) if _m_prefk else None
LABEL_ARM = (f"MASTER-EV-PPO+P(fitur)[{TAG_ARM}]" if _is_pref_feat
            else f"MASTER-EV-PPO+P[{TAG_ARM}]" if _is_pref
            else f"MASTER-EV-PPO[{TAG_ARM}]")
POLICY_CLS = MasterEVPPOPrefPolicy if _is_pref else MasterEVPPOPolicy
POLICY_KW = dict(pref_feature_mode=True) if _is_pref_feat else dict()
POLICY_KW["n_critics"] = N_CRITICS
if _is_nohist:
    POLICY_KW["use_hist"] = False
if _is_stattn:
    POLICY_KW["use_station_attn"] = True
    if STATION_ATTN_DIM is not None:
        POLICY_KW["station_attn_dim"] = STATION_ATTN_DIM
if _is_concat:
    POLICY_KW["use_concat_head"] = True
    POLICY_KW["concat_attn_dim"] = CONCAT_ATTN_DIM
    POLICY_KW["concat_mlp_hidden"] = CONCAT_MLP_HIDDEN
if _is_sepcrit:
    POLICY_KW["separate_critic_heads"] = True
    POLICY_KW["critic_head_small"] = CRITIC_HEAD_SMALL
FORECASTER_CLS = K.VW if _is_vwf else FormulaForecaster
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])
K_REC = 3


def muat_policy(seed, n_spklu):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    assert os.path.exists(ckpt), (
        f"checkpoint tak ditemukan: {ckpt}\n"
        f"latih dulu lewat: python _run_master_ev_ppo_pipeline.py --n-train-seed {seed+1} ...")
    pol = POLICY_CLS(n_spklu, **POLICY_KW)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


def fac_dari_policy(policy):
    def fac(sim, _pol=policy):
        return MasterEVPPOInferenceAgent(_pol, sim, FORECASTER_CLS(), k=K_REC,
                                         pref_hist_k=PREF_HIST_K)
    return fac


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)
    sim0 = common.fresh_sim(K.DS)
    n_spklu = len(sim0.spklus)

    per_seed = {}
    agregat = {}
    harian = {}
    for mode in ("abs", "signed"):
        for beku in (False, True):
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

    out = dict(horizon=TAG, seeds=SEEDS, per_seed=per_seed, agregat=agregat, harian=harian)
    nama = f"uji_{TAG_ARM}_metrik_{TAG}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    main()
