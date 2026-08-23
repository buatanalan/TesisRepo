"""Evaluasi metrik-kaya (gini/acc/wait/trust) SATU checkpoint lintas N SEED LINGKUNGAN
berbeda -- pemisahan checkpoint-seed vs eval-seed, TAK ADA di `_uji_master_ev_ppo_metrik.
py` asli (di sana 1 seed = 1 checkpoint = 1 titik evaluasi, cuma sebanyak seed yang
dilatih). Di sini: SATU checkpoint (biasanya seed ber-Gini MEDIAN, sama konvensi
`_run_master_ev_ppo_pipeline.py::pick_median`), dievaluasi lintas `n_eval_seed`
lingkungan simulasi berbeda -- pola SAMA PERSIS `eval_policy_gini` (Wilcoxon+Cohen's d
di eval_results.json), TAPI dgn metrik kaya (acc/wait/trust) yang di situ tak dihitung.

Pemakaian:
    python _uji_master_ev_ppo_metrik_full.py <ckpt_seed> <n_eval_seed> <horizon> <tag_arm>

Contoh (checkpoint seed median `pref_feat_nohist_acc1_vwf_K3`, 10 seed evaluasi):
    python _uji_master_ev_ppo_metrik_full.py 1 10 30d master_ev_ppo_pref_feat_nohist_acc1_vwf_K3

Konvensi penamaan `tag_arm` (pref_feat/nohist/accN/vwf/K<n>/prefk<n>) DIWARISI PERSIS
dari `_uji_master_ev_ppo_metrik.py` (regex sama) -- lihat file itu utk detail per suku.
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

assert len(sys.argv) > 4, (
    "pemakaian: python _uji_master_ev_ppo_metrik_full.py <ckpt_seed> <n_eval_seed> "
    "<horizon> <tag_arm>")
CKPT_SEED = int(sys.argv[1])
N_EVAL_SEED = int(sys.argv[2])
TAG = sys.argv[3]
TAG_ARM = sys.argv[4]

# --- Dekode konfigurasi dari nama tag -- IDENTIK `_uji_master_ev_ppo_metrik.py` ---
_is_pref_feat = "pref_feat" in TAG_ARM
_is_pref = "pref" in TAG_ARM
_is_vwf = "vwf" in TAG_ARM
_m_critics = re.search(r"_K(\d+)(?:_|$)", TAG_ARM)
N_CRITICS = int(_m_critics.group(1)) if _m_critics else 1
_is_nohist = "nohist" in TAG_ARM
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
FORECASTER_CLS = K.VW if _is_vwf else FormulaForecaster
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])
K_REC = 3


def muat_policy(ckpt_seed, n_spklu):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{ckpt_seed}.pt")
    assert os.path.exists(ckpt), f"checkpoint tak ditemukan: {ckpt}"
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
    print(f"checkpoint seed={CKPT_SEED}  n_eval_seed={N_EVAL_SEED}  tag_arm={TAG_ARM}", flush=True)
    sim0 = common.fresh_sim(K.DS)
    n_spklu = len(sim0.spklus)
    policy = muat_policy(CKPT_SEED, n_spklu)
    fac = fac_dari_policy(policy)

    per_seed = {}
    agregat = {}
    harian = {}
    eval_seeds = list(range(N_EVAL_SEED))
    for mode in ("abs", "signed"):
        for beku in (False, True):
            label = f"{LABEL_ARM}|{mode}|{'beku' if beku else 'dinamis'}"
            runs = []
            for sd in eval_seeds:
                print(f"  [{label}] eval_seed={sd} ...", flush=True)
                r = K.satu_run(fac, mode, sd, beku)
                runs.append(r)
            per_seed[label] = runs
            agregat[label] = K.agg(runs)
            harian[label] = K.agg_harian(runs)
            print(f"  [{label}] gini={agregat[label]['gini']:.4f} "
                 f"wait={agregat[label]['wait']:.1f} trust={agregat[label]['trust']:.3f} "
                 f"acc={agregat[label]['acc']:.3f}", flush=True)

    out = dict(horizon=TAG, ckpt_seed=CKPT_SEED, eval_seeds=eval_seeds,
              per_seed=per_seed, agregat=agregat, harian=harian)
    nama = f"uji_{TAG_ARM}_full{N_EVAL_SEED}_metrik_{TAG}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    main()
