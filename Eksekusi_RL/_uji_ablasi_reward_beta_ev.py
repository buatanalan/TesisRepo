"""Ablasi 2x2 utk MASTER-EV: preset reward (default lama vs `seimbang4x` terkalibrasi,
sama dgn H-PPO/P-PPO) x mode Dynamic Gradient Re-weighting (`gap_ratio` MASTER asli vs
`fixed` 1/K seragam) -- mengisolasi DUA kandidat penyebab gini MASTER-EV masih jauh dari
greedy/H-PPO-P-PPO (lihat diskusi rekomendasi perbaikan):

  (1) MasterEVTrainer SEBELUMNYA memakai RewardCalculator() default (alpha_gini=0,5),
      BUKAN preset `seimbang4x` (alpha_gini=2,6019) yg jadi kalibrasi WAJIB H-PPO/P-PPO
      sejak Tahap 0.1 -- gerbang itu ditutup PERSIS krn suku gini terlalu lemah utk
      menghasilkan gradien berarti pada preset lama.
  (2) `beta` (bobot penggabungan 2 kritik) teramati KOLAPS ke [~0,~1] dlm 1-2 chunk saat
      smoke-test (`beta_mode="gap_ratio"` baku) -- berpotensi jadi sumber `critic_loss`
      yg liar, independen dari soal kalibrasi reward.

4 kondisi (2x2), SEED TUNGGAL & anggaran DIKURANGI (uji arah cepat, bukan angka final --
mirror `_uji_ablasi_prox_P.py`, tapi di sini via pipeline `MasterEVTrainer` LANGSUNG,
bukan lewat `_run_master_ev_pipeline.py`, supaya tak menimpa checkpoint produksi
`master_ev_actor_seed*.pt`).

Pemakaian:
    python _uji_ablasi_reward_beta_ev.py [n_updates] [seed]
    (default n_updates=150, seed=0 -- anggaran directional, BUKAN skala penuh 300)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, common
from marl_spklu.rl.master_ev_trainer import MasterEVTrainer, MasterEVInferenceAgent
from marl_spklu.rl.rewards import RewardCalculator

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

N_UPDATES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TAG = "30d"
DS = os.path.join(common.ROOT, K.HORIZON[TAG])
ROLLOUT_STEPS = 96
K_REC = 3
MODE = "signed"

_REWARD_DEFAULT = dict()
_REWARD_SB4X = dict(alpha_wait=0.0046, beta_prox=0.1, alpha_gini=2.6019,
                    alpha_flock=0.0208, use_delta_gini=True)

KONDISI = [
    ("default_gap", _REWARD_DEFAULT, "gap_ratio"),
    ("sb4x_gap",     _REWARD_SB4X,    "gap_ratio"),
    ("default_fixed", _REWARD_DEFAULT, "fixed"),
    ("sb4x_fixed",    _REWARD_SB4X,    "fixed"),
]


def latih(tag, reward_kw, beta_mode, seed):
    rc = RewardCalculator(**reward_kw)
    tr = MasterEVTrainer(DS, rollout_steps=ROLLOUT_STEPS, seed=seed, verbose=False,
                         k=K_REC, reward_calc=rc, beta_mode=beta_mode)
    actor, _critic = tr.train(n_updates=N_UPDATES)
    ckpt = os.path.join(common.OUTDIR, f"ablasi_evrb_{tag}_seed{seed}.pt")
    torch.save(actor.state_dict(), ckpt)
    return ckpt, tr.history


def muat(tag, seed):
    from marl_spklu.rl.master_ev_policy import MasterEVActor
    from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER_EV
    ckpt = os.path.join(common.OUTDIR, f"ablasi_evrb_{tag}_seed{seed}.pt")
    actor = MasterEVActor(STATION_FEAT_DIM_MASTER_EV)
    actor.load_state_dict(torch.load(ckpt, map_location="cpu"))
    actor.eval()
    return actor


def fac_dari_actor(actor):
    def fac(sim, _actor=actor):
        agent = MasterEVInferenceAgent(_actor, k=K_REC)
        agent.bind_to_sim(sim)
        return agent
    return fac


def main():
    print(f"horizon={TAG} ({DS})  n_updates={N_UPDATES} seed={SEED}", flush=True)

    for tag, reward_kw, beta_mode in KONDISI:
        ckpt = os.path.join(common.OUTDIR, f"ablasi_evrb_{tag}_seed{SEED}.pt")
        if os.path.exists(ckpt):
            print(f"  [{tag}] -- SKIP (checkpoint ada)", flush=True)
            continue
        print(f"  [{tag}] -- latih (preset={'sb4x' if reward_kw else 'default'} "
             f"beta={beta_mode})...", flush=True)
        _ckpt, hist = latih(tag, reward_kw, beta_mode, SEED)
        last = hist[-1] if hist else {}
        print(f"  [{tag}] -- selesai (critic_loss_akhir={last.get('critic_loss', 0):.3f} "
             f"beta_akhir={last.get('beta')})", flush=True)

    agregat = {}
    for tag, _, _ in KONDISI:
        actor = muat(tag, SEED)
        fac = fac_dari_actor(actor)
        r = K.satu_run(fac, MODE, SEED, beku=False)
        agregat[tag] = {k: r[k] for k in ("gini", "wait", "trust", "acc") if k in r}
        print(f"  [{tag}] gini={agregat[tag]['gini']:.4f} wait={agregat[tag]['wait']:.1f} "
             f"trust={agregat[tag]['trust']:.3f} acc={agregat[tag]['acc']:.3f}", flush=True)

    d_gini_preset = agregat["sb4x_gap"]["gini"] - agregat["default_gap"]["gini"]
    d_gini_beta = agregat["default_fixed"]["gini"] - agregat["default_gap"]["gini"]
    print(f"\nd_gini (sb4x - default, beta=gap_ratio): {d_gini_preset:+.4f} "
         f"({'preset MEMBANTU' if d_gini_preset < -0.005 else 'preset TAK signifikan membantu'})",
         flush=True)
    print(f"d_gini (fixed - gap_ratio, preset=default): {d_gini_beta:+.4f} "
         f"({'DGR kolaps MERUGIKAN' if d_gini_beta < -0.005 else 'DGR bukan sumber masalah utama'})",
         flush=True)

    common.save_json(dict(n_updates=N_UPDATES, seed=SEED, mode=MODE, agregat=agregat,
                          d_gini_preset=d_gini_preset, d_gini_beta=d_gini_beta),
                     f"uji_ablasi_reward_beta_ev_seed{SEED}.json")
    print(f"\nSAVED -> outputs/uji_ablasi_reward_beta_ev_seed{SEED}.json", flush=True)


if __name__ == "__main__":
    main()
