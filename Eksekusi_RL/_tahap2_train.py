import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.experiments.ablations import constant_trust

# Sapuan trust statis (preseden paper: mu_hat divariasikan) -- 3 titik merentang 0,4-0,9
# (rendah/tengah/tinggi), mengikuti pola sapuan mu_hat arsip (0,2/0,5/0,8) tapi digeser ke
# rentang yang diminta user. Tiap titik direplikasi PENUH (3 train-seed x 100 update),
# BUKAN dikurangi -- konsistensi antar-seed (kriteria gerbang) harus tetap terukur di
# setiap titik trust, bukan hanya satu.
CONFIG = dict(
    dataset_path=common.DATASET_KANONIK,   # rezim operasi 4x sudah dibekukan sbg SUBSTRAT
    trust_values=[0.4, 0.65, 0.9],
    n_train_seed=3,
    anggaran_chunk=300,     # n_updates -- dicatat, dipakai identik di Tahap 3/4 (naik dari
                            # 100->300 setelah perbaikan GAE C1+C4: spread antar-seed sudah
                            # turun 3-10x tapi belum capai target ketat <=0,005; gini masih
                            # trending turun di update ke-100 pd beberapa seed, indikasi
                            # belum konvergen penuh -- anggaran lbh besar diharapkan menutup
                            # sisa gap tanpa perlu perbaikan struktural lagi)
    k=2,
    rollout_steps=288,      # REVISI diagnosis: 96->288 (3 hari) -- batch regresi kritik
                            # per-update sebelumnya (~40-50 keputusan) terlalu kecil/berisik
                            # (EV thrashing, sempat -603) -- chunk 3x lbh besar utk target
                            # return yg lbh stabil secara statistik.
    lr=1e-4,                # REVISI: 3e-4->1e-4, langkah optimasi lbh kecil/stabil
    vf_coef=0.25,            # REVISI KEDUA (dibalik): percobaan 1.0 TERBUKTI SALAH ARAH --
                            # aktor & kritik berbagi backbone + clip-norm gradien BERSAMA
                            # (max_grad_norm=0.5); vf_coef besar bikin loss kritik
                            # mendominasi anggaran gradien bersama itu, menyisakan nyaris
                            # nol utk aktor (entropi TOTAL beku 1,791->1,792, grad_norm
                            # mentah meledak sampai 608). Diturunkan DI BAWAH default (0.5)
                            # agar aktor dpt porsi anggaran gradien yg wajar.
    ent_coef=0.002,          # dipertahankan (arah sudah benar: turun dari 0,01 asli agar
                            # bonus entropi tak lagi mengunci aktor di distribusi seragam)
)
common.save_json(CONFIG, "02_config_beku.json")
print("CONFIG:", CONFIG, flush=True)

results = []
t0 = time.time()
for tv in CONFIG["trust_values"]:
    for seed in range(CONFIG["n_train_seed"]):
        print(f"=== trust={tv} train_seed={seed} ===", flush=True)
        with constant_trust(value=tv):
            tr = TorchContinuingTrainer(
                CONFIG["dataset_path"], k=CONFIG["k"], rollout_steps=CONFIG["rollout_steps"],
                reward_calc=RewardCalculator.seimbang(), seed=seed, verbose=True,
                log_path=os.path.join(common.OUTDIR, f"02_train_log_t{tv}_seed{seed}.jsonl"),
                lr=CONFIG["lr"], vf_coef=CONFIG["vf_coef"], ent_coef=CONFIG["ent_coef"])
            policy = tr.train(FormulaForecaster(), n_updates=CONFIG["anggaran_chunk"])
        ckpt_path = os.path.join(common.OUTDIR, f"02_pdqn_A_t{tv}_seed{seed}.pt")
        torch.save(policy.state_dict(), ckpt_path)
        results.append(dict(trust_value=tv, seed=seed, ckpt=ckpt_path, history=tr.history,
                            obs_dim=tr.obs_dim, critic_obs_dim=tr.critic_obs_dim, N=tr.N))
        common.save_json(results, "02_training_results.json")   # simpan progresif
        print(f"trust={tv} seed={seed} DONE, elapsed={time.time()-t0:.1f}s", flush=True)

print("ALL DONE, total elapsed", time.time() - t0, flush=True)
