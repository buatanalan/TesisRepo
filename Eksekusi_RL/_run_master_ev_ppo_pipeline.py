"""Pelatihan+evaluasi lengan MASTER perspektif-EV, tulang punggung PPOTrainer standar
(marl_spklu/rl/master_ev_ppo_policy.py) -- pengganti `_run_master_ev_pipeline.py`
(trainer custom kritik-per-timestep, terbukti tak stabil 300 chunk penuh).

Dataset SELALU regime 4x. Jalankan (server, latar belakang):
    nohup .venv/Scripts/python.exe Eksekusi_RL/_run_master_ev_ppo_pipeline.py \
        > Eksekusi_RL/outputs/master_ev_ppo_pipeline.log 2>&1 &
"""
import sys, os, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.rl.master_ev_ppo_policy import (MasterEVPPOTrainer, MasterEVPPOPolicy,
                                                MasterEVPPOPrefPolicy, MasterEVPPOInferenceAgent)
from marl_spklu.rl.master_ev_ppo_acc import MasterEVPPOAccRolloutAgent, STREAM_ACCURACY
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.agents.greedy_agent import GreedyAgent
from scipy.stats import wilcoxon, spearmanr

T0 = time.time()
def elapsed():
    return f"{time.time()-T0:.1f}s"

p = argparse.ArgumentParser()
p.add_argument("--n-train-seed", type=int, default=3)
p.add_argument("--n-eval-seed", type=int, default=10)
p.add_argument("--n-updates", type=int, default=300)
p.add_argument("--rollout-steps", type=int, default=96)
p.add_argument("--k", type=int, default=3)
p.add_argument("--n-critics", type=int, default=1)
p.add_argument("--beta-mode", type=str, default="fixed", choices=["fixed", "gap_ratio"],
              help="cara gabung advantage antar-aliran kritik (n_critics>1). 'fixed' (BAKU "
                   "-- juga default PPOTrainer sendiri, bobot seragam 1/K SEPANJANG waktu, "
                   "TIDAK adaptif) | 'gap_ratio' (Dynamic Gradient Re-weighting MASTER "
                   "sesungguhnya -- bobot menyesuaikan diri thd aliran yg paling tertinggal). "
                   "SEMUA eksperimen K2/K3 sebelum ini diam-diam memakai 'fixed' -- DGR "
                   "TAK PERNAH aktif walau n_critics>1 (lihat diagnosis beta terpaku 1/K).")
p.add_argument("--beta-sigma", type=float, default=0.1,
              help="suhu softmax utk beta_mode='gap_ratio' (PPOTrainer.beta_sigma, BAKU "
                   "0.1 -- juga default PPOTrainer sendiri). BAKU 0.1 TERLALU TAJAM: gap "
                   "diclip [0,10] tapi z=gap/sigma bisa smp 100 -> beta kolaps nyaris "
                   "one-hot antar-chunk (diamati smoke-test & diagnosis kolaps seed 0 K3 "
                   "90d acc1 -- entropi kebijakan runtuh permanen stlh lonjakan grad_norm "
                   "beriringan dgn swing beta ekstrem). Naikkan (mis. 1.0-2.0) utk "
                   "melunakkan transisi bobot antar-stream.")
p.add_argument("--pref", action="store_true",
              help="tambahkan modul preferensi PDQN (MasterEVPPOPrefPolicy) -- pengujian "
                   "ULANG hipotesis 'P gagal krn identitas-ambigu di perspektif stasiun', "
                   "kini di unit-agen-EV + kritik V(s) stabil")
p.add_argument("--pref-feature-mode", action="store_true",
              help="riwayat preferensi sbg vektor fitur (bukan one-hot identitas) -- "
                   "hanya berlaku bila --pref diberikan")
p.add_argument("--no-hist", action="store_true",
              help="ABLASI: matikan kontribusi hist_lstm (c_t) -- utk isolasi dugaan "
                   "hist_lstm & pref_lstm saling mengganggu (satu-satunya hasil P yg "
                   "pernah positif terjadi SEBELUM hist_lstm ada). Hanya bermakna dgn "
                   "--pref; tanpa --pref cuma menghapus proxy-trust tanpa gantinya.")
p.add_argument("--dataset", type=str, default="4x",
              help="'4x' (BAKU, 30 hari) | '1x' (DATASET_KANONIK, TAK sepadan) | path eksplisit "
                   "(mis. scenario_dataset_klaster12_4x_90d.json -- WAJIB sertakan --horizon 90d)")
p.add_argument("--horizon", type=str, default="30d",
              help="penanda tag checkpoint/hasil -- WAJIB diubah ('90d') saat --dataset menunjuk "
                   "dataset 90-hari, supaya tak menimpa diam-diam checkpoint 30-hari yg sudah ada "
                   "(tag 'master_ev_ppo*' polos khusus utk horizon baku 30d)")
p.add_argument("--alpha-trust", type=float, default=0.0,
              help="bobot suku shaping trust eksplisit (RewardCalculator.alpha_trust, "
                   "BAKU MATI=0.0 -- reward += alpha_trust*(trust_baru-trust_lama) tiap "
                   "sesi PATUH selesai). >0 WAJIB tag terpisah, lihat di bawah.")
p.add_argument("--alpha-accept", type=float, default=0.0,
              help="bobot suku shaping acceptance eksplisit (RewardCalculator.alpha_accept, "
                   "BAKU MATI=0.0). SIMETRIS (+patuh/-tolak, SEGERA -- beda dari wait_reward "
                   "yg hanya-positif & tertunda). Diagnosis riwayat training membuktikan "
                   "acceptance terkikis 0,68->0,39 tanpa suku ini. >0 WAJIB tag terpisah.")
p.add_argument("--alpha-equity", type=float, default=0.0,
              help="bobot suku pemerataan LOKAL eksplisit (RewardCalculator."
                   "local_equity_reward, BAKU MATI=0.0) -- ALTERNATIF thd Gini global "
                   "(alpha_gini): Markovian sungguhan, individual, teratribusi ke stasiun "
                   "yg DIPILIH keputusan ini vs rata2 populasi saat itu (bukan statistik "
                   "SELURUH populasi lintas-langkah spt Gini). WAJIB --n-critics 4 (aliran "
                   "STREAM_EQUITY terpisah dari GLOBAL3 -- diagnosis 2026-08-21 acceptance "
                   "vs gini 'tarik-menarik' saat berbagi 1 head kritik). >0 WAJIB tag "
                   "terpisah, lihat Diagnosis_Gini_sbg_Reward 2026-08-22.")
p.add_argument("--gini-mode", type=str, default="dense", choices=["dense", "terminal"],
              help="'dense' (BAKU -- gini_reward per-langkah/delta, perilaku LAMA) | "
                   "'terminal' (Opsi 1, APROKSIMASI per-chunk -- lihat catatan "
                   "MasterEVPPOTrainer.__init__: proposal asli 'reward 0 sepanjang "
                   "episode, hanya -Gini di t=T' butuh episode PENUH 30-90 hari "
                   "tersimpan utuh, bertentangan dgn training chunked rollout_steps=96 "
                   "-- di sini 'episode' didekati per-CHUNK, level absolut, HANYA "
                   "transisi terakhir chunk. BUKAN implementasi murni proposal asli).")
p.add_argument("--cmdp-epsilon", type=float, default=None,
              help="Opsi 3 (CMDP): target ambang Gini (constraint Gini<=epsilon). WAJIB "
                   "diberikan bila --cmdp-lr-dual>0. Referensi: gini K3 base sudah "
                   "capai 0,046-0,072 -- epsilon di sekitar situ (mis. 0,06-0,08) masuk "
                   "akal sbg target awal.")
p.add_argument("--cmdp-lr-dual", type=float, default=0.0,
              help="Opsi 3 (CMDP): laju dual ascent utk `alpha_gini` sbg pengali "
                   "Lagrange (BAKU MATI=0.0 -- alpha_gini statis spt biasa/preset). "
                   ">0 -> alpha_gini MULAI DARI 0 (KKT), diperbarui tiap chunk "
                   "`alpha_gini += lr*(gini_terukur-epsilon)`, clip >=0. >0 WAJIB "
                   "tag terpisah & --cmdp-epsilon eksplisit.")
p.add_argument("--use-station-attn", action="store_true",
              help="Self-attention SATU-KEPALA antar-stasiun jalur AKTOR (2026-08-23, "
                   "BAKU MATI) -- StationSelfAttention, N baris masuk N baris keluar "
                   "(beda dari AttentionPooling kritik yg meringkas ke 1 vektor). "
                   "Gerbang nol-awal (spt pref_gate) -- aman dinyalakan pada checkpoint "
                   "baru tanpa mengganggu perilaku awal.")
p.add_argument("--station-attn-dim", type=int, default=None,
              help="Dimensi proyeksi Q/K/V/out StationSelfAttention (BAKU None -> ikut "
                   "--hidden, biasanya 64 -> 4 proyeksi Linear(64,64) = 16.640 parameter, "
                   "hampir 3x StationEncoder itu sendiri 5.888. DIDUGA penyebab "
                   "volatilitas gini & kolaps determinisme pada eksperimen pertama "
                   "(anggaran 300-chunk sama dgn baseline, tapi kapasitas jauh lebih "
                   "besar). Kecilkan eksplisit (mis. 16) utk uji dugaan ini. Hanya "
                   "berlaku bila --use-station-attn diberikan.")
p.add_argument("--reward-preset", type=str, default="raw", choices=["raw", "seimbang4x"],
              help="'raw' (BAKU -- konstruktor RewardCalculator() MENTAH, alpha_wait=1.0 "
                   "alpha_gini=0.5 use_delta_gini=False -- TAK PERNAH dikalibrasi utk rezim "
                   "4x, dipertahankan hanya demi kompatibilitas mundur checkpoint lama) | "
                   "'seimbang4x' (RewardCalculator.seimbang4x() -- preset TERKALIBRASI utk "
                   "rezim 4x, alpha_wait=0.0046 alpha_gini=2.6019 alpha_flock=0.0208 "
                   "use_delta_gini=True. Diagnosis 2026-08-21: 'raw' membuat gini level "
                   "ABSOLUT nyaris-konstan tiap keputusan -- shg praktis tak jadi sinyal "
                   "belajar; 'seimbang4x' memakai DELTA-gini spt didesain).")
p.add_argument("--forecaster", type=str, default="formula", choices=["formula", "vwf"],
              help="'formula' (BAKU, FormulaForecaster -- kasar, rata2 tetap per konektor) "
                   "| 'vwf' (VirtualWaitForecaster -- basis DISAMAKAN dgn eta_norm yg dilihat "
                   "aktor, sim.compute_virtual_wait; diagnosis event-trust menemukan celah "
                   "besar antara keduanya sbg penyebab dominan erosi trust RL). >0 WAJIB tag "
                   "terpisah, lihat di bawah.")
p.add_argument("--alpha-acc", type=float, default=0.0,
              help="bobot suku shaping AKURASI-JANJI eksplisit (RewardCalculator."
                   "accuracy_reward, STREAM_ACCURACY, BAKU MATI=0.0). Diagnosis "
                   "2026-08-23 (_diagnosis_rec_activity_vs_deltaW.py): |Delta W| trip "
                   "patuh melonjak 5,8x saat rec_activity tinggi -- 2 percobaan "
                   "memperbaiki FORECASTER gagal (lihat kalibrasi_congestion_aware_vwf."
                   "json), jadi suku ini menghukum KEBIJAKAN, bukan janji. WAJIB "
                   "--n-critics 5 (rollout_agent_cls otomatis dialihkan ke "
                   "MasterEVPPOAccRolloutAgent). >0 WAJIB tag terpisah.")
p.add_argument("--tau-acc", type=float, default=None,
              help="ambang |Delta W| (menit) sebelum accuracy_reward mulai menghukum -- "
                   "baku None -> DELTAW_TOL_HIGH (marl_spklu/env/user.py), ambang PERSIS "
                   "yg sama dipakai User.update_trust menghukum trust.")
p.add_argument("--overwrite", action="store_true",
              help="WAJIB diberikan bila `eval_results.json` yang SUDAH ADA untuk tag "
                   "ini punya `n_updates` LEBIH BESAR dari run ini (mis. hasil serius "
                   "300-update ditimpa smoke-test 5-update) -- pengaman ditambahkan "
                   "2026-08-23 setelah checkpoint eksperimen utama nyaris hilang krn "
                   "tabrakan tag. Tanpa ini, pipeline BERHENTI sebelum menulis "
                   "eval_results.json (training & checkpoint TETAP jalan/tersimpan -- "
                   "hanya file ringkasan evaluasi yang dilindungi).")
p.add_argument("--pref-hist-k", type=int, default=None,
              help="ABLASI: panjang jendela riwayat P (baku None=PDQN_HIST_K=10, sama "
                   "dgn lengan +P lain & PDQN diskrit). Diperkecil (mis. 5, disamakan "
                   "hist_lstm) utk uji dugaan jendela 10 terlalu didominasi padding-nol "
                   "bagi pengguna beriwayat pendek. TERISOLASI dari PDQN_HIST_K bersama.")
args = p.parse_args()

assert not (args.pref_feature_mode and not args.pref), "--pref-feature-mode butuh --pref"
assert args.alpha_acc == 0.0 or args.n_critics == 5, (
    "--alpha-acc>0 butuh --n-critics 5 (STREAM_ACCURACY, lihat master_ev_ppo_acc.py)")
ROLLOUT_AGENT_CLS = MasterEVPPOAccRolloutAgent if args.alpha_acc != 0.0 else None
POLICY_CLS = MasterEVPPOPrefPolicy if args.pref else MasterEVPPOPolicy
POLICY_KW = dict(pref_feature_mode=args.pref_feature_mode) if args.pref else dict()
if args.no_hist:
    POLICY_KW["use_hist"] = False
if args.use_station_attn:
    POLICY_KW["use_station_attn"] = True
    if args.station_attn_dim is not None:
        POLICY_KW["station_attn_dim"] = args.station_attn_dim

_DATASET_4X = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
if args.dataset == "4x":
    DATASET = _DATASET_4X
    assert os.path.exists(DATASET), f"dataset 4x tak ditemukan: {DATASET}"
elif args.dataset == "1x":
    DATASET = common.DATASET_KANONIK
    print(f"[{elapsed()}] !! PERINGATAN: regime 1x -- hasil TAK SEPADAN.", flush=True)
else:
    DATASET = args.dataset
    if not os.path.isabs(DATASET) and not os.path.exists(DATASET):
        DATASET = os.path.join(common.ROOT, DATASET)
    assert os.path.exists(DATASET), f"dataset tak ditemukan: {DATASET}"
    assert args.horizon != "30d", (
        "--dataset kustom diberikan tapi --horizon masih baku ('30d') -- kemungkinan besar "
        "kekeliruan (checkpoint akan bertabrakan diam-diam dgn hasil 30d yg sudah ada). "
        "Sertakan --horizon eksplisit, mis. --horizon 90d.")

_horizon_suffix = "" if args.horizon == "30d" else f"_{args.horizon}"
_trust_suffix = "" if args.alpha_trust == 0.0 else f"_trust{args.alpha_trust:g}"
_accept_suffix = "" if args.alpha_accept == 0.0 else f"_acc{args.alpha_accept:g}"
_equity_suffix = "" if args.alpha_equity == 0.0 else f"_eq{args.alpha_equity:g}"
_fc_suffix = "" if args.forecaster == "formula" else f"_{args.forecaster}"
_rw_suffix = "" if args.reward_preset == "raw" else f"_{args.reward_preset}"
_critics_suffix = "" if args.n_critics == 1 else f"_K{args.n_critics}"
_beta_suffix = "_gap" if args.beta_mode == "gap_ratio" else ""
_sigma_suffix = ("" if args.beta_sigma == 0.1 else f"_sig{args.beta_sigma:g}")
_hist_suffix = "_nohist" if args.no_hist else ""
_prefk_suffix = "" if args.pref_hist_k is None else f"_prefk{args.pref_hist_k}"
_gini_mode_suffix = "" if args.gini_mode == "dense" else f"_{args.gini_mode}"
_cmdp_suffix = "" if args.cmdp_lr_dual == 0.0 else f"_cmdpE{args.cmdp_epsilon:g}lr{args.cmdp_lr_dual:g}"
_acc_suffix = "" if args.alpha_acc == 0.0 else f"_accW{args.alpha_acc:g}"
_stattn_suffix = (("_stattn" + (f"d{args.station_attn_dim}" if args.station_attn_dim is not None else ""))
                  if args.use_station_attn else "")
_suffix = (("_pref_feat" if args.pref_feature_mode else ("_pref" if args.pref else ""))
          + _hist_suffix + _prefk_suffix + _trust_suffix + _accept_suffix + _equity_suffix
          + _acc_suffix + _gini_mode_suffix + _cmdp_suffix + _stattn_suffix
          + _fc_suffix + _rw_suffix + _critics_suffix + _beta_suffix + _sigma_suffix
          + _horizon_suffix)
TAG_ARM = "master_ev_ppo" + _suffix
print(f"[{elapsed()}] Dataset: {DATASET}", flush=True)
print(f"[{elapsed()}] Lengan: tag={TAG_ARM} (perspektif-EV, PPOTrainer standar, V(s) atensi tunggal, "
     f"pref={args.pref} pref_feature_mode={args.pref_feature_mode} alpha_trust={args.alpha_trust} "
     f"forecaster={args.forecaster})", flush=True)
print(f"[{elapsed()}] Anggaran: n_updates={args.n_updates} rollout_steps={args.rollout_steps} "
     f"k={args.k} n_critics={args.n_critics}", flush=True)


def make_forecaster():
    from marl_spklu.rl.forecaster import FormulaForecaster, VirtualWaitForecaster
    return VirtualWaitForecaster() if args.forecaster == "vwf" else FormulaForecaster()


def make_reward_calc():
    if args.reward_preset == "seimbang4x":
        return RewardCalculator.seimbang4x(alpha_trust=args.alpha_trust,
                                           alpha_accept=args.alpha_accept,
                                           alpha_equity=args.alpha_equity,
                                           alpha_acc=args.alpha_acc, tau_acc=args.tau_acc)
    return RewardCalculator(alpha_trust=args.alpha_trust, alpha_accept=args.alpha_accept,
                            alpha_equity=args.alpha_equity,
                            alpha_acc=args.alpha_acc, tau_acc=args.tau_acc)


def train_one(seed, tag):
    rc = make_reward_calc()
    tr = MasterEVPPOTrainer(DATASET, rollout_steps=args.rollout_steps, seed=seed, verbose=False,
                            reward_calc=rc, k=args.k, n_critics=args.n_critics,
                            policy_cls=POLICY_CLS, policy_kw=POLICY_KW,
                            pref_hist_k=args.pref_hist_k, beta_mode=args.beta_mode,
                            beta_sigma=args.beta_sigma, gini_mode=args.gini_mode,
                            cmdp_epsilon=args.cmdp_epsilon, cmdp_lr_dual=args.cmdp_lr_dual,
                            rollout_agent_cls=ROLLOUT_AGENT_CLS)
    policy = tr.train(make_forecaster(), n_updates=args.n_updates)
    ckpt = os.path.join(common.OUTDIR, f"{tag}_actor_seed{seed}.pt")
    torch.save(policy.state_dict(), ckpt)
    return dict(seed=seed, ckpt=ckpt, history=tr.history)


def load_policy(ckpt_path, n_spklu):
    pol = POLICY_CLS(n_spklu, n_critics=args.n_critics, **POLICY_KW)
    pol.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    pol.eval()
    return pol


def eval_policy_gini(ckpt_path, dataset_path, n_eval_seed, k):
    sim0 = common.fresh_sim(dataset_path)
    pol = load_policy(ckpt_path, len(sim0.spklus))
    ginis = []
    for s in range(n_eval_seed):
        sim = common.fresh_sim(dataset_path)
        random.seed(s); np.random.seed(s)
        agent = MasterEVPPOInferenceAgent(pol, sim, make_forecaster(), k=k,
                                          pref_hist_k=args.pref_hist_k,
                                          rollout_agent_cls=ROLLOUT_AGENT_CLS)
        sim.run(max_steps=sim.max_steps, agent=agent)
        served = np.array([sp.total_served for sp in sim.spklus.values()], float)
        ginis.append(common.gini(served))
    return ginis


# ---------------------------------------------------------------------------
# DIAGNOSIS akurasi-janji (2026-08-23): rho(rec_activity, |Delta W|) + perbandingan
# kuartil, PERSIS metodologi `_diagnosis_rec_activity_vs_deltaW.py` -- dibangun LANGSUNG
# ke pipeline ini (bukan skrip ad-hoc terpisah) supaya SETIAP checkpoint yang dilatih
# di sini otomatis punya angka pembanding yang sama, tanpa perlu diulang manual.
# Acuan H1a (`eq1_vwf_seimbang4x_K4_gap_sig1`, forecaster=vwf): rho=0,359, |DW|
# rendah=4,39 mnt, |DW| tinggi=25,58 mnt, rasio=5,83x (n=17.657, 5 seed).
# ---------------------------------------------------------------------------
def diagnose_accuracy(ckpt_path, dataset_path, n_diag_seed, k):
    from marl_spklu.rl.master_ev_ppo_policy import MasterEVPPORolloutAgent
    sim0 = common.fresh_sim(dataset_path)
    pol = load_policy(ckpt_path, len(sim0.spklus))
    RolloutCls = ROLLOUT_AGENT_CLS or MasterEVPPORolloutAgent

    side, records = {}, []
    orig_decide = RolloutCls.on_decision
    def patched_decide(self, user, chosen_spklu_id, recs, feasible_spklus):
        if self._pending is not None:
            side[id(self._pending[0])] = self._pending[5]
        return orig_decide(self, user, chosen_spklu_id, recs, feasible_spklus)
    RolloutCls.on_decision = patched_decide

    orig_complete = RolloutCls.on_charge_complete
    def patched_complete(self, user):
        tr = self._user_trip_tr.get(user.user_id)
        if tr is not None and tr.complied:
            rrc = side.pop(id(tr), None)
            if rrc is not None:
                records.append((rrc, user.wait_time - tr.disp_estwait))
        return orig_complete(self, user)
    RolloutCls.on_charge_complete = patched_complete

    for s in range(n_diag_seed):
        sim = common.fresh_sim(dataset_path)
        random.seed(s); np.random.seed(s)
        agent = MasterEVPPOInferenceAgent(pol, sim, make_forecaster(), k=k,
                                          pref_hist_k=args.pref_hist_k,
                                          rollout_agent_cls=ROLLOUT_AGENT_CLS)
        sim.run(max_steps=sim.max_steps, agent=agent)

    RolloutCls.on_decision = orig_decide
    RolloutCls.on_charge_complete = orig_complete

    if not records:
        return dict(n=0)
    rrc = np.array([r[0] for r in records], dtype=float)
    dw = np.array([r[1] for r in records], dtype=float)
    adw = np.abs(dw)
    rho, p_rho = spearmanr(rrc, adw)
    p25, p75 = np.percentile(rrc, [25, 75])
    low, high = adw[rrc <= p25], adw[rrc >= p75]
    return dict(n=len(records), rho=float(rho), p_rho=float(p_rho),
               deltaW_abs_mean=float(adw.mean()), deltaW_abs_median=float(np.median(adw)),
               deltaW_abs_low_rec_activity=float(low.mean()) if len(low) else None,
               deltaW_abs_high_rec_activity=float(high.mean()) if len(high) else None,
               ratio_high_low=float(high.mean() / max(low.mean(), 1e-9)) if len(low) and len(high) else None,
               frac_deltaW_over10_low=float((low > 10).mean()) if len(low) else None,
               frac_deltaW_over10_high=float((high > 10).mean()) if len(high) else None)


def eval_baseline_gini(dataset_path, agent_factory, n_eval_seed):
    ginis = []
    for s in range(n_eval_seed):
        sim = common.fresh_sim(dataset_path)
        random.seed(s); np.random.seed(s)
        sim.run(max_steps=sim.max_steps, agent=agent_factory())
        served = np.array([sp.total_served for sp in sim.spklus.values()], float)
        ginis.append(common.gini(served))
    return ginis


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def pick_median(results, dataset_path, n_eval_seed, k):
    per_seed = {}
    for run in results:
        per_seed[run["seed"]] = eval_policy_gini(run["ckpt"], dataset_path, n_eval_seed, k)
    order = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))
    return per_seed[order[len(order) // 2]], per_seed


print(f"[{elapsed()}] === PELATIHAN MasterEV-PPO ({args.n_train_seed} seed) ===", flush=True)
results = []
results_path = f"{TAG_ARM}_training_results.json"
existing = {}
try:
    import json
    with open(os.path.join(common.OUTDIR, results_path), encoding="utf-8") as f:
        for row in json.load(f):
            existing[row["seed"]] = row
    print(f"[{elapsed()}] {len(existing)} seed sudah selesai sebelumnya (resume)", flush=True)
except FileNotFoundError:
    pass

for seed in range(args.n_train_seed):
    if seed in existing:
        print(f"[{elapsed()}]   seed={seed} -- SKIP (checkpoint sudah ada)", flush=True)
        results.append(existing[seed])
        continue
    print(f"[{elapsed()}]   seed={seed} -- mulai training", flush=True)
    row = train_one(seed, TAG_ARM)
    print(f"[{elapsed()}]   seed={seed} -- SELESAI", flush=True)
    results.append(row)
    common.save_json(results, results_path)
print(f"[{elapsed()}] Pelatihan selesai ({len(results)} seed)", flush=True)

print(f"[{elapsed()}] === Evaluasi vs greedy_queue/greedy_util ({args.n_eval_seed} seed) ===", flush=True)
gq = eval_baseline_gini(DATASET, lambda: GreedyAgent(mode="queue"), args.n_eval_seed)
gu = eval_baseline_gini(DATASET, lambda: GreedyAgent(mode="utilization"), args.n_eval_seed)
ginis_policy, per_seed = pick_median(results, DATASET, args.n_eval_seed, args.k)
spread = float(max(np.mean(v) for v in per_seed.values())
              - min(np.mean(v) for v in per_seed.values()))
w_gq = wilcoxon(ginis_policy, gq)
w_gu = wilcoxon(ginis_policy, gu)
d_gq = cohens_d(ginis_policy, gq)
d_gu = cohens_d(ginis_policy, gu)

print(f"[{elapsed()}] MasterEV-PPO={np.mean(ginis_policy):.4f}  greedy_queue={np.mean(gq):.4f}  "
     f"greedy_util={np.mean(gu):.4f}", flush=True)
print(f"[{elapsed()}] p_vs_gq={w_gq.pvalue:.4f} d={d_gq:+.3f}  "
     f"p_vs_gu={w_gu.pvalue:.4f} d={d_gu:+.3f}  spread_antar_seed={spread:.4f}", flush=True)

# Diagnosis akurasi-janji (rec_activity vs |Delta W|) -- checkpoint seed MEDIAN (sama
# yg dipakai ginis_policy di atas), n_diag_seed sama dgn n_eval_seed utk data cukup.
median_seed = sorted(per_seed.keys(), key=lambda sd: np.mean(per_seed[sd]))[len(per_seed) // 2]
median_ckpt = next(r["ckpt"] for r in results if r["seed"] == median_seed)
print(f"[{elapsed()}] === Diagnosis akurasi-janji (seed={median_seed}, "
     f"{args.n_eval_seed} seed) ===", flush=True)
diag = diagnose_accuracy(median_ckpt, DATASET, args.n_eval_seed, args.k)
if diag.get("n", 0) > 0:
    print(f"[{elapsed()}] n={diag['n']} rho(rec_act,|DW|)={diag['rho']:+.4f} (p={diag['p_rho']:.1e}) "
         f"|DW| rendah={diag['deltaW_abs_low_rec_activity']:.2f} tinggi="
         f"{diag['deltaW_abs_high_rec_activity']:.2f} rasio={diag['ratio_high_low']:.2f}x "
         f"(acuan H1a: rho=+0,359 rendah=4,39 tinggi=25,58 rasio=5,83x)", flush=True)
else:
    print(f"[{elapsed()}] diagnosis: 0 trip patuh terekam (cek k/forecaster)", flush=True)

_eval_out_path = os.path.join(common.OUTDIR, f"{TAG_ARM}_eval_results.json")
if os.path.exists(_eval_out_path) and not args.overwrite:
    import json as _json
    _existing_eval = _json.load(open(_eval_out_path, encoding="utf-8"))
    _existing_nu = (_existing_eval.get("config") or {}).get("n_updates")
    if _existing_nu is not None and _existing_nu > args.n_updates:
        raise SystemExit(
            f"[{elapsed()}] MENOLAK menimpa {_eval_out_path}: n_updates run ini "
            f"({args.n_updates}) LEBIH KECIL dari yang sudah tersimpan ({_existing_nu}) "
            f"-- kemungkinan besar run smoke-test yang tak sengaja menimpa hasil serius "
            f"(persis insiden 2026-08-23, checkpoint eksperimen utama nyaris hilang). "
            f"Training & checkpoint SUDAH tersimpan dgn aman di atas -- ulangi HANYA "
            f"perintah ini dgn --overwrite bila memang bermaksud menimpa.")

common.save_json(dict(
    gini_policy=ginis_policy, gini_greedy_queue=gq, gini_greedy_util=gu,
    mean_policy=float(np.mean(ginis_policy)), mean_gq=float(np.mean(gq)), mean_gu=float(np.mean(gu)),
    p_vs_gq=float(w_gq.pvalue), p_vs_gu=float(w_gu.pvalue),
    cohens_d_vs_gq=d_gq, cohens_d_vs_gu=d_gu, spread_antar_seed=spread,
    per_seed_means={str(k): float(np.mean(v)) for k, v in per_seed.items()},
    diagnosis_akurasi_janji=diag,
    diagnosis_acuan_H1a=dict(rho=0.359, deltaW_abs_low_rec_activity=4.39,
                             deltaW_abs_high_rec_activity=25.58, ratio_high_low=5.83,
                             sumber="master_ev_ppo_eq1_vwf_seimbang4x_K4_gap_sig1, n=17657"),
    config=dict(n_train_seed=args.n_train_seed, n_eval_seed=args.n_eval_seed,
               n_updates=args.n_updates, rollout_steps=args.rollout_steps, k=args.k,
               n_critics=args.n_critics, pref=args.pref,
               pref_feature_mode=args.pref_feature_mode, horizon=args.horizon,
               alpha_trust=args.alpha_trust, alpha_accept=args.alpha_accept,
               alpha_equity=args.alpha_equity, alpha_acc=args.alpha_acc, tau_acc=args.tau_acc,
               forecaster=args.forecaster, beta_mode=args.beta_mode,
               beta_sigma=args.beta_sigma, reward_preset=args.reward_preset,
               gini_mode=args.gini_mode, cmdp_epsilon=args.cmdp_epsilon,
               cmdp_lr_dual=args.cmdp_lr_dual, dataset=DATASET)),
    f"{TAG_ARM}_eval_results.json")

print(f"[{elapsed()}] === SEMUA SELESAI (MasterEV-PPO) ===", flush=True)
