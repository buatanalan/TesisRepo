"""Uji keselarasan preferensi (BUKAN cuma Gini) -- greedy_queue/greedy_util tidak pernah
melihat preferensi pengguna sama sekali, jadi perbandingan Gini semata TIDAK ADIL utk
menilai apakah PDQN (tugas GANDA: preferensi + pemerataan) "gagal". Metrik tambahan:
- acceptance_rate: dgn trust DIBEKUKAN sama utk semua metode, makin tinggi -> rekomendasi
  makin selaras dgn yg pengguna sendiri inginkan (bukan didorong trust tinggi).
- wait_by_compliance ratio (Objektif 2 tesis, target <=1,2): kelompok patuh TIDAK
  dirugikan dibanding kelompok pilih-sendiri.
Dihitung dari `sim.logs` yg sudah ada (metrics.wait_stats_by_compliance) -- tanpa
instrumentasi baru, tanpa retraining."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import torch
import common
from marl_spklu.experiments import metrics as _metrics
from marl_spklu.rl.pdqn_agent import PDQNInferenceAgent
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.experiments.ablations import constant_trust

DS_4X_30D = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
TRUST_A = 0.5
N_EVAL_SEED = 10

results_A = json.load(open(os.path.join(common.OUTDIR, "pdqn_A_training_results.json")))

def load_qnet(run):
    from marl_spklu.rl.pdqn_policy import PDQNQNetwork
    q_net = PDQNQNetwork(run["obs_dim"], run["N"], n_types=run["n_types"],
                         use_preference=run["use_preference"],
                         pref_feature_mode=run["pref_feature_mode"])
    q_net.load_state_dict(torch.load(run["ckpt"]))
    q_net.eval()
    return q_net

def eval_pref_metrics_pdqn(run, n_eval_seed):
    q_net = load_qnet(run)
    accept_rates, ratios, criterion_ok = [], [], []
    for s in range(n_eval_seed):
        with constant_trust(value=TRUST_A):
            sim = common.fresh_sim(DS_4X_30D)
            random.seed(s); np.random.seed(s)
            agent = PDQNInferenceAgent(q_net, forecaster=FormulaForecaster(),
                                       pref_feature_mode=run["pref_feature_mode"])
            agent.bind_to_sim(sim)
            sim.run(max_steps=sim.max_steps, agent=agent)
        accept_rates.append(float(np.mean([L["complied"] for L in sim.logs])) if sim.logs else 0.0)
        wc = _metrics.wait_stats_by_compliance(sim.logs)
        ratios.append(wc["ratio"])
        criterion_ok.append(wc["criterion_20pct_ok"])
    return accept_rates, ratios, criterion_ok

def eval_pref_metrics_baseline(agent_factory, n_eval_seed):
    accept_rates, ratios, criterion_ok = [], [], []
    for s in range(n_eval_seed):
        with constant_trust(value=TRUST_A):
            sim = common.fresh_sim(DS_4X_30D)
            random.seed(s); np.random.seed(s)
            sim.run(max_steps=sim.max_steps, agent=agent_factory())
        accept_rates.append(float(np.mean([L["complied"] for L in sim.logs])) if sim.logs else 0.0)
        wc = _metrics.wait_stats_by_compliance(sim.logs)
        ratios.append(wc["ratio"])
        criterion_ok.append(wc["criterion_20pct_ok"])
    return accept_rates, ratios, criterion_ok

print("=== PDQN Titik A (3 seed) -- keselarasan preferensi ===")
all_pdqn = {}
for run in results_A:
    acc, rat, ok = eval_pref_metrics_pdqn(run, N_EVAL_SEED)
    rat_valid = [r for r in rat if r is not None]
    print(f"seed={run['seed']}: acceptance_rate={np.mean(acc):.4f} "
         f"wait_by_compliance_ratio={np.mean(rat_valid) if rat_valid else float('nan'):.4f} "
         f"(target<=1.2) frac_ok={np.mean([o for o in ok if o is not None]) if any(o is not None for o in ok) else float('nan'):.2f}")
    all_pdqn[run["seed"]] = dict(acceptance_rates=acc, ratios=rat, criterion_ok=ok)

print()
print("=== greedy_queue -- keselarasan preferensi ===")
acc_gq, rat_gq, ok_gq = eval_pref_metrics_baseline(lambda: GreedyAgent(mode="queue"), N_EVAL_SEED)
rat_gq_valid = [r for r in rat_gq if r is not None]
print(f"acceptance_rate={np.mean(acc_gq):.4f} wait_by_compliance_ratio="
     f"{np.mean(rat_gq_valid) if rat_gq_valid else float('nan'):.4f}")

print()
print("=== greedy_util -- keselarasan preferensi ===")
acc_gu, rat_gu, ok_gu = eval_pref_metrics_baseline(lambda: GreedyAgent(mode="utilization"), N_EVAL_SEED)
rat_gu_valid = [r for r in rat_gu if r is not None]
print(f"acceptance_rate={np.mean(acc_gu):.4f} wait_by_compliance_ratio="
     f"{np.mean(rat_gu_valid) if rat_gu_valid else float('nan'):.4f}")

print()
print("=== RINGKASAN ===")
mean_acc_pdqn = np.mean([np.mean(v["acceptance_rates"]) for v in all_pdqn.values()])
print(f"PDQN   : acceptance_rate={mean_acc_pdqn:.4f}")
print(f"greedy_queue: acceptance_rate={np.mean(acc_gq):.4f}")
print(f"greedy_util : acceptance_rate={np.mean(acc_gu):.4f}")

common.save_json(dict(
    pdqn_per_seed=all_pdqn,
    pdqn_mean_acceptance=float(mean_acc_pdqn),
    greedy_queue_acceptance=acc_gq, greedy_queue_ratio=rat_gq,
    greedy_util_acceptance=acc_gu, greedy_util_ratio=rat_gu,
    trust_used=TRUST_A,
), "pdqn_preference_eval_results.json")
print("DONE")
