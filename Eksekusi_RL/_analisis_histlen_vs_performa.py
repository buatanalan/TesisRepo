"""Analisis stratifikasi: performa keputusan (complied, wait_time) dikelompokkan
berdasarkan PANJANG RIWAYAT (`hist_len`) yang tersedia di `pref_hist` SAAT keputusan
itu diambil (2026-08-30) -- menjawab dugaan 4 (`pref_lstm` kekurangan sinyal pada
riwayat pendek/kosong) secara empiris langsung dari checkpoint yg SUDAH ADA, TANPA
retrain.

Cara kerja: `MasterPureRolloutAgent._record_pref` dipanggil PERSIS SEKALI per
keputusan (via `on_decision`), SEBELUM entri baru ditambahkan ke deque -- jadi
`len(self._pref_hist.get(user.user_id, []))` pada titik itu = panjang riwayat yg
DILIHAT `pref_lstm` untuk keputusan ybs. Dicatat berurutan PER USER (bukan global),
lalu dicocokkan ke `sim.logs` (yg juga terurut per-user krn satu EV cuma bisa
punya SATU sesi aktif) memakai indeks urutan per-user -- BUKAN indeks global,
krn `sim.logs` terurut berdasar WAKTU SELESAI charging (bisa beda urutan dgn
waktu KEPUTUSAN diambil kalau ada antrean).

Pemakaian:
    python Eksekusi_RL/_analisis_histlen_vs_performa.py 0,1,2 90d master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1
"""
import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.rl.master_pure_hybrid_policy import MasterHybridPPOActor
from marl_spklu.rl.master_pure_trainer import MasterPureRolloutAgent
from marl_spklu.rl.rollout import RewardCalculatorStub
from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.training import _fresh_sim as _fresh_sim_common

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "90d"
TAG_ARM = sys.argv[3] if len(sys.argv) > 3 else "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1"
K_REC = 3
ACTOR_KW = dict(vec_dim=8, bid_hidden=16, pref_d_lstm=8, pref_d_attn=8, station_attn_dim=8,
                pref_feature_mode="_preffeat" in TAG_ARM,
                pref_pair_outcome="_pairout" in TAG_ARM,
                use_station_attn="_noattn" not in TAG_ARM,
                station_feat_dim=(10 if "_evobs" in TAG_ARM else 7))
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])


def muat_policy(seed, n_spklu):
    ckpt = os.path.join(common.OUTDIR, f"{TAG_ARM}_actor_seed{seed}.pt")
    assert os.path.exists(ckpt), f"checkpoint tak ditemukan: {ckpt}"
    pol = MasterHybridPPOActor(n_spklu, **ACTOR_KW)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval()
    return pol


class _InstrumentedAgent:
    """Bungkus rollout agent Hybrid, catat hist_len per-user PER URUTAN keputusan
    (bukan indeks global) sebelum `_record_pref` menambah entri baru."""

    def __init__(self, actor, forecaster, k):
        self.actor = actor
        self.actor.eval()
        self.forecaster = forecaster
        self.k = int(k)
        self.hist_len_seq = defaultdict(list)   # user_id -> [hist_len saat tiap keputusan, berurutan]
        self._roll = None

    def bind_to_sim(self, sim):
        _bb = getattr(self.actor, "backbone", None)
        self._roll = MasterPureRolloutAgent(
            self.actor, sim, RewardCalculatorStub(), self.forecaster,
            noise_std=0.0, k=self.k,
            pref_feature_mode=getattr(_bb, "pref_feature_mode", False),
            pref_pair_outcome=getattr(_bb, "pref_pair_outcome", False))
        self.sim = sim

    def get_recommendation(self, feasible_spklus):
        recs = self._roll.get_recommendation(feasible_spklus)
        self._roll.transitions.clear()
        return recs

    def predict_waits(self, feasible_spklus):
        return self._roll.predict_waits(feasible_spklus)

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        h = self._roll._pref_hist.get(user.user_id)
        self.hist_len_seq[user.user_id].append(0 if h is None else len(h))
        self._roll.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        self._roll.transitions.clear()

    def on_charge_complete(self, user):
        self._roll.on_charge_complete(user)
        self._roll.transitions.clear()


def main():
    n_spklu = len(_fresh_sim_common(K.DS).spklus)
    print(f"horizon={TAG} arm={TAG_ARM} seeds={SEEDS}", flush=True)

    bucket_edges = [0, 1, 3, 6, 10]   # [0], [1-2], [3-5], [6-9], [10]
    bucket_labels = ["0 (kosong)", "1-2", "3-5", "6-9", "10 (penuh)"]

    def bucket_of(h):
        if h == 0:
            return 0
        if h <= 2:
            return 1
        if h <= 5:
            return 2
        if h <= 9:
            return 3
        return 4

    agg_wait = defaultdict(list)
    agg_complied = defaultdict(list)
    n_total_decisions = 0

    for sd in SEEDS:
        policy = muat_policy(sd, n_spklu)
        agent = _InstrumentedAgent(policy, FormulaForecaster(), K_REC)
        sim = common.fresh_sim(K.DS, rekam_deret=True)
        import random
        random.seed(sd); np.random.seed(sd)
        agent.bind_to_sim(sim)
        sim.run(max_steps=sim.max_steps, agent=agent)

        # Cocokkan hist_len (urutan KEPUTUSAN per-user) dgn sim.logs (urutan SELESAI
        # per-user) -- satu EV cuma py 1 sesi aktif, jadi urutan per-user identik.
        per_user_logs = defaultdict(list)
        for l in sim.logs:
            per_user_logs[l["user"]].append(l)

        matched = 0
        for uid, hist_seq in agent.hist_len_seq.items():
            logs_u = per_user_logs.get(uid, [])
            n = min(len(hist_seq), len(logs_u))
            for i in range(n):
                h = hist_seq[i]
                l = logs_u[i]
                b = bucket_of(h)
                agg_complied[b].append(1.0 if l["complied"] else 0.0)
                if l["complied"]:
                    agg_wait[b].append(l["wait_time"])
                matched += 1
        n_total_decisions += matched
        print(f"  seed={sd} keputusan tercocokkan: {matched}", flush=True)

    print(f"\nTotal keputusan tercocokkan (semua seed): {n_total_decisions}\n")
    print(f"{'bucket hist_len':18s} {'n':>7s} {'acc (patuh)':>12s} {'wait (patuh saja)':>20s}")
    for b, label in enumerate(bucket_labels):
        n = len(agg_complied[b])
        if n == 0:
            print(f"{label:18s} {'(kosong)':>7s}")
            continue
        acc = float(np.mean(agg_complied[b]))
        wait_vals = agg_wait[b]
        wait_mean = float(np.mean(wait_vals)) if wait_vals else float("nan")
        print(f"{label:18s} {n:7d} {acc:12.4f} {wait_mean:20.2f}")


if __name__ == "__main__":
    main()
