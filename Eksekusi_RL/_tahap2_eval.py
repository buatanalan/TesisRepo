"""Eval bersih Tahap 2 v3 -- protokol 5 METRIK (lihat Rencana_Tahap2_v3 §3).

Melaporkan Gini SAJA menyesatkan: objektif pemerataan BERTENTANGAN dgn waktu tunggu, dan
acceptance adalah objektif rancangan modul P. Karena itu tiap lengan dinilai atas:
  1. gini_served          -- objektif pemerataan (utama)
  2. acceptance           -- objektif rancangan modul P
  3. mean_wait            -- objektif yang bertentangan dgn (1)
  4. wait_ratio           -- kendala (patuh <= 1,2x default); definisi disamakan dgn
                             harness.metrics.wait_stats_by_compliance
  5. total_served         -- deteksi kolaps

Semua lengan dievaluasi pada REALISASI LINGKUNGAN YANG SAMA per eval-seed (perbandingan
BERPASANGAN) -- variasi lingkungan terkontrol, jadi selisih antar-lengan bisa dibaca.
"""
import sys, os, json, random, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contextlib
import numpy as np
import torch
import common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.experiments.ablations import constant_trust_shadow
from marl_spklu.agents.greedy_agent import GreedyAgent

EVAL_SEEDS = [0, 1, 2]
TRUST = 0.5
DATASET = "scenario_dataset_klaster12_4x.json"


class VirtualWaitForecaster(ForecasterBase):
    def predict(self, spklus, time_now_min=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {sid: 0.0 for sid in spklus}
        return {sid: float(sim.compute_virtual_wait(user, s, time_now_min))
                for sid, s in spklus.items()}


def evaluasi(agent_factory, ds, eval_seed, trust_dinamis=False):
    """`trust_dinamis=True` (Tahap 3): trust TIDAK dibekukan -- kebijakan harus dievaluasi
    di lingkungan yang SAMA dgn tempat ia dilatih, kalau tidak perbandingannya tak sah."""
    ctx = contextlib.nullcontext() if trust_dinamis else constant_trust_shadow(value=TRUST)
    with ctx:
        sim = common.fresh_sim(ds)
        random.seed(eval_seed); np.random.seed(eval_seed)
        sim.run(max_steps=sim.max_steps, agent=agent_factory(sim))

    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
    logs = sim.logs
    w = np.array([L["wait_time"] for L in logs], dtype=float)
    comp = np.array([bool(L["complied"]) for L in logs], dtype=bool)

    # wait_ratio: rata-rata tunggu kelompok PATUH vs kelompok TIDAK patuh (proksi
    # "default"), sesuai semangat harness.wait_stats_by_compliance.
    wr = (float(w[comp].mean() / w[~comp].mean())
          if comp.any() and (~comp).any() and w[~comp].mean() > 1e-9 else float("nan"))

    return dict(
        gini=float(common.gini(served)),
        acceptance=float(comp.mean()) if comp.size else 0.0,
        mean_wait=float(w.mean()) if w.size else 0.0,
        wait_ratio=wr,
        total_served=int(served.sum()),
    )


def muat_lengan(ds):
    """Kumpulkan {label: agent_factory} dari checkpoint Tahap 2 yang tersedia."""
    fc = VirtualWaitForecaster()
    L = {}
    # S0 -- KONDISI NATURAL (tanpa intervensi): tak ada rekomendasi sama sekali, pengguna
    # murni mengikuti preferensinya. Ini pembanding UTAMA tesis ("turunkan Gini dibanding
    # kondisi natural"). Angka S0 dari Tahap 1 (Gini 0,558) TIDAK bisa dipakai di sini --
    # itu `gini_mean` utilisasi sepanjang waktu, metrik BERBEDA dari Gini `total_served`
    # yang dipakai Tahap 2. Harus diukur ulang dgn protokol yang sama.
    L["S0_natural"] = (lambda sim: None)
    # Pembanding Tahap 3: S0 & greedy versi TRUST DINAMIS (sufiks `_t3_` memicu mode itu).
    L["S0_natural_t3_"] = (lambda sim: None)
    for mode in ["queue", "utilization"]:
        L[f"greedy_{mode}_t3_"] = (lambda sim, m=mode: GreedyAgent(mode=m, top_k=2))
    for mode in ["queue", "utilization"]:
        L[f"greedy_{mode}"] = (lambda sim, m=mode: GreedyAgent(mode=m, top_k=2))

    from marl_spklu.rl.policy import HPPOPolicy
    from marl_spklu.rl.p_ppo_policy import PPPOPolicy
    from marl_spklu.rl.rollout import InferenceAgent

    for meta_p in sorted(glob.glob(os.path.join(common.OUTDIR, "t2_*_meta.json"))):
        meta = json.load(open(meta_p))
        stem = os.path.basename(meta_p).replace("_meta.json", "")
        ck = os.path.join(common.OUTDIR, stem + ".pt")
        if not os.path.exists(ck):
            continue
        cls = PPPOPolicy if meta.get("policy_cls") == "PPPOPolicy" else HPPOPolicy
        kw = dict(n_critics=meta.get("n_critics", 1))
        if cls is PPPOPolicy:
            # Dimensi modul preferensi pernah diubah 64 -> 16; checkpoint LAMA tak
            # mencatatnya, jadi fallback ke 64 (nilai saat checkpoint itu dilatih).
            kw["pref_d_lstm"] = meta.get("pref_d_lstm", 64)
            kw["pref_d_attn"] = meta.get("pref_d_attn", 64)
        pol = cls(meta["obs_dim"], meta["critic_obs_dim"], meta["N"], **kw)
        pol.load_state_dict(torch.load(ck))
        pol.eval()
        for k, v in (meta.get("policy_kw") or {}).items():
            setattr(pol, k, v)
        L[stem.replace("t2_", "")] = (
            lambda sim, p=pol: InferenceAgent(p, sim, fc, k=2, epsilon=0.0, threshold=0.20))
    return L


def main():
    """Argumen: `--force` = evaluasi ulang semua lengan (abaikan cache).

    Tanpa `--force`, lengan yang SUDAH ada di `t2_eval_konsolidasi.json` DILEWATI --
    eval satu lengan = 3 simulasi 30-hari, dan jumlah lengan terus bertambah tiap tahap,
    jadi mengulang semuanya tiap kali membuang waktu (>10 menit pada 8 lengan)."""
    force = "--force" in sys.argv
    ds = os.path.join(common.ROOT, DATASET)
    lengan = muat_lengan(ds)

    hasil = {}
    cache_p = os.path.join(common.OUTDIR, "t2_eval_konsolidasi.json")
    if os.path.exists(cache_p) and not force:
        hasil = json.load(open(cache_p)).get("hasil", {})
    baru = [k for k in lengan if k not in hasil]
    print(f"{len(lengan)} lengan total | {len(baru)} perlu dievaluasi "
          f"x {len(EVAL_SEEDS)} eval-seed\n", flush=True)

    for label in baru:
        dinamis = "_t3_" in label
        runs = [evaluasi(lengan[label], ds, sd, trust_dinamis=dinamis) for sd in EVAL_SEEDS]
        hasil[label] = runs
        agg = {k: (float(np.nanmean([r[k] for r in runs])),
                   float(np.nanstd([r[k] for r in runs]))) for k in runs[0]}
        print("%-26s gini=%.4f±%.4f  accept=%.3f±%.3f  wait=%6.1f±%4.1f  "
              "wr=%.2f  served=%d" % (
                  label, agg["gini"][0], agg["gini"][1], agg["acceptance"][0],
                  agg["acceptance"][1], agg["mean_wait"][0], agg["mean_wait"][1],
                  agg["wait_ratio"][0], int(agg["total_served"][0])), flush=True)

    common.save_json(dict(eval_seeds=EVAL_SEEDS, trust=TRUST, hasil=hasil),
                     "t2_eval_konsolidasi.json")
    print("\nSAVED -> t2_eval_konsolidasi.json")


if __name__ == "__main__":
    main()
