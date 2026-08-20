"""Diagnosis MEKANISME trust per-EVENT dan per-EV -- kondisi apa yang membuat trust naik
(zona reward, |actual-est|<=TOL_LOW) vs turun (zona penalti, signed>=TOL_HIGH, mode
`TRUST_PENALTY_MODE="signed"` -- HANYA under-promise/actual>est yang dihukum, over-
estimate TIDAK, lihat `User.update_trust`) vs netral (mayoritas, tak ada perubahan).

Sumber data: `Simulator.logs` (diisi tiap sesi charging selesai, field `est_wait`/
`wait_time`/`complied`/`trust_after`/`spklu`/`user`/`hari`) -- TIDAK perlu hook baru,
seluruh informasi yg dibutuhkan sudah dicatat simulator sendiri.

Pemakaian:
    python _diagnosis_trust_events.py greedy_queue
    python _diagnosis_trust_events.py master_ev_ppo <path_ckpt_actor.pt>
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import common
from marl_spklu.env.user import DELTAW_TOL_LOW, DELTAW_TOL_HIGH, TRUST_PENALTY_MODE
from marl_spklu.agents.greedy_agent import GreedyAgent

DS = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
SEED = 0


def classify(signed, delta_w):
    if delta_w <= DELTAW_TOL_LOW:
        return "reward"
    trigger = signed if TRUST_PENALTY_MODE == "signed" else delta_w
    if trigger >= DELTAW_TOL_HIGH:
        return "penalty"
    return "neutral"


def run_and_collect(fac, ds, seed):
    sim = common.fresh_sim(ds)
    random.seed(seed); np.random.seed(seed)
    agent = fac(sim)
    sim.run(max_steps=sim.max_steps, agent=agent)
    rows = []
    per_user_count = {}
    for e in sim.logs:
        if not e.get("complied"):
            continue
        est = float(e["est_wait"]); act = float(e["wait_time"])
        signed = act - est
        delta_w = abs(signed)
        zone = classify(signed, delta_w)
        uid = e["user"]
        per_user_count[uid] = per_user_count.get(uid, 0) + 1
        rows.append(dict(zone=zone, signed=signed, delta_w=delta_w, spklu=e["spklu"],
                         trust_after=float(e["trust_after"]), visit_no=per_user_count[uid],
                         user=uid, hari=e["hari"], est=est, act=act))
    return rows


def summarize(rows, label):
    print(f"\n=== {label} ===")
    n = len(rows)
    print(f"total sesi PATUH (yg trust-nya bisa berubah): {n}")
    for zone in ("reward", "neutral", "penalty"):
        sub = [r for r in rows if r["zone"] == zone]
        frac = len(sub) / n if n else 0.0
        print(f"  zona {zone:8s}: {len(sub):5d} ({frac*100:5.1f}%)"
             + (f"  |actual-est| rata2={np.mean([r['delta_w'] for r in sub]):.2f} menit"
                if sub else ""))

    # --- per stasiun: proporsi zona ---
    print("  --- per stasiun ---")
    stations = sorted(set(r["spklu"] for r in rows))
    for sid in stations:
        sub = [r for r in rows if r["spklu"] == sid]
        if not sub:
            continue
        rw = sum(1 for r in sub if r["zone"] == "reward") / len(sub)
        pn = sum(1 for r in sub if r["zone"] == "penalty") / len(sub)
        bias = np.mean([r["signed"] for r in sub])   # + = actual > est (under-promise)
        print(f"    {sid:12s} n={len(sub):4d}  reward={rw*100:4.1f}%  penalti={pn*100:4.1f}%  "
             f"bias(signed) rata2={bias:+6.2f} menit")

    # --- per urutan kunjungan (pertama vs berulang) ---
    print("  --- per urutan kunjungan pengguna (1=pertama kali) ---")
    max_visit = max((r["visit_no"] for r in rows), default=1)
    for v in range(1, min(max_visit, 6) + 1):
        sub = [r for r in rows if r["visit_no"] == v]
        if not sub:
            continue
        rw = sum(1 for r in sub if r["zone"] == "reward") / len(sub)
        pn = sum(1 for r in sub if r["zone"] == "penalty") / len(sub)
        print(f"    kunjungan-ke-{v}: n={len(sub):4d}  reward={rw*100:4.1f}%  penalti={pn*100:4.1f}%")

    # --- lintasan trust per pengguna: pengguna dgn >=3 kunjungan, trust pertama vs terakhir ---
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user"], []).append(r)
    multi = {u: sorted(rs, key=lambda r: r["visit_no"]) for u, rs in by_user.items() if len(rs) >= 3}
    if multi:
        deltas = [rs[-1]["trust_after"] - rs[0]["trust_after"] for rs in multi.values()]
        print(f"  --- pengguna dgn >=3 kunjungan (n={len(multi)}) ---")
        print(f"    delta trust (kunjungan-terakhir - kunjungan-pertama): "
             f"rata2={np.mean(deltas):+.3f}  median={np.median(deltas):+.3f}  "
             f"naik={sum(1 for d in deltas if d>0)}  turun={sum(1 for d in deltas if d<0)}  "
             f"tetap={sum(1 for d in deltas if d==0)}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "greedy_queue"
    if which in ("greedy_queue", "greedy_util"):
        mode = "queue" if which == "greedy_queue" else "utilization"
        fac = lambda sim: GreedyAgent(mode=mode, top_k=2)
        rows = run_and_collect(fac, DS, SEED)
        summarize(rows, which)
    elif which == "master_ev_ppo":
        ckpt = sys.argv[2]
        # forecaster opsional sbg argumen ke-3: "formula" (baku, kasar -- FormulaForecaster)
        # | "vwf" (VirtualWaitForecaster, basis DISAMAKAN dgn eta_norm yg dilihat aktor).
        # HARUS SAMA dgn forecaster yg dipakai saat CHECKPOINT ini dilatih -- mismatch
        # latih/uji di sini murni soal EstWait yg ditampilkan, bukan bobot jaringan (jaringan
        # tetap dimuat sama), tapi hasil diagnosis jadi tak mencerminkan kondisi asli training.
        fc_name = sys.argv[3] if len(sys.argv) > 3 else "formula"
        from marl_spklu.rl.master_ev_ppo_policy import MasterEVPPOPolicy, MasterEVPPOInferenceAgent
        from marl_spklu.rl.forecaster import FormulaForecaster, VirtualWaitForecaster
        forecaster = VirtualWaitForecaster() if fc_name == "vwf" else FormulaForecaster()
        sim0 = common.fresh_sim(DS)
        pol = MasterEVPPOPolicy(len(sim0.spklus))
        pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
        pol.eval()
        fac = lambda sim: MasterEVPPOInferenceAgent(pol, sim, forecaster, k=3)
        rows = run_and_collect(fac, DS, SEED)
        summarize(rows, f"MASTER-EV-PPO+{fc_name} ({ckpt})")
    else:
        raise SystemExit(f"opsi tak dikenal: {which}")


if __name__ == "__main__":
    main()
