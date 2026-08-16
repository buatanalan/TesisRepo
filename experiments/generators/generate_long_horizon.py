import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""
generate_long_horizon.py
Membuat dataset dengan median interaksi per user = 10.

Strategi yang benar: REPLIKASI PENUH schedule (semua events) N kali,
masing-masing di-shift waktu. Dengan cara ini SETIAP user mendapat
trip yang sama diulang N kali → median pasti N × median_asli.

Dari scenario_dataset_180d.json: median=6, max_steps=17280 (180 hari)
Target median=10: perlu ceil(10/6) = 2 replikasi penuh = total 360 hari
  180d base + 180d replikasi → 360d total, median = 6*2 = 12
  Atau: 1.67x replikasi → ambil 1 replikasi penuh (hasilkan ~12) dan 
  kita terima median ~12 ≥ 10 (lebih aman untuk signifikansi statistik)

OUTPUT: scenario_long_360d.json (2 replikasi = 360 hari, median ~12)
"""
import json, copy, random, os, sys
import numpy as np
from collections import defaultdict, Counter
import statistics as stat

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SRC = "scenario_dataset_180d.json"
DST = "scenario_long_360d.json"

print(f"Membaca {SRC}...")
with open(SRC) as f:
    base = json.load(f)

orig_max_steps = base["metadata"]["max_steps"]   # 17280 (180 hari)
orig_schedule  = base["schedule"]                # 21719 events
users          = base["users"]                   # 3366 users
spklus         = base["spklus"]                  # 50 SPKLU

print(f"Base  : {orig_max_steps} step = {orig_max_steps*15/60/24:.0f} hari")
print(f"        {len(users)} users, {len(orig_schedule)} events")

# ─── Berapa replikasi? ────────────────────────────────────────────────────────
# Dengan 2 replikasi: total 2 × 17280 = 34560 step (360 hari)
# Setiap user mendapat 2× trips → median = 2 × 6 = 12 ≥ 10 ✓
N_REPS = 2   # replikasi tambahan (total = 1 base + 2 rep = 3 blok)

TOTAL_STEPS = orig_max_steps * (N_REPS + 1)  # 3 × 17280 = 51840 — terlalu banyak
# Revisi: 1 base + 1 replikasi = 2 blok → 2 × 17280 = 34560 (360 hari)
N_REPS = 1
TOTAL_STEPS = orig_max_steps * (N_REPS + 1)
print(f"\nTarget: {N_REPS+1} blok × {orig_max_steps} step = {TOTAL_STEPS} step = {TOTAL_STEPS*15/60/24:.0f} hari")

# ─── Replikasi PENUH schedule ─────────────────────────────────────────────────
all_events = list(orig_schedule)  # blok 0 (asli, step 0..17279)

for rep in range(1, N_REPS + 1):
    offset = rep * orig_max_steps
    print(f"Replikasi {rep}: offset={offset} (step {offset}..{offset+orig_max_steps-1})")
    for ev in orig_schedule:
        new_ev = copy.deepcopy(ev)
        jitter = int(np.random.randint(-3, 4))   # ±3 step (±45 mnt)
        new_step = int(np.clip(ev["step"] + offset + jitter, offset, offset + orig_max_steps - 1))
        new_ev["step"] = new_step
        # Noise kecil SoC (±5%)
        new_ev["soc"] = round(float(np.clip(ev["soc"] + np.random.normal(0, 5.0), 5, 95)), 4)
        # Noise sangat kecil lokasi (±0.5 km)
        loc = ev["spawn_loc"]
        new_ev["spawn_loc"] = [
            round(loc[0] + float(np.random.normal(0, 0.5)), 2),
            round(loc[1] + float(np.random.normal(0, 0.5)), 2),
        ]
        all_events.append(new_ev)

all_events.sort(key=lambda e: e["step"])
total_events = len(all_events)
print(f"\nTotal events: {total_events} ({len(orig_schedule)} × {N_REPS+1} = {len(orig_schedule)*(N_REPS+1)})")

# ─── Statistik distribusi ─────────────────────────────────────────────────────
uid_trips = defaultdict(int)
for ev in all_events:
    uid_trips[ev["user_id"]] += 1

trips_vals = sorted(uid_trips.values())
print(f"\n{'='*55}")
print(f"STATISTIK DISTRIBUSI INTERAKSI PER USER")
print(f"{'='*55}")
print(f"  Horizon    : {TOTAL_STEPS} step = {TOTAL_STEPS*15/60/24:.0f} hari")
print(f"  Users      : {len(trips_vals)}")
print(f"  Events     : {total_events}")
print(f"  Min/Max    : {min(trips_vals)} / {max(trips_vals)}")
print(f"  Mean       : {stat.mean(trips_vals):.2f}")
print(f"  Median     : {stat.median(trips_vals):.1f}  <-- target: >= 10")
print(f"  P25/P75    : {np.percentile(trips_vals,25):.0f} / {np.percentile(trips_vals,75):.0f}")
print(f"  P90 / P99  : {np.percentile(trips_vals,90):.0f} / {np.percentile(trips_vals,99):.0f}")

dist = Counter(trips_vals)
print(f"\n  Distribusi trips per user (top-20):")
cumul = 0
for k in sorted(dist)[:22]:
    cnt = dist[k]
    cumul += cnt
    pct = cnt / len(trips_vals) * 100
    cpct = cumul / len(trips_vals) * 100
    bar = "#" * max(1, round(pct / 2))
    print(f"  {k:>3} trip: {cnt:>5} users ({pct:>4.1f}% | cum {cpct:>5.1f}%) {bar}")

# Konfirmasi target
med = stat.median(trips_vals)
if med >= 10:
    print(f"\n[OK] Target median >= 10 TERCAPAI: median = {med:.1f}")
else:
    print(f"\n[WARN] Target belum tercapai: median = {med:.1f} < 10")

# ─── Simpan ───────────────────────────────────────────────────────────────────
out = {
    "metadata": {
        "seed"        : SEED,
        "num_users"   : len(users),
        "num_events"  : total_events,
        "max_steps"   : TOTAL_STEPS,
        "horizon_days": round(TOTAL_STEPS * 15 / 60 / 24, 1),
        "n_reps"      : N_REPS,
        "description" : (
            f"Long horizon: {N_REPS+1} blok x {orig_max_steps} step (= {TOTAL_STEPS*15/60/24:.0f} hari). "
            f"Base: {SRC}. "
            f"Median trips/user = {med:.1f}. "
            "Setiap user mendapat replikasi penuh pola spawn."
        ),
        "base_dataset": SRC,
    },
    "spklus"  : spklus,
    "users"   : users,
    "schedule": all_events,
}

print(f"\nMenyimpan ke {DST}...")
with open(DST, "w") as f:
    json.dump(out, f, separators=(",", ":"))

size_mb = os.path.getsize(DST) / 1e6
print(f"[OK] {DST} ({size_mb:.1f} MB)")
print(f"     Median trips/user = {med:.1f}")
