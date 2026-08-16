import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import json
import matplotlib.pyplot as plt

# Load dataset
with open('scenario_dataset.json', 'r') as f:
    data = json.load(f)

# Extract SPKLU locations
spklus = data.get('spklus', [])
spklu_x = [s['location'][0] for s in spklus]
spklu_y = [s['location'][1] for s in spklus]

# Extract a sample of EV spawn locations from the schedule
schedule = data.get('schedule', [])
# Taking first 500 EVs for visualization
sample_evs = schedule[:500]
ev_x = [ev['spawn_loc'][0] for ev in sample_evs if 'spawn_loc' in ev]
ev_y = [ev['spawn_loc'][1] for ev in sample_evs if 'spawn_loc' in ev]

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(spklu_x, spklu_y, c='blue', marker='^', s=100, label='SPKLU Location', alpha=0.8)
plt.scatter(ev_x, ev_y, c='red', marker='o', s=20, label='EV Spawn Location (Sample)', alpha=0.5)

plt.title('Visualization of SPKLU Locations and EV Spawn Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()


import json
import random
import numpy as np
import matplotlib.pyplot as plt
from marl_spklu.env.simulator import Simulator
from marl_spklu.env.history_buffer import HistoryBuffer

N_TIMESTEPS = 200          # jumlah timestep (15 menit/step) yang divisualisasikan
MAX_CHARGERS_TO_PLOT = 25  # batas total charger per cluster agar heatmap tetap terbaca

with open('scenario_dataset.json', 'r') as f:
    dataset = json.load(f)

seed = dataset['metadata']['seed']
max_steps = dataset['metadata']['max_steps']
spklu_cluster = {s['id']: s.get('cluster_id') for s in dataset['spklus']}

# Pass 1: jalankan simulasi penuh sekali untuk menemukan cluster yang paling
# banyak melayani EV, dibatasi ke cluster yang jumlah total charger-nya kecil
# supaya heatmap per-charger di bawah tetap terbaca.
sim_full = Simulator({}, [], HistoryBuffer(list(spklu_cluster.keys())))
sim_full.load_from_dataset('scenario_dataset.json')
np.random.seed(seed)
random.seed(seed)
sim_full.run(max_steps=max_steps)

served_per_cluster, capacity_per_cluster = {}, {}
for sid, spklu in sim_full.spklus.items():
    c = spklu_cluster.get(sid)
    served_per_cluster[c] = served_per_cluster.get(c, 0) + spklu.total_served
    capacity_per_cluster[c] = capacity_per_cluster.get(c, 0) + sum(spklu.capacities.values())

candidates = [c for c, cap in capacity_per_cluster.items() if cap <= MAX_CHARGERS_TO_PLOT]
target_cluster = max(candidates, key=lambda c: served_per_cluster.get(c, 0))
cluster_spklu_ids = sorted(sid for sid, c in spklu_cluster.items() if c == target_cluster)
print(f"Cluster terpilih: {target_cluster} | SPKLU: {cluster_spklu_ids} | "
      f"total dilayani: {served_per_cluster[target_cluster]} | total charger: {capacity_per_cluster[target_cluster]}")

# Daftar charger individual (spklu_id, tipe_konektor, indeks_slot) di cluster ini.
# Catatan: indeks_slot bersifat posisional (urutan pengisian FIFO), bukan ID
# fisik charger tetap -- model SPKLU tidak menyimpan identitas charger per slot.
chargers = [
    (sid, c_type, slot)
    for sid in cluster_spklu_ids
    for c_type, cap in sim_full.spklus[sid].capacities.items()
    for slot in range(cap)
]

# Pass 2: ulang simulasi dari awal (seed sama -> trajectory sama), rekam
# okupansi tiap charger di atas pada tiap step.
sim = Simulator({}, [], HistoryBuffer(list(spklu_cluster.keys())))
sim.load_from_dataset('scenario_dataset.json')
np.random.seed(seed)
random.seed(seed)

occupancy = np.zeros((len(chargers), N_TIMESTEPS))
for step in range(N_TIMESTEPS):
    sim.step_once(step)
    for i, (sid, c_type, slot) in enumerate(chargers):
        charging_list = sim.spklus[sid].charging.get(c_type, [])
        occupancy[i, step] = 1 if slot < len(charging_list) else 0

fig, ax = plt.subplots(figsize=(14, max(4, len(chargers) * 0.3)))
ax.imshow(occupancy, aspect='auto', cmap='Reds', interpolation='nearest')
ax.set_yticks(range(len(chargers)))
ax.set_yticklabels([f"{sid} ({c_type}#{slot})" for sid, c_type, slot in chargers], fontsize=7)
ax.set_xlabel('Timestep (15 menit/step)')
ax.set_title(f'Okupansi Charger di Cluster {target_cluster} sepanjang {N_TIMESTEPS} timestep (merah = terisi)')
plt.tight_layout()
plt.show()

import json
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from marl_spklu.env.simulator import Simulator
from marl_spklu.env.history_buffer import HistoryBuffer

N_EACH = 3  # jumlah sampel EV berulang & tidak berulang yang divisualisasikan

with open('scenario_dataset.json', 'r') as f:
    dataset = json.load(f)

seed = dataset['metadata']['seed']
max_steps = dataset['metadata']['max_steps']

# EV "berulang" = muncul >1 kali di schedule (hasil recurrence), "tidak
# berulang" = cuma sekali. Diurutkan dari yang paling sering muncul agar
# sampel yang dipilih cukup ilustratif (siklusnya kelihatan jelas di plot).
event_counts = Counter(ev['user_id'] for ev in dataset['schedule'])
recurring = sorted([uid for uid, n in event_counts.items() if n > 1], key=lambda u: -event_counts[u])
non_recurring = sorted([uid for uid, n in event_counts.items() if n == 1])

sample_recurring = recurring[:N_EACH]
sample_non_recurring = non_recurring[:N_EACH]
selected_users = sample_recurring + sample_non_recurring
print("Sampel berulang:", [(u, event_counts[u]) for u in sample_recurring])
print("Sampel tidak berulang:", [(u, event_counts[u]) for u in sample_non_recurring])

spklu_ids = [s['id'] for s in dataset['spklus']]
sim = Simulator({}, [], HistoryBuffer(spklu_ids))
sim.load_from_dataset('scenario_dataset.json')
np.random.seed(seed)
random.seed(seed)

user_lookup = {u.user_id: u for u in sim.users}
state_history = {uid: [] for uid in selected_users}
for step in range(max_steps):
    sim.step_once(step)
    for uid in selected_users:
        state_history[uid].append(user_lookup[uid].state)

state_order = ["idle", "spawned", "traveling", "queuing", "charging", "done"]
state_to_y = {s: i for i, s in enumerate(state_order)}

fig, axes = plt.subplots(len(selected_users), 1, figsize=(14, 1.6 * len(selected_users)), sharex=True)
for ax, uid in zip(axes, selected_users):
    y = [state_to_y[s] for s in state_history[uid]]
    is_recurring = uid in sample_recurring
    color = "tab:blue" if is_recurring else "tab:orange"
    label = "Berulang" if is_recurring else "Tidak Berulang"
    ax.step(range(max_steps), y, where="post", color=color)
    ax.set_yticks(range(len(state_order)))
    ax.set_yticklabels(state_order, fontsize=8)
    ax.set_ylabel(f"{uid}\n({label}, {event_counts[uid]}x)", fontsize=8, rotation=0, ha="right", va="center")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Timestep (15 menit/step)")
fig.suptitle("Status EV Berulang vs Tidak Berulang Sepanjang Simulasi")
plt.tight_layout()
plt.show()

import json
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

PROXIMITY_KM = 5.0  # samakan dengan --proximity-km yang dipakai saat generate dataset

with open('scenario_dataset.json', 'r') as f:
    dataset = json.load(f)

spklus = dataset['spklus']
locs = [tuple(s['location']) for s in spklus]
cluster_ids = [s.get('cluster_id') for s in spklus]
n = len(spklus)

# Seluruh jarak pasangan (km, lokasi sudah dalam km di dataset), dipisah
# intra-cluster vs inter-cluster.
intra_dists, inter_dists = [], []
for i, j in itertools.combinations(range(n), 2):
    d = math.dist(locs[i], locs[j])
    if cluster_ids[i] == cluster_ids[j]:
        intra_dists.append(d)
    else:
        inter_dists.append(d)
intra_dists = np.array(intra_dists)
inter_dists = np.array(inter_dists)

# Jarak ke tetangga terdekat per SPKLU (lintas cluster, posisi fisik
# sebenarnya -- bukan cuma dalam cluster yang sama).
nn_dists = []
for i in range(n):
    best = min(math.dist(locs[i], locs[j]) for j in range(n) if j != i)
    nn_dists.append(best)
nn_dists = np.array(nn_dists)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
max_d = max(intra_dists.max() if len(intra_dists) else 0, 1.0)
bins = np.linspace(0, max_d, 40)
ax.hist(intra_dists, bins=bins, alpha=0.7, label=f'Intra-cluster (n={len(intra_dists)})', color='tab:blue')
if len(inter_dists):
    ax.hist(inter_dists, bins=np.linspace(0, inter_dists.max(), 40), alpha=0.5,
            label=f'Inter-cluster (n={len(inter_dists)})', color='tab:gray')
ax.axvline(PROXIMITY_KM, color='red', linestyle='--', label=f'proximity_km = {PROXIMITY_KM}')
ax.set_xlabel('Jarak pasangan SPKLU (km)')
ax.set_ylabel('Jumlah pasangan')
ax.set_title('Distribusi Jarak Seluruh Pasangan SPKLU')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(nn_dists, bins=30, color='tab:green', alpha=0.8)
ax.axvline(PROXIMITY_KM, color='red', linestyle='--', label=f'proximity_km = {PROXIMITY_KM}')
ax.axvline(np.median(nn_dists), color='black', linestyle=':', label=f'median = {np.median(nn_dists):.2f} km')
ax.set_xlabel('Jarak ke tetangga terdekat (km)')
ax.set_ylabel('Jumlah SPKLU')
ax.set_title('Distribusi Jarak Nearest-Neighbor per SPKLU')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

pct_intra_within = 100 * (intra_dists <= PROXIMITY_KM).sum() / len(intra_dists) if len(intra_dists) else float('nan')
print(f"Total SPKLU: {n}, total cluster lokal: {len(set(cluster_ids))}")
print(f"Pasangan intra-cluster <= {PROXIMITY_KM} km: {(intra_dists <= PROXIMITY_KM).sum()}/{len(intra_dists)} ({pct_intra_within:.1f}%)")
print(f"Jarak intra-cluster: median={np.median(intra_dists):.2f} km, maks={intra_dists.max():.2f} km" if len(intra_dists) else "Tidak ada pasangan intra-cluster")
print(f"Jarak nearest-neighbor: median={np.median(nn_dists):.2f} km, min={nn_dists.min():.2f} km, maks={nn_dists.max():.2f} km")

import os
import sys
import json
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROXIMITY_KM = 5.0  # samakan dengan --proximity-km yang dipakai saat generate dataset
SEED = 42            # samakan dengan --seed yang dipakai saat generate dataset

SYN_DIR = os.path.join(os.getcwd(), 'Synthetic Model')
if SYN_DIR not in sys.path:
    sys.path.insert(0, SYN_DIR)
import main as syn_main
from preprocessing import label_captive_stations
from spatial_clustering import project_to_utm, fit_gmm_bic, predict_clusters

def nn_distances(locs):
    n = len(locs)
    return np.array([min(math.dist(locs[i], locs[j]) for j in range(n) if j != i) for i in range(n)])

def all_pairwise(locs):
    return np.array([math.dist(locs[i], locs[j]) for i, j in itertools.combinations(range(len(locs)), 2)])

# --- Muat & proses SPKLU publik RIIL (sebelum kurasi cluster) ---
np.random.seed(SEED)
df_trx = pd.read_csv(os.path.join(syn_main.DATA_DIR, 'df_clean_standard.csv'))
df_master = syn_main.load_master(syn_main.DATA_DIR)
df_trx = df_trx[df_trx['ID_SPKLU'].isin(df_master['ID_SPKLU'])].copy()
trx_agg = syn_main.aggregate_trx(df_trx)
df_master = label_captive_stations(df_master, trx_agg)
df_public_real = df_master[df_master['is_public']].copy()
df_public_real = project_to_utm(df_public_real)

gmm_reg, _ = fit_gmm_bic(df_public_real, k_range=range(2, 3), random_state=SEED)
df_public_real = predict_clusters(df_public_real, gmm_reg)
df_public_real, _ = syn_main.enforce_local_proximity(df_public_real, proximity_km=PROXIMITY_KM)

real_locs = list(zip((df_public_real['X_UTM'] / 1000.0).values, (df_public_real['Y_UTM'] / 1000.0).values))
real_clusters = df_public_real['cluster_id'].values

real_nn = nn_distances(real_locs)
real_all = all_pairwise(real_locs)
real_intra = np.array([
    math.dist(real_locs[i], real_locs[j])
    for cid in set(real_clusters)
    for i, j in itertools.combinations([k for k, c in enumerate(real_clusters) if c == cid], 2)
])

# --- SPKLU SINTETIK (dataset saat ini, hasil kurasi cluster) ---
with open('scenario_dataset.json', 'r') as f:
    dataset = json.load(f)
syn_locs = [tuple(s['location']) for s in dataset['spklus']]
syn_clusters_arr = [s.get('cluster_id') for s in dataset['spklus']]

syn_nn = nn_distances(syn_locs)
syn_all = all_pairwise(syn_locs)
syn_intra = np.array([
    math.dist(syn_locs[i], syn_locs[j])
    for cid in set(syn_clusters_arr)
    for i, j in itertools.combinations([k for k, c in enumerate(syn_clusters_arr) if c == cid], 2)
])

# --- Plot berdampingan ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
panels = [
    ("Nearest-Neighbor", real_nn, syn_nn),
    ("Intra-Local-Group", real_intra, syn_intra),
    ("All-Pairwise", real_all, syn_all),
]
for ax, (title, real_d, syn_d) in zip(axes, panels):
    max_d = max(real_d.max(), syn_d.max())
    bins = np.linspace(0, max_d, 40)
    ax.hist(real_d, bins=bins, alpha=0.5, density=True, label=f'Riil (n={len(real_d)})', color='tab:orange')
    ax.hist(syn_d, bins=bins, alpha=0.5, density=True, label=f'Sintetik (n={len(syn_d)})', color='tab:blue')
    ax.axvline(np.median(real_d), color='tab:orange', linestyle=':', label=f'median riil={np.median(real_d):.2f} km')
    ax.axvline(np.median(syn_d), color='tab:blue', linestyle=':', label=f'median sintetik={np.median(syn_d):.2f} km')
    ax.set_title(title)
    ax.set_xlabel('Jarak (km)')
    ax.set_ylabel('Densitas (dinormalisasi)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("=== Ringkasan median (km) ===")
print(f"{'Metrik':<20s} {'Riil':>10s} {'Sintetik':>10s}")
print(f"{'Nearest-neighbor':<20s} {np.median(real_nn):>10.2f} {np.median(syn_nn):>10.2f}")
print(f"{'Intra-local-group':<20s} {np.median(real_intra):>10.2f} {np.median(syn_intra):>10.2f}")
print(f"{'All-pairwise':<20s} {np.median(real_all):>10.2f} {np.median(syn_all):>10.2f}")
print()
print("Catatan: NN & intra-local-group SEHARUSNYA mirip (keduanya proxy kepadatan lokal,")
print("dan cluster lokal sintetik disampel dari sebaran cluster lokal riil yang sama).")
print("All-pairwise WAJAR berbeda -- sintetik hanya mencakup wilayah hasil kurasi cluster")
print("(lihat SARAN_PERBAIKAN.md), bukan seluruh Jabodetabek seperti data riil.")

# Visualisasi Generasi Perilaku Baru Pengguna (MXL & Range Anxiety)
import json
import matplotlib.pyplot as plt
import numpy as np

with open('scenario_dataset.json', 'r') as f:
    dataset = json.load(f)

# 1. Distribusi beta_state (Habit Strength)
users = dataset.get('users', [])
beta_states = [u.get('beta_state', 4.04) for u in users]

# 2. Distribusi SOC (Range Anxiety)
schedule = dataset.get('schedule', [])
socs = [ev.get('soc', 50.0) for ev in schedule]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot beta_state
axes[0].hist(beta_states, bins=40, color='tab:purple', alpha=0.7)
axes[0].axvline(np.mean(beta_states), color='black', linestyle=':', label=f'mean = {np.mean(beta_states):.2f}')
axes[0].set_title('Distribusi Habit Strength ($\\beta_{{state}}$) per Pengguna')
axes[0].set_xlabel('$\\beta_{{state}}$')
axes[0].set_ylabel('Jumlah Pengguna')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot SOC
axes[1].hist(socs, bins=40, color='tab:red', alpha=0.7)
axes[1].axvline(np.mean(socs), color='black', linestyle=':', label=f'mean = {np.mean(socs):.1f}%')
axes[1].axvline(30.0, color='red', linestyle='--', label='Batas Range Anxiety (30%)')
axes[1].set_title('Distribusi State of Charge (SOC) Saat Muncul')
axes[1].set_xlabel('SOC (%)')
axes[1].set_ylabel('Jumlah Kejadian (Event)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

