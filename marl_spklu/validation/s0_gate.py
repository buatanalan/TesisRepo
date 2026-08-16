import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_metrics import cluster_gini, proximity_gini

def calculate_gini(array):
    array = np.array(array, dtype=np.float64).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def run_gate(metrics_file="simulation_metrics.json", dataset_file="scenario_dataset.json", proximity_threshold_km=5.0):
    print("=== Menjalankan Validasi Formal S0 Gate ===")
    if not os.path.exists(metrics_file):
        print(f"[FAIL] File {metrics_file} tidak ditemukan. Jalankan run_baseline.py (S0) terlebih dahulu.")
        sys.exit(1)
        
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    # Aggregate by SPKLU
    spklu_counts = {}
    for m in metrics:
        sid = m["spklu"]
        spklu_counts[sid] = spklu_counts.get(sid, 0) + 1
        
    if not spklu_counts:
        print("[FAIL] Tidak ada data pengguna yang dilayani di log S0.")
        sys.exit(1)
        
    util_array = np.array(list(spklu_counts.values()))
    
    # 1. Cek Gini Index > 0.3
    gini = calculate_gini(util_array)
    print(f"Indeks Gini: {gini:.4f}")
    
    if gini < 0.3:
        raise AssertionError(f"Gagal Gate S0: Indeks Gini {gini:.4f} < 0.3. Ketimpangan struktural tidak terjadi!")
        
    # 2. Cek Rasio Pemanfaatan >= 3x
    max_util = np.max(util_array)
    min_util = np.min(util_array)
    ratio = max_util / min_util if min_util > 0 else float('inf')
    
    print(f"Rasio SPKLU Paling Sibuk vs Sepi: {ratio:.2f}x")
    
    if ratio < 3.0:
        raise AssertionError(f"Gagal Gate S0: Rasio Utilisasi {ratio:.2f}x < 3.0x. SPKLU tidak cukup timpang!")
        
    print("[PASS] Validasi S0 Lulus. Lingkungan simulasi terbukti menghasilkan ketimpangan struktural yang relevan untuk intervensi RL.")

    # 3. Metrik tambahan: pisahkan ketimpangan ANTAR wilayah (cluster GMM,
    # bisa berjarak ratusan km, wajar karena beda demand) dari ketimpangan
    # LOKAL antar SPKLU yang benar-benar berdekatan (<= proximity_threshold_km),
    # yang relevan untuk klaim riset "2 SPKLU berdekatan tapi timpang".
    print("\n=== Metrik Ketimpangan Spasial (informasional) ===")
    if not os.path.exists(dataset_file):
        print(f"[WARN] {dataset_file} tidak ditemukan, skip metrik per-cluster & proximity.")
        return True

    with open(dataset_file, "r") as f:
        spklus = json.load(f)["spklus"]

    per_cluster, mean_cluster_gini = cluster_gini(spklus, spklu_counts)
    print(f"Gini per-cluster GMM: { {k: round(v, 4) for k, v in per_cluster.items()} }")
    print(f"Rata-rata Gini DALAM cluster (ketimpangan lokal per wilayah): {mean_cluster_gini:.4f}")

    proximity_results, mean_proximity_gini = proximity_gini(spklus, spklu_counts, threshold_km=proximity_threshold_km)
    print(f"\nJumlah grup SPKLU berdekatan (<= {proximity_threshold_km} km, >=2 SPKLU): {len(proximity_results)}")
    for r in proximity_results:
        print(f"  {r['spklu_ids']} -> counts={r['counts']} Gini={r['gini']:.4f}")
    print(f"Rata-rata Gini ANTAR SPKLU berdekatan (<= {proximity_threshold_km} km): {mean_proximity_gini:.4f}")

    return True

if __name__ == "__main__":
    run_gate()
