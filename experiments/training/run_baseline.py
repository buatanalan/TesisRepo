import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import numpy as np
import json
import os
import random
from marl_spklu.env.simulator import Simulator
from marl_spklu.env.history_buffer import HistoryBuffer

def run_discrete_baseline():
    dataset_file = "scenario_dataset.json"
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset {dataset_file} tidak ditemukan. Jalankan generate_dataset.py terlebih dahulu.")
        return

    print(f"=== Memulai Simulasi Skenario S0 (Load dari {dataset_file}) ===")

    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    seed = dataset["metadata"]["seed"]
    # max_steps diambil dari metadata dataset (bukan di-hardcode) supaya
    # horizon simulasi selalu konsisten dengan panjang dataset yang
    # digenerate (mis. --days 30), tidak diam-diam terpotong ke 7 hari.
    max_steps = dataset["metadata"]["max_steps"]
    spklu_ids = [s["id"] for s in dataset["spklus"]]

    np.random.seed(seed)
    random.seed(seed)

    # 1. Setup History
    history = HistoryBuffer(spklu_ids, window_size_15m=max_steps)

    # 2. Init Simulator and Load Dataset
    sim = Simulator({}, [], history, log_actor_states=True)
    sim.load_from_dataset(dataset_file)

    # 3. Run Simulation
    print(f"Menjalankan simulasi untuk {len(sim.users)} user...")
    sim.run(max_steps=max_steps)
    sim.export_actor_logs("s0_actor_trace")
    print("Simulasi Selesai!\n")
    
    # 4. Analisis Hasil
    print("=== Hasil Simulasi S0 ===")
    util_array = []
    for sid, spklu in sim.spklus.items():
        served = spklu.total_served
        wait = sum([log["wait_time"] for log in sim.logs if log["spklu"] == sid])
        avg_wait = wait / served if served > 0 else 0
        print(f"[{sid}] Total Dilayani: {served} mobil | Rata-rata Waktu Tunggu: {avg_wait:.2f} menit")
        util_array.append(served)
        
    gini = HistoryBuffer.calculate_gini(np.array(util_array))
    print(f"\nIndeks Gini Ketimpangan Utilisasi: {gini:.4f}")
    
    # 5. Simpan Log
    log_file = "simulation_detailed_logs.json"
    with open(log_file, "w") as f:
        json.dump(sim.detailed_logs, f, indent=4)
    print(f"\n[INFO] Log terperinci dari {len(sim.detailed_logs)} event telah disimpan ke {log_file}")
    
    metrics_file = "simulation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(sim.logs, f, indent=4)
    print(f"[INFO] Log metrik antrian telah disimpan ke {metrics_file}")

if __name__ == "__main__":
    run_discrete_baseline()
