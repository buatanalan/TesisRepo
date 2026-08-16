import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import json
import os
from marl_spklu.env.simulator import Simulator
from marl_spklu.agents.greedy_agent import GreedyAgent

def run_s1():
    print("=== Memulai Simulasi Skenario S1 (Greedy Navigation) ===")

    from marl_spklu.env.history_buffer import HistoryBuffer

    dataset_file = "scenario_dataset.json"
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    # max_steps & spklu_ids diambil dari metadata dataset (bukan di-hardcode)
    # supaya horizon & history buffer selalu konsisten dengan dataset yang
    # sedang dipakai, tidak diam-diam terpotong/salah ID seperti sebelumnya.
    max_steps = dataset["metadata"]["max_steps"]
    spklu_ids = [s["id"] for s in dataset["spklus"]]

    sim = Simulator({}, [], HistoryBuffer(spklu_ids, window_size_15m=max_steps), log_actor_states=True)
    sim.load_from_dataset(dataset_file)

    print(f"Menjalankan simulasi S1 untuk {len(sim.users)} user...")

    # Inisialisasi Agen S1
    agent = GreedyAgent()

    sim.run(max_steps=max_steps, agent=agent)
    sim.export_actor_logs("s1_actor_trace")
    
    print("Simulasi Selesai!\n")
    print("=== Hasil Simulasi S1 ===")
    
    spklu_metrics = {}
    for sid in sim.spklus.keys():
        spklu_metrics[sid] = {"count": 0, "wait_times": []}
        
    for log in sim.logs:
        sid = log["spklu"]
        spklu_metrics[sid]["count"] += 1
        spklu_metrics[sid]["wait_times"].append(log["wait_time"])
        
    for sid, m in spklu_metrics.items():
        if m["count"] > 0:
            avg_wait = sum(m["wait_times"]) / m["count"]
            print(f"[{sid}] Total Dilayani: {m['count']} mobil | Rata-rata Waktu Tunggu: {avg_wait:.2f} menit")
        else:
            print(f"[{sid}] Total Dilayani: 0 mobil")
            
    print(f"\nHerding Index S1 (Kejadian Pindah Berjamaah): {sim.herding_events} kejadian")
    if sim.herding_events > 0:
        print("[PERINGATAN] S1 terbukti menghasilkan Herding Effect! Kebijakan naif tanpa CTDE terbukti bermasalah.")
    
    with open("simulation_s1_metrics.json", "w") as f:
        json.dump(sim.logs, f, indent=4)
        
    print(f"\n[INFO] Log metrik antrian S1 telah disimpan ke simulation_s1_metrics.json")
    
if __name__ == "__main__":
    run_s1()
