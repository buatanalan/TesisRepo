"""Estimasi waktu eksekusi: dari rencana (a priori) atau dari log berjalan (a posteriori).

Dua cara, dan yang kedua jauh lebih dapat dipercaya:

  1. RENCANA -- pakai tabel kalibrasi di bawah. Berguna sebelum menjalankan apa pun,
     tetapi angkanya terikat pada mesin tempat kalibrasi diukur.

  2. LOG BERJALAN -- baca laju sebenarnya dari JSONL yang ditulis per-iterasi, lalu
     ekstrapolasi. Ini otomatis menyesuaikan mesin, beban, dan jumlah thread. Cukup
     tunggu ~2 menit setelah run dimulai.

Pemakaian:
    python _estimasi_waktu.py rencana 12 hppo 30d      # 12 run H-PPO horizon 30 hari
    python _estimasi_waktu.py rencana 12 pppo 90d
    python _estimasi_waktu.py log outputs/t2_hppo_30d_abs_seed0.jsonl
    python _estimasi_waktu.py log outputs/*.jsonl       # semua yang sedang berjalan
"""
import sys, os, json, glob, time

# --- Kalibrasi, diukur pada mesin lokal (Windows, CPU, torch 2.12.1+cpu, 2026-08-17) ---
# TRAINING: biaya ditentukan N_UPDATES x ROLLOUT_STEPS, BUKAN panjang dataset. Horizon
# 90d hanya ~16% lebih lambat (overhead objek & antrean tertunda pass pertama), bukan 3x
# meski datasetnya 3x lebih panjang. Ini sering disalahduga.
MENIT_PER_RUN = {          # 300 update x 288 langkah rollout
    ("hppo", "30d"): 20.8,   # n=18, rentang 19,7-22,4
    ("hppo", "90d"): 24.1,   # n=2
    ("pppo", "30d"): 24.2,   # n=12, rentang 22,0-27,5
    ("pppo", "90d"): 27.2,   # n=2
}

# EVALUASI: satu pass penuh -> biaya SEBANDING panjang dataset (beda dgn training).
MS_PER_LANGKAH = {"greedy": 2.85, "rl": 13.60}   # RL ~4,8x greedy (forward jaringan)
MAX_STEPS = {"30d": 2878, "90d": 8640}


def fmt(menit):
    if menit < 60:
        return f"{menit:.0f} menit"
    return f"{menit/60:.1f} jam ({menit:.0f} menit)"


def rencana(n_run, metode, horizon):
    key = (metode, horizon)
    if key not in MENIT_PER_RUN:
        raise SystemExit(f"kombinasi tak dikenal: {key} (pilih {list(MENIT_PER_RUN)})")
    per = MENIT_PER_RUN[key]
    total = n_run * per
    print(f"TRAINING  {n_run} run x {metode} {horizon}")
    print(f"  per run : {per:.1f} menit")
    print(f"  TOTAL   : {fmt(total)}")
    print()
    # Evaluasi: 3 baseline + 2 RL per aturan, x2 aturan, + 4 sel silang -- per seed.
    ms = MAX_STEPS[horizon]
    e_greedy = MS_PER_LANGKAH["greedy"] * ms / 1000 / 60
    e_rl = MS_PER_LANGKAH["rl"] * ms / 1000 / 60
    n_seed = max(1, n_run // 4)          # 4 lengan per seed pada faktorial 2x2
    per_seed = 2 * (3 * e_greedy + 2 * e_rl) + 4 * e_rl    # 2 aturan + sel silang
    print(f"EVALUASI  horizon {horizon}, {n_seed} seed")
    print(f"  1 lengan greedy : {e_greedy*60:.0f} detik")
    print(f"  1 lengan RL     : {e_rl*60:.0f} detik")
    print(f"  TOTAL           : {fmt(per_seed * n_seed)}")
    print()
    print(f"KESELURUHAN : {fmt(total + per_seed * n_seed)}")
    print()
    print("CATATAN: angka ini terikat mesin kalibrasi. Setelah run dimulai ~2 menit,")
    print("pakai mode `log` -- ia mengukur laju sebenarnya di mesin Anda.")


def dari_log(pola, n_updates=300):
    berkas = sorted(glob.glob(pola)) if any(c in pola for c in "*?[") else [pola]
    berkas = [f for f in berkas if os.path.exists(f)]
    if not berkas:
        raise SystemExit(f"tak ada berkas cocok: {pola}")
    sekarang = time.time()
    print("%-38s %6s %9s %11s %11s" % ("log", "iter", "detik/it", "sisa", "status"))
    print("-" * 80)
    for f in berkas:
        try:
            baris = [json.loads(x) for x in open(f) if x.strip()]
        except Exception:
            continue
        it = [b for b in baris if "n_tr" in b]
        if not it:
            continue
        n = len(it)
        # Laju diukur dari mtime berkas dibagi jumlah iterasi -- tak perlu stempel waktu
        # di dalam log, dan otomatis mencakup semua overhead nyata.
        umur = os.path.getmtime(f) - os.path.getctime(f)
        if umur <= 0 or n < 2:
            continue
        per_it = umur / n
        sisa_it = max(0, n_updates - n)
        sisa = sisa_it * per_it / 60
        diam = (sekarang - os.path.getmtime(f)) / 60
        status = "SELESAI" if sisa_it == 0 else ("MANDEK?" if diam > 5 else "berjalan")
        print("%-38s %6d %9.2f %11s %11s" % (
            os.path.basename(f), n, per_it, "-" if sisa_it == 0 else fmt(sisa), status))
    print()
    print("`MANDEK?` = berkas tak tersentuh >5 menit. Proses mungkin mati; cek dgn ps.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "rencana":
        rencana(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "log":
        dari_log(sys.argv[2])
    else:
        raise SystemExit(__doc__)
