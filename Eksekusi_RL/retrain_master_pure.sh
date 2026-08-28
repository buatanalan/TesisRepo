#!/bin/bash
# MASTER benar-benar murni (2026-08-28) -- 3 tahap WAJIB berurutan:
#   1. Pra-latih spesialis stream 0 (wait/CWT-analog)
#   2. Pra-latih spesialis stream 1 (gini/pemerataan, pengganti CP)
#   3. Training DGR (memuat KEDUA spesialis di atas sbg Q*/b* beku, Pers. 13)
# Semua pakai n_updates=300 (anggaran sama, sesuai kesepakatan 2026-08-28).
# 3 seed per tahap -> total 9 kali n_updates=300 (3x lebih lama dari eksperimen
# h6b_utama biasa -- WAJAR, MASTER murni butuh 2 spesialis + 1 training gabungan).
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure.sh > Eksekusi_RL/outputs/retrain_master_pure.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d"

echo "=== 1. Spesialis stream 0 (wait) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON

echo "=== 2. Spesialis stream 1 (gini) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON

echo "=== 3. Training DGR (gabungan, Q*/b* beku dari tahap 1&2) ==="
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
