#!/bin/bash
# Master-PPO (2026-08-28) -- TIGA tahap (diubah dari SATU): pra-latih spesialis
# wait & gini (r_star = rerata return 20% chunk terakhir, acuan TETAP gap-ratio),
# baru training DGR gabungan. n_updates=300, 3 seed per tahap.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure_ppo.sh > Eksekusi_RL/outputs/retrain_master_pure_ppo.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_ppo_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d"

echo "=== 1. Spesialis stream 0 (wait) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON

echo "=== 2. Spesialis stream 1 (gini) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON

echo "=== 3. Training DGR (r_star tetap dari tahap 1&2) ==="
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
