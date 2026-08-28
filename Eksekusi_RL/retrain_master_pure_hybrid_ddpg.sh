#!/bin/bash
# Master-Hybrid DDPG (2026-08-29) -- MASTER murni + modul P (late-inject) + station
# attention di sisi aktor + keluaran VEKTOR->bid skalar kontinu (argmax menang).
# Ukuran modul tambahan KECIL (vec_dim=8, pref_d_lstm/attn=8, station_attn_dim=8).
# TIGA tahap. n_updates=300, 3 seed per tahap. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure_hybrid_ddpg.sh > Eksekusi_RL/outputs/retrain_master_pure_hybrid_ddpg.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ddpg_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d"

echo "=== 1. Spesialis stream 0 (wait) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON

echo "=== 2. Spesialis stream 1 (gini) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON

echo "=== 3. Training DGR ==="
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
