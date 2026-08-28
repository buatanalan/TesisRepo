#!/bin/bash
# Evaluasi metrik kaya utk ketiga tag hasil retrain_master_pure.sh (30d).
# Jalankan SETELAH retrain_master_pure.sh selesai (ketiga tahap), dari root repo:
#   nohup bash Eksekusi_RL/eval_master_pure_metrik.sh > Eksekusi_RL/outputs/eval_master_pure_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_metrik.py

echo "=== 1. spesialis0 (wait) 30d ==="
$PY $UJI 0,1,2 30d master_pure_specialist0_wait

echo "=== 2. spesialis1 (gini) 30d ==="
$PY $UJI 0,1,2 30d master_pure_specialist1_gini

echo "=== 3. dgr (gabungan) 30d ==="
$PY $UJI 0,1,2 30d master_pure_dgr

echo "=== SELESAI ==="
