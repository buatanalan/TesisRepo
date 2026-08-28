#!/bin/bash
# Evaluasi metrik kaya ketiga tag retrain_master_pure_hybrid_ddpg.sh (30d).
#   nohup bash Eksekusi_RL/eval_master_pure_hybrid_ddpg_metrik.sh > Eksekusi_RL/outputs/eval_master_pure_hybrid_ddpg_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ddpg_metrik.py

echo "=== 1. spesialis0 (wait) 30d ==="
$PY $UJI 0,1,2 30d master_hybrid_ddpg_specialist0_wait

echo "=== 2. spesialis1 (gini) 30d ==="
$PY $UJI 0,1,2 30d master_hybrid_ddpg_specialist1_gini

echo "=== 3. dgr (gabungan) 30d ==="
$PY $UJI 0,1,2 30d master_hybrid_ddpg_dgr

echo "=== SELESAI ==="
