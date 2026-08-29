#!/bin/bash
# Evaluasi metrik kaya utk ketiga tag hasil retrain_master_pure_ppo_cwtfail.sh (30d,
# wait_fail_threshold=120, wait_fail_penalty=-2.0). Jalankan SETELAH retrain selesai
# (ketiga tahap), dari root repo:
#   nohup bash Eksekusi_RL/eval_master_pure_ppo_cwtfail_metrik.sh > Eksekusi_RL/outputs/eval_master_pure_ppo_cwtfail_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_ppo_metrik.py

echo "=== 1. spesialis0 (wait, cwtfail) 30d ==="
$PY $UJI 0,1,2 30d master_pure_ppo_specialist0_wait_cwtfail120pen-2

echo "=== 2. spesialis1 (gini, cwtfail) 30d ==="
$PY $UJI 0,1,2 30d master_pure_ppo_specialist1_gini_cwtfail120pen-2

echo "=== 3. dgr (gabungan, cwtfail) 30d ==="
$PY $UJI 0,1,2 30d master_pure_ppo_dgr_cwtfail120pen-2

echo "=== SELESAI ==="
