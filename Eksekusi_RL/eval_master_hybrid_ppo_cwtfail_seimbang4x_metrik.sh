#!/bin/bash
# Evaluasi metrik kaya utk retrain_master_hybrid_ppo_cwtfail_seimbang4x.sh (30d).
# Jalankan SETELAH retrain selesai, dari root repo:
#   nohup bash Eksekusi_RL/eval_master_hybrid_ppo_cwtfail_seimbang4x_metrik.sh > Eksekusi_RL/outputs/eval_master_hybrid_ppo_cwtfail_seimbang4x_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py

echo "=== 1. spesialis0 (wait) 30d ==="
$PY $UJI 0,1,2 30d master_hybrid_ppo_specialist0_wait_cwtfail120pen-2_seimbang4x

echo "=== 2. spesialis1 (gini) 30d ==="
$PY $UJI 0,1,2 30d master_hybrid_ppo_specialist1_gini_cwtfail120pen-2_seimbang4x

echo "=== 3. dgr (gabungan) 30d ==="
$PY $UJI 0,1,2 30d master_hybrid_ppo_dgr_cwtfail120pen-2_seimbang4x

echo "=== SELESAI ==="
