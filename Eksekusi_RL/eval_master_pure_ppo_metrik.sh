#!/bin/bash
# Evaluasi metrik kaya utk retrain_master_pure_ppo.sh (30d).
# Jalankan SETELAH retrain_master_pure_ppo.sh selesai, dari root repo:
#   nohup bash Eksekusi_RL/eval_master_pure_ppo_metrik.sh > Eksekusi_RL/outputs/eval_master_pure_ppo_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_ppo_metrik.py

echo "=== Master-PPO 30d ==="
$PY $UJI 0,1,2 30d master_pure_ppo

echo "=== SELESAI ==="
