#!/bin/bash
# Evaluasi metrik kaya utk ketiga tag hasil retrain_master_pure_ppo_clip.sh (30d,
# wait_reward_clip=2.0). Jalankan SETELAH retrain_master_pure_ppo_clip.sh selesai
# (ketiga tahap), dari root repo:
#   nohup bash Eksekusi_RL/eval_master_pure_ppo_clip_metrik.sh > Eksekusi_RL/outputs/eval_master_pure_ppo_clip_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_ppo_metrik.py

echo "=== 1. spesialis0 (wait, klip) 30d ==="
$PY $UJI 0,1,2 30d master_pure_ppo_specialist0_wait_clip2

echo "=== 2. spesialis1 (gini, klip) 30d ==="
$PY $UJI 0,1,2 30d master_pure_ppo_specialist1_gini_clip2

echo "=== 3. dgr (gabungan, klip) 30d ==="
$PY $UJI 0,1,2 30d master_pure_ppo_dgr_clip2

echo "=== SELESAI ==="
