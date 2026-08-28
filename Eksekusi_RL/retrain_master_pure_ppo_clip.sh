#!/bin/bash
# Master-PPO dgn KLIP wait_reward (2026-08-29) -- respons diagnosis kolaps entropi
# PPO-di-atas-Master-murni (rec_entropy -> ~0 pd seed tertentu, dipicu ekor tebal
# wait_reward yg mendominasi advantage GAE). Sama TIGA tahap spt
# retrain_master_pure_ppo.sh, tapi reward_calc pakai wait_reward_clip=2.0 (maks
# improvement dikreditkan = 2 x wait_scale = 120 menit) di KETIGA tahap (spesialis
# wait, spesialis gini, dan DGR gabungan -- konsisten satu reward_calc).
# n_updates=300, 3 seed per tahap. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure_ppo_clip.sh > Eksekusi_RL/outputs/retrain_master_pure_ppo_clip.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_ppo_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d --wait-reward-clip 2.0"

echo "=== 1. Spesialis stream 0 (wait, klip) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON

echo "=== 2. Spesialis stream 1 (gini, klip) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON

echo "=== 3. Training DGR (r_star tetap dari tahap 1&2, klip) ==="
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
