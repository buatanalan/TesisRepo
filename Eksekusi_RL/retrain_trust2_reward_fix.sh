#!/bin/bash
# Retrain P+nohist+trust2 K3 (30d & 90d) dgn pipeline reward SUDAH DIPERBAIKI
# (seimbang4x + gap_ratio + sigma1) -- sebelumnya trust2 hanya diuji dgn reward
# mentah blm terkalibrasi. Perbandingan apple-to-apple dgn K3 base & P+accept
# versi baru (retrain_reward_fix.sh). Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_trust2_reward_fix.sh > Eksekusi_RL/outputs/retrain_trust2_reward_fix.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --no-hist --n-critics 3 --reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --forecaster vwf --alpha-trust 2"

echo "=== 1. P+nohist+trust2 K3 30d ==="
$PY $PIPE $COMMON

echo "=== 2. P+nohist+trust2 K3 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
