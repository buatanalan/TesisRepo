#!/bin/bash
# Uji StationConcatDecisionHead (pooling+MLP nonlinear per-stasiun, 2026-08-24) --
# sepadan h1a_pemerataan_dgr, HANYA beda --use-concat-head. 30d & 90d.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_concat_head.sh > Eksekusi_RL/outputs/retrain_concat_head.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--no-hist --n-critics 3 --forecaster vwf --reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --use-concat-head"

echo "=== 1. concat-head 30d ==="
$PY $PIPE $COMMON

echo "=== 2. concat-head 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
