#!/bin/bash
# Opsi 1: Gini TERMINAL (APROKSIMASI per-chunk, lihat catatan MasterEVPPOTrainer).
# K3 base (tanpa P/accept/trust) sbg basis paling bersih utk isolasi efek gini_mode.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_gini_opsi1_terminal.sh > Eksekusi_RL/outputs/retrain_gini_opsi1_terminal.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--n-critics 3 --reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --forecaster vwf --gini-mode terminal"

echo "=== 1. K3 base + gini terminal 30d ==="
$PY $PIPE $COMMON

echo "=== 2. K3 base + gini terminal 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
