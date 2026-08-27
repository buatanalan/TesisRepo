#!/bin/bash
# Uji StationSelfAttention (2026-08-23) -- sepadan dgn h1a_pemerataan_dgr
# (Eksekusi_Hipotesis/), HANYA beda --use-station-attn, supaya selisihnya bisa
# diatribusikan langsung ke lapisan atensi baru. 30d & 90d.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_station_attn.sh > Eksekusi_RL/outputs/retrain_station_attn.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--no-hist --n-critics 3 --forecaster vwf --reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --use-station-attn"

echo "=== 1. station-attn 30d ==="
$PY $PIPE $COMMON

echo "=== 2. station-attn 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
