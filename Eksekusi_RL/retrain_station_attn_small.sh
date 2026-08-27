#!/bin/bash
# Uji StationSelfAttention dgn kapasitas dikecilkan (d_attn=16, bukan 64 baku) --
# diagnosis ukuran-terlalu-besar dari retrain_station_attn.sh sebelumnya. Sepadan
# h1a_pemerataan_dgr, hanya beda --use-station-attn --station-attn-dim 16.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_station_attn_small.sh > Eksekusi_RL/outputs/retrain_station_attn_small.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--no-hist --n-critics 3 --forecaster vwf --reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --use-station-attn --station-attn-dim 16"

echo "=== 1. station-attn-d16 30d ==="
$PY $PIPE $COMMON

echo "=== 2. station-attn-d16 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
