#!/bin/bash
# Evaluasi metrik kaya utk 2 checkpoint retrain_station_attn_small.sh (d_attn=16).
# Jalankan SETELAH retrain_station_attn_small.sh selesai, dari root repo di server:
#   nohup bash Eksekusi_RL/eval_station_attn_small_metrik.sh > Eksekusi_RL/outputs/eval_station_attn_small_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. station-attn-d16 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_nohist_stattnd16_vwf_seimbang4x_K3_gap_sig1

echo "=== 2. station-attn-d16 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_nohist_stattnd16_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== SELESAI ==="
