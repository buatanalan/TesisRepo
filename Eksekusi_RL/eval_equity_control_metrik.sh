#!/bin/bash
# Evaluasi metrik kaya utk 2 checkpoint retrain_equity_control.sh (K4+equity).
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/eval_equity_control_metrik.sh > Eksekusi_RL/outputs/eval_equity_control_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. K4+equity 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_eq1_vwf_seimbang4x_K4_gap_sig1

echo "=== 2. K4+equity 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_eq1_vwf_seimbang4x_K4_gap_sig1_90d

echo "=== SELESAI ==="
