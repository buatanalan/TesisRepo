#!/bin/bash
# Evaluasi metrik kaya utk 2 checkpoint retrain_concat_head.sh.
# Jalankan SETELAH retrain_concat_head.sh selesai, dari root repo di server:
#   nohup bash Eksekusi_RL/eval_concat_head_metrik.sh > Eksekusi_RL/outputs/eval_concat_head_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. concat-head 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_nohist_concatA8H32_vwf_seimbang4x_K3_gap_sig1

echo "=== 2. concat-head 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_nohist_concatA8H32_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== SELESAI ==="
