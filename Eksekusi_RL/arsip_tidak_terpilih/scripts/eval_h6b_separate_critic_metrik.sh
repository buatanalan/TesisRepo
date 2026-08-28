#!/bin/bash
# Evaluasi metrik kaya utk 2 checkpoint retrain_h6b_separate_critic.sh.
# Jalankan SETELAH retrain_h6b_separate_critic.sh selesai, dari root repo di server:
#   nohup bash Eksekusi_RL/eval_h6b_separate_critic_metrik.sh > Eksekusi_RL/outputs/eval_h6b_separate_critic_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. h6b-separate-critic 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_nohist_acc1_sepcritH32_vwf_seimbang4x_K3_gap_sig1

echo "=== 2. h6b-separate-critic 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_pref_feat_nohist_acc1_sepcritH32_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== SELESAI ==="
