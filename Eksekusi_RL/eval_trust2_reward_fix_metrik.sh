#!/bin/bash
# Evaluasi metrik kaya utk 2 checkpoint retrain_trust2_reward_fix.sh.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/eval_trust2_reward_fix_metrik.sh > Eksekusi_RL/outputs/eval_trust2_reward_fix_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. P+nohist+trust2 K3 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_nohist_trust2_vwf_seimbang4x_K3_gap_sig1

echo "=== 2. P+nohist+trust2 K3 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_pref_feat_nohist_trust2_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== SELESAI ==="
