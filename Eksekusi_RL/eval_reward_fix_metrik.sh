#!/bin/bash
# Evaluasi metrik kaya utk 4 checkpoint retrain_reward_fix.sh (seimbang4x + gap_ratio +
# sigma1 + acceptance_reward di GLOBAL3). Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/eval_reward_fix_metrik.sh > Eksekusi_RL/outputs/eval_reward_fix_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. K3 base 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_vwf_seimbang4x_K3_gap_sig1

echo "=== 2. K3 base 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== 3. P+nohist+acc1 K3 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_nohist_acc1_vwf_seimbang4x_K3_gap_sig1

echo "=== 4. P+nohist+acc1 K3 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_pref_feat_nohist_acc1_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== SELESAI ==="
