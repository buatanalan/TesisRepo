#!/bin/bash
# Evaluasi metrik kaya utk checkpoint retrain_h6b_latectx_stattn_sepcrit.sh (30d saja).
# Jalankan SETELAH retrain_h6b_latectx_stattn_sepcrit.sh selesai, dari root repo di server:
#   nohup bash Eksekusi_RL/eval_h6b_latectx_stattn_sepcrit_metrik.sh > Eksekusi_RL/outputs/eval_h6b_latectx_stattn_sepcrit_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== h6b-latectx+stattn+sepcrit 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_small_latectx_nohist_acc1_stattn_sepcritH32_vwf_seimbang4x_K3_gap_sig1

echo "=== SELESAI ==="
