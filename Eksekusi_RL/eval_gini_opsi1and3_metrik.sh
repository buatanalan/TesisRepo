#!/bin/bash
# Evaluasi metrik kaya utk checkpoint Opsi 1 (terminal) & Opsi 3 (CMDP).
# Jalankan SETELAH kedua retrain_gini_opsi*.sh selesai, dari root repo di server:
#   nohup bash Eksekusi_RL/eval_gini_opsi1and3_metrik.sh > Eksekusi_RL/outputs/eval_gini_opsi1and3_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. Opsi 1 (terminal) 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_terminal_vwf_seimbang4x_K3_gap_sig1

echo "=== 2. Opsi 1 (terminal) 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_terminal_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== 3. Opsi 3 (CMDP) 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_cmdpE0.07lr0.5_vwf_seimbang4x_K3_gap_sig1

echo "=== 4. Opsi 3 (CMDP) 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_cmdpE0.07lr0.5_vwf_seimbang4x_K3_gap_sig1_90d

echo "=== SELESAI ==="
