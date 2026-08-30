#!/bin/bash
# Evaluasi metrik kaya utk retrain_hybrid_ppo_3arah.sh (3 varian x 2 horizon = 6 tag).
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_3arah_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_3arah_metrik.log 2>&1 &
set -e
PY=.venv/Scripts/python.exe
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py

eval_arm () {
  local HORIZON=$1; local SUF=$2
  $PY $UJI 0,1,2 $HORIZON master_hybrid_ppo_specialist0_wait_$SUF
  $PY $UJI 0,1,2 $HORIZON master_hybrid_ppo_specialist1_gini_$SUF
  $PY $UJI 0,1,2 $HORIZON master_hybrid_ppo_dgr_$SUF
}

echo "########## 30 HARI ##########"
eval_arm 30d cwtfail120pen-2_preffeat_pairout_pg0.1
eval_arm 30d cwtfail120pen-2
eval_arm 30d cwtfail120pen-2_preffeat_pairout_pg0.1_noattn

echo "########## 90 HARI ##########"
eval_arm 90d 90d_cwtfail120pen-2_preffeat_pairout_pg0.1
eval_arm 90d 90d_cwtfail120pen-2
eval_arm 90d 90d_cwtfail120pen-2_preffeat_pairout_pg0.1_noattn

echo "=== SELESAI ==="
