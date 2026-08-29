#!/bin/bash
# Evaluasi metrik kaya utk retrain_master_hybrid_evobs.sh (30d).
#   nohup bash Eksekusi_RL/eval_master_hybrid_evobs_metrik.sh > Eksekusi_RL/outputs/eval_master_hybrid_evobs_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
SUF=cwtfail120pen-2_preffeat_pairout_pg0.1_evobs

echo "=== Master-Hybrid PPO (ev-obs) ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
$PY $UJI 0,1,2 30d master_hybrid_ppo_specialist0_wait_$SUF
$PY $UJI 0,1,2 30d master_hybrid_ppo_specialist1_gini_$SUF
$PY $UJI 0,1,2 30d master_hybrid_ppo_dgr_$SUF

echo "=== Master-Hybrid DDPG (ev-obs) ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ddpg_metrik.py
$PY $UJI 0,1,2 30d master_hybrid_ddpg_specialist0_wait_$SUF
$PY $UJI 0,1,2 30d master_hybrid_ddpg_specialist1_gini_$SUF
$PY $UJI 0,1,2 30d master_hybrid_ddpg_dgr_$SUF

echo "=== SELESAI ==="
