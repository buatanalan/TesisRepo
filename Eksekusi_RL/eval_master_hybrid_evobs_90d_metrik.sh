#!/bin/bash
# Evaluasi metrik kaya utk retrain_master_hybrid_evobs_90d.sh.
#   nohup bash Eksekusi_RL/eval_master_hybrid_evobs_90d_metrik.sh > Eksekusi_RL/outputs/eval_master_hybrid_evobs_90d_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
SUF=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_evobs

echo "=== Master-Hybrid PPO (ev-obs) 90d ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF

echo "=== Master-Hybrid DDPG (ev-obs) 90d -- lewati bila blok 2 retrain dilewati ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ddpg_metrik.py
$PY $UJI 0,1,2 90d master_hybrid_ddpg_specialist0_wait_$SUF
$PY $UJI 0,1,2 90d master_hybrid_ddpg_specialist1_gini_$SUF
$PY $UJI 0,1,2 90d master_hybrid_ddpg_dgr_$SUF

echo "=== SELESAI ==="
