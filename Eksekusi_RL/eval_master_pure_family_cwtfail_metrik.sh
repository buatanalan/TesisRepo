#!/bin/bash
# Evaluasi metrik kaya utk retrain_master_pure_family_cwtfail.sh (30d, threshold=120,
# penalty=-2.0) -- MASTER-DDPG murni, Master-Hybrid-DDPG, Master-Hybrid-PPO.
# Jalankan SETELAH retrain selesai, dari root repo:
#   nohup bash Eksekusi_RL/eval_master_pure_family_cwtfail_metrik.sh > Eksekusi_RL/outputs/eval_master_pure_family_cwtfail_metrik.log 2>&1 &
set -e
PY=.venv/bin/python

echo "=== MASTER-DDPG murni (cwtfail) ==="
UJI=Eksekusi_RL/_uji_master_pure_metrik.py
$PY $UJI 0,1,2 30d master_pure_specialist0_wait_cwtfail120pen-2
$PY $UJI 0,1,2 30d master_pure_specialist1_gini_cwtfail120pen-2
$PY $UJI 0,1,2 30d master_pure_dgr_cwtfail120pen-2

echo "=== Master-Hybrid-DDPG (cwtfail) ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ddpg_metrik.py
$PY $UJI 0,1,2 30d master_hybrid_ddpg_specialist0_wait_cwtfail120pen-2
$PY $UJI 0,1,2 30d master_hybrid_ddpg_specialist1_gini_cwtfail120pen-2
$PY $UJI 0,1,2 30d master_hybrid_ddpg_dgr_cwtfail120pen-2

echo "=== Master-Hybrid-PPO (cwtfail) ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
$PY $UJI 0,1,2 30d master_hybrid_ppo_specialist0_wait_cwtfail120pen-2
$PY $UJI 0,1,2 30d master_hybrid_ppo_specialist1_gini_cwtfail120pen-2
$PY $UJI 0,1,2 30d master_hybrid_ppo_dgr_cwtfail120pen-2

echo "=== SELESAI ==="
