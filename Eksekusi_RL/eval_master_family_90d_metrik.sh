#!/bin/bash
# Evaluasi metrik kaya 90d utk retrain_master_family_90d.sh.
# Jalankan SETELAH retrain selesai (keduabelas tahap), dari root repo:
#   nohup bash Eksekusi_RL/eval_master_family_90d_metrik.sh > Eksekusi_RL/outputs/eval_master_family_90d_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
FAIL=cwtfail120pen-2
PREF=cwtfail120pen-2_preffeat_pairout_pg0.1

echo "=== [1/4] MASTER-DDPG murni 90d ==="
UJI=Eksekusi_RL/_uji_master_pure_metrik.py
$PY $UJI 0,1,2 90d master_pure_specialist0_wait_90d_$FAIL
$PY $UJI 0,1,2 90d master_pure_specialist1_gini_90d_$FAIL
$PY $UJI 0,1,2 90d master_pure_dgr_90d_$FAIL

echo "=== [2/4] MASTER-PPO murni 90d ==="
UJI=Eksekusi_RL/_uji_master_pure_ppo_metrik.py
$PY $UJI 0,1,2 90d master_pure_ppo_specialist0_wait_90d_$FAIL
$PY $UJI 0,1,2 90d master_pure_ppo_specialist1_gini_90d_$FAIL
$PY $UJI 0,1,2 90d master_pure_ppo_dgr_90d_$FAIL

echo "=== [3/4] Master-Hybrid DDPG 90d ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ddpg_metrik.py
$PY $UJI 0,1,2 90d master_hybrid_ddpg_specialist0_wait_90d_$PREF
$PY $UJI 0,1,2 90d master_hybrid_ddpg_specialist1_gini_90d_$PREF
$PY $UJI 0,1,2 90d master_hybrid_ddpg_dgr_90d_$PREF

echo "=== [4/4] Master-Hybrid PPO 90d ==="
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_90d_$PREF
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_90d_$PREF
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_90d_$PREF

echo "=== SELESAI ==="
