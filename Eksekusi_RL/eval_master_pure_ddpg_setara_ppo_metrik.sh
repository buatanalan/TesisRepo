#!/bin/bash
# Evaluasi MASTER-DDPG murni dgn anggaran gradien setara PPO (retrain_master_pure_
# ddpg_setara_ppo.sh). PPO TIDAK dievaluasi ulang -- hasilnya sudah ada & tak berubah.
#
#   nohup bash Eksekusi_RL/eval_master_pure_ddpg_setara_ppo_metrik.sh > Eksekusi_RL/outputs/eval_master_pure_ddpg_setara_ppo_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_metrik.py
SUF=90d_cwtfail120pen-2_upc62

echo "=== MASTER-DDPG (anggaran gradien setara PPO) ==="
$PY $UJI 0,1,2 90d master_pure_specialist0_wait_$SUF
$PY $UJI 0,1,2 90d master_pure_specialist1_gini_$SUF
$PY $UJI 0,1,2 90d master_pure_dgr_$SUF

echo "=== SELESAI ==="
echo "Bandingkan thd master_pure_dgr_90d_cwtfail120pen-2 (anggaran lama) dan"
echo "master_pure_ppo_dgr_90d_cwtfail120pen-2 (PPO, sudah ada, TAK diulang)."
