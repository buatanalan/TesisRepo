#!/bin/bash
# Uji ULANG SELURUH keluarga MASTER murni/Hybrid (DDPG+PPO) dgn ambang-gagal+penalti
# tetap pd wait_reward (2026-08-29, replika CWT paper, threshold=120 menit/2 jam,
# penalty=-2.0) -- respons diagnosis: perbandingan Hybrid vs murni SEBELUMNYA blm
# memakai mekanisme ini shg tak adil (kolaps Hybrid-DDPG/DDPG-murni menarik rata2
# turun tanpa circuit-breaker). MASTER-PPO murni SUDAH diuji terpisah
# (retrain_master_pure_ppo_cwtfail.sh) -- script ini melengkapi TIGA lengan yg
# BELUM: DDPG murni, Hybrid-DDPG, Hybrid-PPO. Tiap lengan TIGA tahap (spesialis0/
# spesialis1/dgr), n_updates=300, 3 seed. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure_family_cwtfail.sh > Eksekusi_RL/outputs/retrain_master_pure_family_cwtfail.log 2>&1 &
set -e
PY=.venv/bin/python
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
COMMON30="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d $FAIL"

echo "=== [1/3] MASTER-DDPG murni ==="
PIPE=Eksekusi_RL/_run_master_pure_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON30
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON30
$PY $PIPE --mode dgr $COMMON30

echo "=== [2/3] MASTER-Hybrid-DDPG ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ddpg_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON30
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON30
$PY $PIPE --mode dgr $COMMON30

echo "=== [3/3] MASTER-Hybrid-PPO ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON30
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON30
$PY $PIPE --mode dgr $COMMON30

echo "=== SELESAI ==="
