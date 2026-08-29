#!/bin/bash
# Master-Hybrid-PPO + cwtfail (circuit-breaker wait, threshold=120/penalty=-2.0, sudah
# terbukti menstabilkan & memperbaiki DGR sebelumnya) + reward-preset seimbang4x
# (2026-08-29) -- satu2nya komponen Kandidat A yg BELUM PERNAH ditransplantasi ke
# lengan MASTER manapun. Hipotesis yg diuji: gap gini/entropy/herding tersisa
# (Hybrid-PPO+cwtfail: gini=0,109 entropy=0,149 herding=0,244 vs Kandidat A:
# gini=0,071 entropy=0,266 herding=0,106) disebabkan kalibrasi reward, bukan
# arsitektur (P+attention SUDAH ada di Hybrid). TIGA tahap, n_updates=300, 3 seed.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_hybrid_ppo_cwtfail_seimbang4x.sh > Eksekusi_RL/outputs/retrain_master_hybrid_ppo_cwtfail_seimbang4x.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d --wait-fail-threshold 120 --wait-fail-penalty -2.0 --reward-preset seimbang4x"

echo "=== 1. Spesialis stream 0 (wait) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON

echo "=== 2. Spesialis stream 1 (gini) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON

echo "=== 3. Training DGR (r_star tetap dari tahap 1&2) ==="
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
