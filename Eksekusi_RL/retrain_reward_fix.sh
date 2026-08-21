#!/bin/bash
# Retrain config kunci dgn KEDUA perbaikan reward aktif sekaligus (2026-08-21):
#   --reward-preset seimbang4x  (kalibrasi 4x asli, delta-gini, bukan default mentah)
#   --beta-mode gap_ratio --beta-sigma 1.0  (DGR genuine, transisi lunak)
# + acceptance_reward kini di STREAM_GLOBAL3 (kode, bukan flag).
# Mencakup K3 base (flagship gini) & P+nohist+accept K3 (flagship acceptance/trust),
# 30d & 90d. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_reward_fix.sh > Eksekusi_RL/outputs/retrain_reward_fix.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --forecaster vwf"

echo "=== 1. K3 base 30d ==="
$PY $PIPE --n-critics 3 $COMMON

echo "=== 2. K3 base 90d ==="
$PY $PIPE --n-critics 3 $COMMON --dataset $DS90 --horizon 90d

echo "=== 3. P+nohist+acc1 K3 30d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 $COMMON --alpha-accept 1

echo "=== 4. P+nohist+acc1 K3 90d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 $COMMON --alpha-accept 1 --dataset $DS90 --horizon 90d

echo "=== SEMUA RETRAIN REWARD-FIX SELESAI ==="
