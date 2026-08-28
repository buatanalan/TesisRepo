#!/bin/bash
# h6b_utama versi KECIL (MasterEVPPOPrefPolicySmall) + StationConcatDecisionHead
# (8/32) -- uji apakah memperkecil ukuran meredam kolaps parah ConcatHead (wait
# hingga 847 di 90d, terburuk dari seluruh eksperimen). concat_head TETAP attn_dim=8
# mlp_hidden=32 (sudah kecil dari awal) -- yg mengecil di sini hanya station_dim
# (input concat_head, = hidden = 32 bukan 64). Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_small_concat.sh > Eksekusi_RL/outputs/retrain_h6b_small_concat.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --pref-small --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --n-eval-seed 3 --use-concat-head --concat-attn-dim 8 --concat-mlp-hidden 32"

echo "=== 1. h6b-small-concat 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-small-concat 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
