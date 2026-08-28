#!/bin/bash
# Uji StationConcatDecisionHead dgn ukuran sedikit lebih besar (attn_dim=16,
# mlp_hidden=48, bukan 8/32 baku) -- uji dugaan leher-botol attn_dim=8 memaksa
# jaringan memprioritaskan sinyal pemerataan (gini) di atas sinyal individual
# (wait/acceptance), menghasilkan ketimpangan ANTAR-METRIK. Tujuan di sini
# KESEIMBANGAN, bukan gini serendah mungkin. Sepadan h1a_pemerataan_dgr &
# concat-head 8/32, HANYA beda ukuran. 30d & 90d.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_concat_head_16_48.sh > Eksekusi_RL/outputs/retrain_concat_head_16_48.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--no-hist --n-critics 3 --forecaster vwf --reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --use-concat-head --concat-attn-dim 16 --concat-mlp-hidden 48"

echo "=== 1. concat-head-16-48 30d ==="
$PY $PIPE $COMMON

echo "=== 2. concat-head-16-48 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
