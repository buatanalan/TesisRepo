#!/bin/bash
# StationConcatDecisionHead (attn_dim=8, mlp_hidden=32) DIBANGUN DI ATAS h6b_utama
# (MasterEVPPOPrefPolicy, fusi MASTER+PDQN) -- BUKAN lagi di atas arsitektur dasar
# (setara h1a) spt eksperimen sebelumnya. Konfigurasi dasar disamakan persis dgn
# `h6b_utama_dgr` resmi (Eksekusi_Hipotesis/1_eksperimen.py): --pref --pref-feature-mode
# --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0,
# ditambah --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval (baku Eksekusi_RL). 30d & 90d.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_concat_head.sh > Eksekusi_RL/outputs/retrain_h6b_concat_head.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --use-concat-head --concat-attn-dim 8 --concat-mlp-hidden 32"

echo "=== 1. h6b-concat-head 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-concat-head 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
