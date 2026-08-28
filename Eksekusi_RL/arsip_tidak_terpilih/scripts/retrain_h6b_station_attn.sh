#!/bin/bash
# Uji ISOLASI: StationSelfAttention (residual + gerbang nol-awal, PUNYA jaring
# pengaman) di atas h6b_utama -- dibandingkan dgn retrain_h6b_concat_head.sh
# (StationConcatDecisionHead, TANPA gerbang/jaring pengaman) yg terbukti kolaps
# 1/3 seed. Kalau versi attention-saja ini STABIL di ke-3 seed, penyebab kolaps
# ConcatHead adalah ketiadaan gerbang -- BUKAN attention-di-atas-pref itu sendiri.
# Konfigurasi dasar identik retrain_h6b_concat_head.sh, HANYA beda flag attention.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_station_attn.sh > Eksekusi_RL/outputs/retrain_h6b_station_attn.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --use-station-attn"

echo "=== 1. h6b-station-attn 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-station-attn 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
