#!/bin/bash
# h6b_utama dgn MasterEVPPOPrefPolicySmall (2026-08-28) -- SAMA arsitektur (encoder
# aktor/kritik TETAP terpisah, CTDE penuh), hyperparameter diperkecil ~separuh
# (hidden 64->32, critic_hidden 128->64, pref_d_lstm/pref_d_attn 16->8) -> total
# param 30,7% baku (15.238 vs 49.670). Uji apakah kapasitas berlebih (relatif thd
# anggaran latih 300-chunk yg SAMA dgn arsitektur jauh lebih kecil) berkontribusi
# thd n_kolaps seed berulang di h6b_utama (baseline pun n_kolaps=1 di 90d).
# TANPA station-attn/concat-head/separate-critic (isolasi murni efek ukuran).
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_small.sh > Eksekusi_RL/outputs/retrain_h6b_small.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --pref-small --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf"

echo "=== 1. h6b-small 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-small 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
