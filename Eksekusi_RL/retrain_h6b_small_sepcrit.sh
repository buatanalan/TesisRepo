#!/bin/bash
# h6b_utama versi KECIL (MasterEVPPOPrefPolicySmall, ~30,7% param baku) + separate-
# critic-heads sekaligus -- uji apakah kedua modifikasi yg sama2 punya efek POSITIF
# sendiri2 (small: kolaps wait tak lagi ekstrem; sepcrit: gini terbaik & n_kolaps=0
# di 30d) SALING MENAMBAH bila digabung, atau justru berinteraksi negatif spt
# kombinasi stattn+sepcrit sebelumnya. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_small_sepcrit.sh > Eksekusi_RL/outputs/retrain_h6b_small_sepcrit.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --pref-small --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --separate-critic-heads --critic-head-small 32"

echo "=== 1. h6b-small-sepcrit 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-small-sepcrit 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
