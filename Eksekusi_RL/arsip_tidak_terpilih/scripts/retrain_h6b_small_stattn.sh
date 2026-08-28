#!/bin/bash
# h6b_utama versi KECIL (MasterEVPPOPrefPolicySmall) + StationSelfAttention -- uji
# apakah memperkecil ukuran (yg terbukti meredam LEDAKAN wait, meski tak
# menghilangkan n_kolaps gini) JUGA meredam instabilitas StationSelfAttn (yg
# sendirian, ukuran baku, kolaps parah wait 322-373 di kedua horizon). station_attn
# ikut mengecil otomatis (dim = hidden = 32, bukan 64) krn StationSelfAttention
# menerima dim dari MasterEVPPOPolicy.__init__. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_small_stattn.sh > Eksekusi_RL/outputs/retrain_h6b_small_stattn.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --pref-small --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --n-eval-seed 3 --use-station-attn"

echo "=== 1. h6b-small-stattn 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-small-stattn 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
