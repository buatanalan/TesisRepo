#!/bin/bash
# Gabungan StationSelfAttention (aktor, bergerbang) + separate-critic-heads (kritik)
# di atas h6b_utama, TANPA ConcatHead. Kedua flag independen (attn di sisi aktor,
# separate-critic di sisi kritik) -- diuji bersamaan utk lihat apakah head-kritik-
# terpisah yg terbukti stabil (retrain_h6b_separate_critic.sh, n_kolaps=0 di 30d)
# bisa "menahan" instabilitas StationSelfAttn (n_kolaps=1 di kedua horizon saat
# sendirian), atau instabilitas attn di sisi aktor tetap dominan krn independen dari
# kritik. Konfigurasi dasar identik skrip h6b_* lain. 30d & 90d.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_stattn_sepcrit.sh > Eksekusi_RL/outputs/retrain_h6b_stattn_sepcrit.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --use-station-attn --separate-critic-heads --critic-head-small 32"

echo "=== 1. h6b-stattn-sepcrit 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-stattn-sepcrit 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
