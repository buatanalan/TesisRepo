#!/bin/bash
# Uji head kritik KECIL TERPISAH per-aliran DGR (bukan 1 MLP bersama K-cabang) di
# atas h6b_utama -- generalisasi pola STREAM_EQUITY (n_critics=4, sudah lebih dulu
# dipisah krn 1 head bersama antar-suku berlainan tujuan terbukti saling melemahkan)
# ke SELURUH aliran (WAIT/PROX/GLOBAL3). `critic_station_encoder`+`critic_pool` TETAP
# dibagi -- hanya kepala akhir yg dipecah. TANPA station-attn/concat-head (isolasi murni
# efek head-kritik-terpisah). Konfigurasi dasar identik retrain_h6b_concat_head.sh.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_separate_critic.sh > Eksekusi_RL/outputs/retrain_h6b_separate_critic.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --separate-critic-heads --critic-head-small 32"

echo "=== 1. h6b-separate-critic 30d ==="
$PY $PIPE $COMMON

echo "=== 2. h6b-separate-critic 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
