#!/bin/bash
# LateCtx + StationSelfAttn (n_kolaps=0, gini 0,0707 -- kandidat terbaik sesi ini)
# DITAMBAH separate-critic-heads. Uji apakah menggabungkan dua modifikasi yg SAMA2
# terbukti positif sendiri2 (LateCtx+StationSelfAttn: stabil penuh + gini terbaik;
# SeparateCritic: gini terbaik varian lain + stabil) saling menambah, atau malah
# berinteraksi negatif spt kombinasi lama stattn(context-awal)+sepcrit sebelumnya
# (yg justru memperburuk n_wait_tinggi). HANYA 30 HARI (konsisten eksperimen LateCtx
# lain). Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_latectx_stattn_sepcrit.sh > Eksekusi_RL/outputs/retrain_h6b_latectx_stattn_sepcrit.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
COMMON="--pref --pref-feature-mode --pref-small --late-ctx --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --n-eval-seed 3 --use-station-attn --separate-critic-heads --critic-head-small 32"

echo "=== h6b-latectx+stattn+sepcrit 30d ==="
$PY $PIPE $COMMON

echo "=== SELESAI ==="
