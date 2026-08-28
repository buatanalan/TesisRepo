#!/bin/bash
# MasterEVPPOPrefPolicySmallLateCtx + StationSelfAttention (2026-08-28) -- head
# DASAR "context lambat" (lih. retrain_h6b_latectx.sh) DITAMBAH attention antar-
# stasiun (residual + gerbang nol-awal). Uji apakah StationSelfAttn kini stabil
# (n_kolaps=0) begitu ditempelkan pada emb yg SUDAH terlindung dari zona konflik
# preferensi (attention beroperasi pada emb SETELAH ctx_merge, tapi station_encoder
# yg membentuknya kini murni). 30d SUDAH selesai (gini 0,0707, n_kolaps=0) --
# TAHAP 90d ditambahkan 2026-08-28 utk melengkapi gap pengujian sblm kandidat ini
# difinalkan (Kandidat A blm pernah diuji horizon panjang). Jalankan dari root
# repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_latectx_stattn.sh > Eksekusi_RL/outputs/retrain_h6b_latectx_stattn.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--pref --pref-feature-mode --pref-small --late-ctx --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --n-eval-seed 3 --use-station-attn"

echo "=== h6b-latectx+stattn 30d ==="
$PY $PIPE $COMMON

echo "=== h6b-latectx+stattn 90d ==="
$PY $PIPE $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
