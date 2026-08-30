#!/bin/bash
# Latih ulang dari NOL 3 varian Hybrid-PPO (2026-08-30) utk mengisolasi kontribusi
# Modul P vs station attention, pada 30d DAN 90d (6 konfigurasi total):
#   1. GABUNGAN     -- station attention + P (mode fitur+outcome, gerbang 0.1)
#   2. ATTN SAJA    -- station attention aktif, P TIDAK diaktifkan (baseline hybrid lama)
#   3. P SAJA       -- station attention DIMATIKAN (--no-station-attn), P aktif
#
# WAJIB semua pakai cwtfail120pen-2 (circuit-breaker CFR, terkunci sbg keputusan final
# sebelumnya). TIDAK pakai --ev-obs (observasi stasiun tetap 7 fitur murni, Pers. 11).
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_3arah.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_3arah.log 2>&1 &
set -e
PY=.venv/Scripts/python.exe
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
DS30=4x
DS90=scenario_dataset_klaster12_4x_90d.json
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
PREF="--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1"

run_arm () {
  local NAME=$1; local HORIZON=$2; local DATASET=$3; local EXTRA=$4
  local COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --dataset $DATASET --horizon $HORIZON $FAIL $EXTRA"
  echo "=== $NAME ($HORIZON) ==="
  $PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
  $PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
  $PY $PIPE --mode dgr $COMMON
}

echo "########## 30 HARI ##########"
run_arm "1. Gabungan (attn+P)" 30d $DS30 "$PREF"
# 2. Attention saja @30d DILEWATI -- identik dgn master_hybrid_ppo_dgr_cwtfail120pen-2
#    yg sudah dilatih sebelumnya (kode jalur ini tak berubah), lihat outputs/ yg ada.
run_arm "3. P saja (no-attn)"  30d $DS30 "$PREF --no-station-attn"

echo "########## 90 HARI ##########"
run_arm "1. Gabungan (attn+P)" 90d $DS90 "$PREF"
run_arm "2. Attention saja"    90d $DS90 ""
run_arm "3. P saja (no-attn)"  90d $DS90 "$PREF --no-station-attn"

echo "=== SELESAI ==="
