#!/bin/bash
# Retrain SEMUA konfigurasi n_critics>1 dengan --beta-mode gap_ratio (DGR asli).
# Sebelumnya beta_mode diam-diam default "fixed" (bobot seragam 1/K, TAK adaptif) di
# SELURUH eksperimen K2/K3 sesi ini -- lihat commit "fix: DGR (beta_mode=gap_ratio) tak
# pernah aktif ...". Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_dgr_all.sh > Eksekusi_RL/outputs/retrain_dgr_all.log 2>&1 &
set -e
PY=.venv/Scripts/python.exe
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json

echo "=== 1. K2 base ==="
$PY $PIPE --n-critics 2 --forecaster vwf --beta-mode gap_ratio

echo "=== 2. K3 base 30d ==="
$PY $PIPE --n-critics 3 --forecaster vwf --beta-mode gap_ratio

echo "=== 3. K3 base 90d ==="
$PY $PIPE --n-critics 3 --forecaster vwf --beta-mode gap_ratio --dataset $DS90 --horizon 90d

echo "=== 4. pref_feat K2 ==="
$PY $PIPE --pref --pref-feature-mode --n-critics 2 --forecaster vwf --beta-mode gap_ratio

echo "=== 5. pref_feat K3 30d ==="
$PY $PIPE --pref --pref-feature-mode --n-critics 3 --forecaster vwf --beta-mode gap_ratio

echo "=== 6. pref_feat_nohist K3 30d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 --forecaster vwf --beta-mode gap_ratio

echo "=== 7. pref_feat_nohist K3 90d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 --forecaster vwf --beta-mode gap_ratio --dataset $DS90 --horizon 90d

echo "=== 8. pref_feat_nohist_acc1 K3 30d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 --forecaster vwf --beta-mode gap_ratio --alpha-accept 1

echo "=== 9. pref_feat_nohist_acc1 K3 90d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 --forecaster vwf --beta-mode gap_ratio --alpha-accept 1 --dataset $DS90 --horizon 90d

echo "=== 10. pref_feat_nohist_trust2 K3 30d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 --forecaster vwf --beta-mode gap_ratio --alpha-trust 2

echo "=== 11. pref_feat_nohist_trust2 K3 90d ==="
$PY $PIPE --pref --pref-feature-mode --no-hist --n-critics 3 --forecaster vwf --beta-mode gap_ratio --alpha-trust 2 --dataset $DS90 --horizon 90d

echo "=== SEMUA RETRAIN DGR SELESAI ==="
