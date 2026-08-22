#!/bin/bash
# Kontrol: K4+equity (proksi pemerataan lokal aktif) vs K3 base (equity mati) --
# isolasi murni efek local_equity_reward, BELUM digabung accept/trust. 30d & 90d.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_equity_control.sh > Eksekusi_RL/outputs/retrain_equity_control.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--reward-preset seimbang4x --beta-mode gap_ratio --beta-sigma 1.0 --forecaster vwf"

# PENTING (2026-08-22): tag SAMA dgn run equity SEBELUM fix anti-overshoot
# (local_equity_reward). Pipeline resume via training_results.json yg sudah ada --
# hapus dulu supaya BENAR-BENAR retrain dgn reward yg sudah diperbaiki, bukan
# reuse checkpoint lama diam-diam.
rm -f Eksekusi_RL/outputs/master_ev_ppo_eq1_vwf_seimbang4x_K4_gap_sig1*

echo "=== 1. K4+equity 30d ==="
$PY $PIPE --n-critics 4 --alpha-equity 1.0 $COMMON

echo "=== 2. K4+equity 90d ==="
$PY $PIPE --n-critics 4 --alpha-equity 1.0 $COMMON --dataset $DS90 --horizon 90d

echo "=== SELESAI ==="
