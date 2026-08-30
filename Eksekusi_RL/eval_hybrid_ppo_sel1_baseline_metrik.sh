#!/bin/bash
# Evaluasi SEL-1 (attention OFF, P-paket OFF) -- melengkapi faktorial 2x2.
# Mode CEPAT (signed|dinamis). Setelah sel ini ada, keempat sel dapat dibaca sbg
# efek utama A (attention), efek utama B (P-paket), dan interaksi A x B.
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_sel1_baseline_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_sel1_baseline_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SUF=90d_cwtfail120pen-2_noattn

$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF cepat

echo "=== SELESAI ==="
