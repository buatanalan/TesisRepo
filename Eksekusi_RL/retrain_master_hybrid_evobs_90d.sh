#!/bin/bash
# Master-Hybrid dgn observasi (K x 10) pada horizon 90 HARI (2026-08-30).
#
# MOTIVASI: pada 30d, penambahan 3 fitur pemohon memperbaiki KERAGAMAN rekomendasi
# Hybrid-PPO secara jelas (rec_entropy 0,138 -> 0,179 = +30%; herding 0,256 -> 0,217
# = -15%) TETAPI gini praktis tak bergerak (0,116 -> 0,118). Uji 90d menjawab apakah
# keragaman itu akhirnya BERBUAH pemerataan pada horizon panjang -- sekaligus melengkapi
# perbandingan dgn lengan 90d yang sudah ada (tanpa _evobs).
#
# URUTAN SENGAJA: PPO didahulukan krn itu lengan acuan. Blok DDPG boleh DILEWATI
# (beri komentar) -- buktinya sudah kuat bahwa ia gugur: kolaps 2/3 seed pada 90d
# tanpa evobs, DAN memburuk pd 30d dgn evobs (gini 0,099 -> 0,156; wait 55,6 -> 274,9;
# seed 1 meledak). Dijalankan hanya bila ingin melengkapi tabel bukti.
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_hybrid_evobs_90d.sh > Eksekusi_RL/outputs/retrain_master_hybrid_evobs_90d.log 2>&1 &
set -e
PY=.venv/bin/python
DS90=scenario_dataset_klaster12_4x_90d.json
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --dataset $DS90 --horizon 90d \
        --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
        --pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --ev-obs"

echo "=== [1/2] Master-Hybrid PPO (ev-obs) 90d ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON

echo "=== [2/2] Master-Hybrid DDPG (ev-obs) 90d -- boleh dilewati ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ddpg_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
