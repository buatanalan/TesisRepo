#!/bin/bash
# Master-PPO dgn AMBANG-GAGAL + PENALTI TETAP pd wait_reward (2026-08-29) -- replika
# mekanisme CWT paper MASTER (Lampiran B, 2102.07359v1): EV gagal charging bila CWT
# melewati ambang, diberi penalti TETAP (bukan proporsional) spt eps_cwt=-60 paper.
# AMBANG DITETAPKAN 120 menit (2 jam, bukan 45 menit spt paper -- disesuaikan user
# 2026-08-29, kemungkinan krn rezim beban 4x simulator ini jauh lebih padat drpd
# lingkungan asli paper, 45 menit akan memicu "gagal" pd porsi kasus WAJAR yg terlalu
# besar). --wait-fail-penalty -2.0 (satuan wait_scale, sblm x alpha_wait -- skala
# sebanding klip positif 2.0 yg sudah diuji).
# Direkomendasikan MENGGANTI (bukan menambah) retrain_master_pure_ppo_clip.sh --
# klip lama cuma memotong sisi ATAS improvement (tetap 0 saat gagal jauh), mekanisme
# ini memberi PENALTI eksplisit + circuit-breaker persis spt paper thd ekor tebal.
# TIGA tahap, n_updates=300, 3 seed. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure_ppo_cwtfail.sh > Eksekusi_RL/outputs/retrain_master_pure_ppo_cwtfail.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_ppo_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d --wait-fail-threshold 120 --wait-fail-penalty -2.0"

echo "=== 1. Spesialis stream 0 (wait, cwtfail) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON

echo "=== 2. Spesialis stream 1 (gini, cwtfail) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON

echo "=== 3. Training DGR (r_star tetap dari tahap 1&2, cwtfail) ==="
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
