#!/bin/bash
# Master-Hybrid PPO & DDPG dgn OBSERVASI (K x 10) sesuai spesifikasi o_i Bab IV
# (2026-08-29): 7 fitur stasiun + 3 fitur PEMOHON yang disiarkan -- jarak tempuh
# relatif, SoC, dan kapasitas baterai (`build_joint_obs_master_ev`).
#
# LATAR: audit draf Bab IV menemukan lengan Hybrid terkondisi pemohon HANYA lewat
# riwayat (`pref_hist`), bukan lewat keadaan FISIK pemohon saat ini. Akibatnya agen
# tak dapat menilai jarak tiap kandidat bagi pengguna yg sedang meminta, maupun
# menalar keterjangkauan/range-anxiety -- padahal w5*(1-SoC)*Dist adalah suku yang
# secara eksplisit memodelkan kecemasan jangkauan di sisi pengguna (IV.1.4).
#
# DUA VERSI dipertahankan, TIDAK saling menggantikan:
#   - MASTER murni  : TETAP 7 fitur (Pers.11, stasiun buta thd pemohon) -- kesetiaan
#                     replikasi paper. TIDAK dijalankan di sini.
#   - Master-Hybrid : 10 fitur (flag --ev-obs) -- memenuhi formulasi Bab IV.
# Pembanding langsung: lengan Hybrid TANPA --ev-obs yang sudah ada (tag tanpa _evobs).
#
# TIGA tahap per lengan, n_updates=300, 3 seed, horizon 30d. Jalankan dari root repo:
#   nohup bash Eksekusi_RL/retrain_master_hybrid_evobs.sh > Eksekusi_RL/outputs/retrain_master_hybrid_evobs.log 2>&1 &
set -e
PY=.venv/bin/python
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d \
        --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
        --pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --ev-obs"

echo "=== [1/2] Master-Hybrid PPO (ev-obs) ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON

echo "=== [2/2] Master-Hybrid DDPG (ev-obs) ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ddpg_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
