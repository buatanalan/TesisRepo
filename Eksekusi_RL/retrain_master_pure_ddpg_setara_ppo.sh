#!/bin/bash
# Latih ulang MASTER-DDPG murni dgn anggaran GRADIEN setara PPO (2026-08-31).
#
# TEMUAN (_hitung_langkah_gradien.py, dari hasil yg SUDAH ADA): DDPG dan PPO
# memproses jumlah TRANSISI LINGKUNGAN yang nyaris sama (54.555 vs 54.856, rasio
# 1,01x) TAPI PPO menerima 3,11x lebih banyak LANGKAH GRADIEN dari data yang sama
# (18.643 vs 6.000) -- krn PPO mengulang 10 epoch per batch, sedangkan DDPG dibatasi
# `updates_per_chunk=20` terlepas dari ukuran buffer replay-nya.
#
# --updates-per-chunk 62 menaikkan anggaran gradien DDPG ke ~PPO (20*3,11=62,2).
# HANYA DDPG yang diubah -- PPO (rujukan utama) TIDAK disentuh, supaya hasil PPO yg
# sudah dilaporkan di tempat lain tetap sah.
#
# TIGA TAHAP (spesialis wait, spesialis gini, DGR) SEMUA diberi anggaran ini --
# spesialis menetapkan r_star yg jadi ACUAN DGR, jadi kualitasnya ikut menentukan
# keadilan perbandingan akhir, bukan cuma tahap DGR saja.
#
# VERIFIKASI SETELAH SELESAI (wajib, jangan dilewati):
#   python Eksekusi_RL/_hitung_langkah_gradien.py \
#     master_pure_dgr_90d_cwtfail120pen-2_upc62 master_pure_ppo_dgr_90d_cwtfail120pen-2
#   -> rasio PPO/DDPG harus mendekati 1,0x (bukan 3,11x lagi)
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure_ddpg_setara_ppo.sh > Eksekusi_RL/outputs/retrain_master_pure_ddpg_setara_ppo.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 90d \
        --wait-fail-threshold 120 --wait-fail-penalty -2.0 --updates-per-chunk 62"

echo "=== 1. Spesialis stream 0 (wait, anggaran setara PPO) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON

echo "=== 2. Spesialis stream 1 (gini, anggaran setara PPO) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON

echo "=== 3. Training DGR (r_star dari tahap 1&2, anggaran setara PPO) ==="
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
echo "WAJIB verifikasi: python Eksekusi_RL/_hitung_langkah_gradien.py master_pure_dgr_90d_cwtfail120pen-2_upc62 master_pure_ppo_dgr_90d_cwtfail120pen-2"
