#!/bin/bash
# Master-Hybrid PPO & DDPG dgn riwayat preferensi BERPASANGAN + BLOK HASIL (2026-08-29).
#
# Riwayat per langkah kini 12 dim:
#   [feat(a_hat) 5] ++ [feat(a) 5] ++ [complied, realized_gap_norm]
# `realized_gap_norm` = besaran yg SAMA dipakai `User.update_trust`, sehingga satu
# `pref_lstm` menduga PREFERENSI (titik acuan) DAN KEPERCAYAAN (radius) sekaligus =
# anggaran pergeseran. `hist_lstm` terpisah jadi mubazir, bukan bersaing.
#
# DUA BUG DIPERBAIKI dalam run ini -- hasil Hybrid SEBELUMNYA tidak sah utk menilai P:
#   1. PADDING: basis `RLRolloutAgent` mem-padding di DEPAN, tapi `_encode_pref` memakai
#      `pack_padded_sequence` (mensyaratkan BELAKANG) -> `pref_lstm` selama ini hanya
#      membaca baris PADDING NOL saat LATIH. Kini `pref_pad_right=True`.
#   2. EVALUASI: kedua inference agent Hybrid melewatkan `pref_hist=None` -> P netral
#      saat UJI. Kini keduanya MENDELEGASIKAN ke rollout agent (pola Kandidat A),
#      sehingga jalur pref saat uji identik dgn saat latih.
#
# Circuit-breaker wait (threshold=120/penalty=-2.0) dipertahankan -- sudah terbukti
# menstabilkan kedua lengan. TIGA tahap per lengan, n_updates=300, 3 seed.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_hybrid_pairout.sh > Eksekusi_RL/outputs/retrain_master_hybrid_pairout.log 2>&1 &
set -e
PY=.venv/bin/python
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d \
        --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
        --pref-feature-mode --pref-pair-outcome"

echo "=== [1/2] Master-Hybrid PPO ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON

echo "=== [2/2] Master-Hybrid DDPG ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ddpg_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON

echo "=== SELESAI ==="
