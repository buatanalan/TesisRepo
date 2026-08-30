#!/bin/bash
# Ablasi panjang jendela riwayat P: K=5 (baku K=10) pada konfigurasi Gabungan
# (attn+P), HANYA 90 HARI (2026-08-30). 30d DILEWATI sengaja -- rerata panjang
# riwayat SAAT keputusan di 30 hari cuma 2,78/10 (31% keputusan riwayat KOSONG,
# lihat analisis dugaan 4), jauh di bawah K=5 sekalipun -- mengubah K tak banyak
# berarti pada horizon itu, dan presisi antar-seed 30 hari sudah terbukti terlalu
# bising utk memisahkan varian P (rentang gini antar-seed ~0,02-0,03 vs beda
# antar-varian yg cuma ~0,01). TIDAK retrain Attn-saja/P-saja/Base -- K hanya
# relevan bagi lengan yg pakai P.
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_histK5.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_histK5.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
PREF="--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --pref-hist-k 5"
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d $FAIL $PREF"

echo "=== Gabungan histK5 (90d) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON
echo "=== SELESAI ==="
