#!/bin/bash
# Tambah seed tahap DGR pure3 dari 3 -> 10, untuk keempat sel faktorial 2x2.
#
# HANYA DGR. Spesialis TIDAK ditambah seed-nya, dan itu BUKAN penghematan asal-asalan:
# seluruh seed DGR memuat `r_star` dari spesialis seed 0 saja (`--specialist-seed`
# baku 0, lihat `_load_r_star` di pipeline). Menambah seed spesialis tak akan terpakai.
#
# HEMAT OTOMATIS: pipeline melanjutkan dari `{TAG}_training_results.json`, sehingga
# seed 0-2 yang sudah ada akan di-SKIP dan hanya seed 3-9 yang dilatih.
#   -> 7 seed baru x 4 sel = 28 run (~1,5 mnt/run di server = ~42 menit)
#
# PRASYARAT: retrain_hybrid_ppo_pure3_2x2.sh sudah SELESAI (spesialis + DGR 3 seed).
#
# Jalankan dari root repo:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#     nohup bash Eksekusi_RL/retrain_pure3_dgr_10seed.sh \
#     > Eksekusi_RL/outputs/retrain_pure3_dgr_10seed.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
BASE="--n-train-seed 10 --n-updates 300 --rollout-steps 96 \
      --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d \
      --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
      --pure-streams --alpha-accept 1.0 \
      --pref-feature-mode --pref-pair-outcome"

run_dgr () {
  echo "############## $1 ##############"
  $PY $PIPE --mode dgr $BASE $2
}

run_dgr "sel1: attn OFF, P OFF" "--no-station-attn"
run_dgr "sel2: attn ON , P OFF" ""
run_dgr "sel3: attn OFF, P ON " "--no-station-attn --pref-gate-init 0.1"
run_dgr "sel4: attn ON , P ON " "--pref-gate-init 0.1"

echo "=== SELESAI ==="
