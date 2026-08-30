#!/bin/bash
# Tambah 2 seed (3,4) khusus kondisi trust awal 0.7, keempat sel faktorial 2x2 --
# menguatkan temuan pembalikan (sel2 unggul, sel4 kehilangan posisi) yang sejauh ini
# hanya berdiri di atas 3 checkpoint.
#
# HEMAT OTOMATIS: pipeline melanjutkan dari `{TAG}_training_results.json` -- seed 0-2
# yang sudah ada di-SKIP, hanya seed 3-4 (2 seed baru) yang dilatih.
#   -> 2 seed x 4 sel = 8 run (~1,5 mnt/run di server = ~12 menit)
#
# Jalankan dari root repo:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#     nohup bash Eksekusi_RL/retrain_pure3_it07_5seed.sh \
#     > Eksekusi_RL/outputs/retrain_pure3_it07_5seed.log 2>&1 &
set -e

LOCK=/tmp/retrain_pure3_it07.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
BASE="--n-train-seed 5 --n-updates 300 --rollout-steps 96 \
      --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d \
      --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
      --pure-streams --alpha-accept 1.0 --initial-trust 0.7 \
      --pref-feature-mode --pref-pair-outcome"

run_sel () {
  echo "===== $1 ====="
  $PY $PIPE --mode dgr $BASE $2
}

run_sel "sel1 attnOFF Poff" "--no-station-attn"
run_sel "sel2 attnON  Poff" ""
run_sel "sel3 attnOFF PON " "--no-station-attn --pref-gate-init 0.1"
run_sel "sel4 attnON  PON " "--pref-gate-init 0.1"

echo "=== SELESAI ==="
