#!/bin/bash
# Konfirmasi 3 SEED SUNGGUHAN (2026-08-30) utk ketiga varian kritik-ber-P yg
# sinyalnya sudah konsisten & meyakinkan di 1 seed (retrain_hybrid_ppo_critpref.sh
# + retrain_hybrid_ppo_critpref_3arah.sh): naik acc/wait/trust/gini, turun
# rec_entropy/herding/jn_expost_frac_untung -- pola SAMA di A dan B, tinggal
# dikonfirmasi bukan kebetulan inisialisasi bobot tunggal.
#
# BEDA dari skrip 1-seed: --n-train-seed 3 (bukan 1) di KETIGANYA. 90 hari saja
# (30 hari tetap dilewati, alasan sama spt sebelumnya -- riwayat terlalu tipis).
#
# Jalankan dari root repo di server (durasi ~3x skrip 1-seed, krn 3 checkpoint
# independen per varian, bukan cuma 1):
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_critpref_3seed.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_critpref_3seed.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
DS="--dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d"

run_arm () {
  local NAME=$1; local EXTRA=$2
  local COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 $DS $FAIL $EXTRA"
  echo "=== $NAME (90d, 3 seed) ==="
  $PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
  $PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
  $PY $PIPE --mode dgr $COMMON
}

# C. Gabungan + kritik-ber-P: attention DAN P aktor aktif (pg0.1)
run_arm "C. Gabungan + kritik-ber-P" \
  "--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --critic-pref --critic-pref-gate-init 0.1"

# A. Attn-saja + kritik-ber-P: P aktor MATI (tanpa --pref-gate-init, tetap 0.0)
run_arm "A. Attn-saja + kritik-ber-P" \
  "--pref-feature-mode --pref-pair-outcome --critic-pref --critic-pref-gate-init 0.1"

# B. Pref-saja + kritik-ber-P: --no-station-attn, P aktor aktif (pg0.1)
run_arm "B. Pref-saja + kritik-ber-P" \
  "--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --no-station-attn --critic-pref --critic-pref-gate-init 0.1"

echo "=== SELESAI ==="
