#!/bin/bash
# UJI GENERALISASI faktorial 2x2 PURE3 ke kondisi lain (2026-08-30).
#
# PERTANYAAN: apakah temuan utama bertahan di luar konfigurasi baku?
#   Temuan yg diuji: sel4 (attention+P) terbaik, dan interaksi attention x P POSITIF.
#
# RANCANGAN: SATU FAKTOR BERUBAH pada satu waktu (bukan penyilangan penuh) --
# penyilangan 2x2 x 3 trust x 3 gamma = 36 kondisi x 48 run = 11+ jam dan sebagian
# besar selnya tak menjawab pertanyaan apa pun. Cukup 4 kondisi tambahan di sekitar
# baku, masing-masing 2x2 penuh (krn interaksi ADALAH temuan yang diuji, jadi keempat
# selnya wajib ada):
#
#   baku            : initial_trust 0.5, gamma 0.99   <- SUDAH ADA, tak diulang
#   A. trust rendah : initial_trust 0.3
#   B. trust tinggi : initial_trust 0.7
#   C. gamma rendah : gamma 0.95
#   D. gamma tinggi : gamma 0.999
#
# BEDA JENIS PERTANYAAN (penting saat menulis):
#   initial_trust = properti LINGKUNGAN  -> menguji validitas eksternal
#   gamma         = hiperparameter ALGORITMA -> menguji sensitivitas penyetelan
# Keduanya sah, tapi menjawab hal berbeda dan jangan dicampur dalam satu klaim.
#
# CATATAN utk gamma: efeknya TEREDAM oleh `max_step_gap=4` yang memutus rantai
# bootstrap antar-transisi berjauhan waktu -- horizon efektif sudah pendek terlepas
# dari gamma. Hasil null di sini WAJAR dan tetap layak dilaporkan.
#
# BIAYA: 4 kondisi x 4 sel x 4 tahap x 3 seed = 192 run (~4,8 jam di server).
#
# Jalankan dari root repo:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#     nohup bash Eksekusi_RL/retrain_pure3_generalisasi.sh \
#     > Eksekusi_RL/outputs/retrain_pure3_generalisasi.log 2>&1 &
set -e

LOCK=/tmp/retrain_pure3_gen.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada yang berjalan (kunci: $LOCK)."
  echo "  Periksa : pgrep -af '_run_master_pure_hybrid_ppo_pipeline'"
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
BASE="--n-train-seed 3 --n-updates 300 --rollout-steps 96 \
      --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d \
      --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
      --pure-streams --alpha-accept 1.0 \
      --pref-feature-mode --pref-pair-outcome"

# 4 sel faktorial; --pref-gate-init 0.1 = Modul P AKTIF, tanpanya gerbang 0.0 = INERT
SEL_NAMA=("sel1 attnOFF Poff" "sel2 attnON  Poff" "sel3 attnOFF PON " "sel4 attnON  PON ")
SEL_ARG=("--no-station-attn" "" "--no-station-attn --pref-gate-init 0.1" "--pref-gate-init 0.1")

jalankan_kondisi () {
  local NAMA=$1; local KOND=$2
  echo
  echo "#################### KONDISI: $NAMA ####################"
  for i in 0 1 2 3; do
    echo "===== ${SEL_NAMA[$i]} | $NAMA ====="
    local C="$BASE $KOND ${SEL_ARG[$i]}"
    $PY $PIPE --mode pretrain_specialist --stream-select 0 $C   # wait
    $PY $PIPE --mode pretrain_specialist --stream-select 1 $C   # gini
    $PY $PIPE --mode pretrain_specialist --stream-select 2 $C   # acceptance
    $PY $PIPE --mode dgr $C
  done
}

jalankan_kondisi "A. trust awal 0.3" "--initial-trust 0.3"
jalankan_kondisi "B. trust awal 0.7" "--initial-trust 0.7"
jalankan_kondisi "C. gamma 0.95"     "--gamma 0.95"
jalankan_kondisi "D. gamma 0.999"    "--gamma 0.999"

echo
echo "=== SELESAI ==="
