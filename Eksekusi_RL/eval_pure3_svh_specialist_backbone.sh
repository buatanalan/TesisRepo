#!/bin/bash
# Evaluasi TERPISAH tiap spesialis (wait/gini/acceptance) PURE3-StationVectorHead
# (`_svh`) utk backbone PPO & DDPG -- checkpoint sudah ada dari
# `jalankan_pure3_svh_backbone.sh` (tahap pretrain_specialist); skrip ini HANYA
# menjalankan evaluasi metrik yg belum dilakukan (yg lama cuma mengevaluasi arm
# DGR gabungan).
#
#   nohup bash Eksekusi_RL/eval_pure3_svh_specialist_backbone.sh > Eksekusi_RL/outputs/eval_pure3_svh_specialist_backbone.log 2>&1 &
set -e

LOCK=/tmp/eval_pure3_svh_specialist.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada eval yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_ppo_pure3_metrik.py
SEEDS=0,1,2,3,4,5,6,7,8,9

waktu() { date '+%Y-%m-%d %H:%M:%S'; }

for BACKBONE in ppo ddpg; do
  for ARM in specialist0_wait specialist1_gini specialist2_accept; do
    TAG="master_pure_${BACKBONE}_pure3_svh_${ARM}_acc1"
    echo "[$(waktu)] === Evaluasi ${BACKBONE} ${ARM} SVH (tag=${TAG}) ==="
    $PY $UJI $SEEDS 30d "$TAG" "$BACKBONE"
  done
done

echo "[$(waktu)] === SELESAI ==="
