#!/bin/bash
# Evaluasi faktorial 2x2 PURE3 pada beban 6x (retrain_pure3_beban6x.sh).
# 10 seed eval (checkpoint bergilir), hanya tahap DGR, mode CEPAT (signed|dinamis).
#   nohup bash Eksekusi_RL/eval_pure3_beban6x_metrik.sh > Eksekusi_RL/outputs/eval_pure3_beban6x_metrik.log 2>&1 &
set -e

LOCK=/tmp/eval_pure3_6x.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada eval yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SEEDS=0,1,2,3,4,5,6,7,8,9
B=master_hybrid_ppo_dgr_90d_load6x_cwtfail120pen-2_preffeat_pairout

$PY $UJI $SEEDS 90d "${B}_noattn_pure3"       cepat
$PY $UJI $SEEDS 90d "${B}_pure3"              cepat
$PY $UJI $SEEDS 90d "${B}_pg0.1_noattn_pure3" cepat
$PY $UJI $SEEDS 90d "${B}_pg0.1_pure3"        cepat

echo "=== SELESAI ==="
