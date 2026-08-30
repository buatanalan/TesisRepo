#!/bin/bash
# Evaluasi ulang kondisi trust awal 0.7 setelah checkpoint bertambah 3 -> 5.
# Checkpoint dipakai bergilir otomatis (`seed % n_checkpoint`) -- dgn 5 checkpoint
# dan 10 seed eval, sebarannya jadi 2/2/2/2/2 (SEIMBANG SEMPURNA, beda dari 4/3/3
# sebelumnya). Hanya perlu dijalankan ulang, kode eval tak perlu diubah.
#
#   nohup bash Eksekusi_RL/eval_pure3_it07_5seed_metrik.sh > Eksekusi_RL/outputs/eval_pure3_it07_5seed_metrik.log 2>&1 &
set -e

LOCK=/tmp/eval_pure3_it07.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada eval yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SEEDS=0,1,2,3,4,5,6,7,8,9

$PY $UJI $SEEDS 90d "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3_it0.7"       cepat
$PY $UJI $SEEDS 90d "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pure3_it0.7"               cepat
$PY $UJI $SEEDS 90d "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_noattn_pure3_it0.7"  cepat
$PY $UJI $SEEDS 90d "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3_it0.7"         cepat

echo "=== SELESAI ==="
