#!/bin/bash
# UJI CEPAT: apakah preset `seimbang4x` (delta-gini + alpha_gini dikalibrasi ulang
# ~5,2x lebih besar) memperbaiki r_star spesialis-gini yang mendekati nol dgn
# preset `raw`? HANYA spesialis gini (bukan 3 spesialis + DGR penuh) utk PPO & DDPG,
# arsitektur SVH, 30 hari -- paling murah utk menjawab pertanyaan ini secara cepat.
#
# Pembanding (preset `raw`, SUDAH ADA):
#   PPO-SVH gini r_star  : (belum tercatat scr terpisah utk arm SVH -- pembanding
#                           utama pakai r_star spesialis-gini P-MASTER: 0,0014/0,0034/-0,00005)
#
#   nohup bash Eksekusi_RL/jalankan_uji_seimbang4x_gini.sh > Eksekusi_RL/outputs/jalankan_uji_seimbang4x_gini.log 2>&1 &
set -e

LOCK=/tmp/uji_seimbang4x_gini.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada run yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
PPO_PIPE=Eksekusi_RL/_run_master_pure_ppo_pure3_pipeline.py
DDPG_PIPE=Eksekusi_RL/_run_master_ddpg_pure3_pipeline.py

waktu() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(waktu)] === PPO+SVH spesialis gini, preset seimbang4x ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0 \
    --reward-preset seimbang4x

echo "[$(waktu)] === DDPG+SVH spesialis gini, preset seimbang4x ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0 \
    --reward-preset seimbang4x

echo "[$(waktu)] === Ringkas r_star (bandingkan dgn preset raw: 0,0014 / 0,0034 / -0,00005) ==="
find Eksekusi_RL/outputs -iname "*specialist1_gini*seimbang4x*r_star_seed*.json" \
    -exec sh -c 'echo "--- {} ---"; cat "{}"' \;

echo "[$(waktu)] === SELESAI ==="
