#!/bin/bash
# Latih + evaluasi PURE3 (wait/gini/acceptance) backbone PPO vs DDPG, arsitektur
# StationVectorHead (`_svh`), pada HORIZON 90 HARI -- menyamakan dgn dataset/horizon
# yg dipakai seluruh hasil ablasi UTAMA Bab V (yg lain pakai 90d; pengujian backbone
# PPO/DDPG selama ini -- termasuk versi `_svh` 30 hari sebelumnya -- masih 30 hari).
#
# Dataset: scenario_dataset_klaster12_4x_90d.json (sudah ada di root repo).
# Tag arm dapat sufiks `_90d` otomatis dari pipeline (before `_acc1`) -- TIDAK
# menimpa hasil `_svh` 30-hari yang sudah ada.
#
# Urutan WAJIB per backbone: 3 spesialis (wait/gini/acceptance) dulu, baru DGR --
# mode='dgr' membaca r_star/checkpoint ketiga spesialis otomatis lewat tag.
#
#   nohup bash Eksekusi_RL/jalankan_pure3_svh_90d_backbone.sh > Eksekusi_RL/outputs/jalankan_pure3_svh_90d_backbone.log 2>&1 &
#
# Pantau progres:   tail -f Eksekusi_RL/outputs/jalankan_pure3_svh_90d_backbone.log
# Batalkan:          cari PID (`ps aux | grep jalankan_pure3_svh_90d`), lalu `kill <PID>`
#
# PERINGATAN WAKTU: horizon 90 hari = 3x lipat langkah simulasi per rollout/eval
# dibanding 30 hari -- perkirakan durasi total run INI kira-kira 3x lipat dari
# `jalankan_pure3_svh_backbone.sh` (yg 30 hari). Jalankan dgn `nohup` & sabar.
set -e

LOCK=/tmp/pure3_svh_90d_backbone.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada run yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
PPO_PIPE=Eksekusi_RL/_run_master_pure_ppo_pure3_pipeline.py
DDPG_PIPE=Eksekusi_RL/_run_master_ddpg_pure3_pipeline.py
UJI=Eksekusi_RL/_uji_master_pure_ppo_pure3_metrik.py
SEEDS=0,1,2,3,4,5,6,7,8,9
DATASET=scenario_dataset_klaster12_4x_90d.json
HORIZON=90d

waktu() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(waktu)] === MULAI: PPO (SVH, 90d) spesialis wait (stream 0) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0 \
    --dataset "$DATASET" --horizon "$HORIZON"

echo "[$(waktu)] === PPO spesialis gini (stream 1) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0 \
    --dataset "$DATASET" --horizon "$HORIZON"

echo "[$(waktu)] === PPO spesialis acceptance (stream 2) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 2 --alpha-accept 1.0 \
    --dataset "$DATASET" --horizon "$HORIZON"

echo "[$(waktu)] === PPO DGR (gabungan 3 aliran) ==="
$PY $PPO_PIPE --mode dgr --alpha-accept 1.0 --dataset "$DATASET" --horizon "$HORIZON"

echo "[$(waktu)] === MULAI: DDPG (SVH, 90d) spesialis wait (stream 0) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0 \
    --dataset "$DATASET" --horizon "$HORIZON"

echo "[$(waktu)] === DDPG spesialis gini (stream 1) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0 \
    --dataset "$DATASET" --horizon "$HORIZON"

echo "[$(waktu)] === DDPG spesialis acceptance (stream 2) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 2 --alpha-accept 1.0 \
    --dataset "$DATASET" --horizon "$HORIZON"

echo "[$(waktu)] === DDPG DGR (gabungan 3 aliran) ==="
$PY $DDPG_PIPE --mode dgr --alpha-accept 1.0 --dataset "$DATASET" --horizon "$HORIZON"

# Nama tag berikut PERKIRAAN dari pola `{...}_svh_dgr{_horizon_suffix}{_clip_suffix}`
# (horizon=90d -> sisipan `_90d` SEBELUM `_acc1`) -- cocokkan dgn `TAG_ARM` yang
# benar2 dicetak di log pelatihan DGR di atas sblm percaya hasil evaluasi ini.
ARM_PPO=master_pure_ppo_pure3_svh_dgr_90d_acc1
ARM_DDPG=master_pure_ddpg_pure3_svh_dgr_90d_acc1

echo "[$(waktu)] === Evaluasi PPO (arm=${ARM_PPO}) ==="
$PY $UJI $SEEDS 90d "$ARM_PPO" ppo

echo "[$(waktu)] === Evaluasi DDPG (arm=${ARM_DDPG}) ==="
$PY $UJI $SEEDS 90d "$ARM_DDPG" ddpg

echo "[$(waktu)] === SELESAI ==="
