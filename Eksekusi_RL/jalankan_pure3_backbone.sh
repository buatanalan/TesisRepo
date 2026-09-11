#!/bin/bash
# Latih + evaluasi PURE3 (wait/gini/acceptance) utk perbandingan backbone PPO vs DDPG
# (dasar Tabel VI.1/VI.2) -- menggantikan basis reward LAMA (2 aliran) yg dipakai
# pipeline `_run_master_pure_ppo_pipeline.py`/`_run_master_ddpg_pipeline.py` (§5.2),
# supaya konsisten dgn skema PURE3 yg dipakai seluruh hasil ablasi utama (Bab V).
#
# Urutan WAJIB per backbone: 3 spesialis (wait/gini/acceptance) dulu, baru DGR --
# mode='dgr' membaca r_star/checkpoint ketiga spesialis otomatis lewat tag.
#
#   nohup bash Eksekusi_RL/jalankan_pure3_backbone.sh > Eksekusi_RL/outputs/jalankan_pure3_backbone.log 2>&1 &
#
# Pantau progres:   tail -f Eksekusi_RL/outputs/jalankan_pure3_backbone.log
# Batalkan:          cari PID (`ps aux | grep jalankan_pure3`), lalu `kill <PID>`
set -e

LOCK=/tmp/pure3_backbone.lock
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

waktu() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(waktu)] === MULAI: PPO spesialis wait (stream 0) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0

echo "[$(waktu)] === PPO spesialis gini (stream 1) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0

echo "[$(waktu)] === PPO spesialis acceptance (stream 2) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 2 --alpha-accept 1.0

echo "[$(waktu)] === PPO DGR (gabungan 3 aliran) ==="
$PY $PPO_PIPE --mode dgr --alpha-accept 1.0

echo "[$(waktu)] === MULAI: DDPG spesialis wait (stream 0) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0

echo "[$(waktu)] === DDPG spesialis gini (stream 1) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0

echo "[$(waktu)] === DDPG spesialis acceptance (stream 2) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 2 --alpha-accept 1.0

echo "[$(waktu)] === DDPG DGR (gabungan 3 aliran) ==="
$PY $DDPG_PIPE --mode dgr --alpha-accept 1.0

# CATATAN: nama tag arm berikut (`master_pure_ppo_pure3_dgr_acc1` /
# `master_pure_ddpg_pure3_dgr_acc1`) adalah PERKIRAAN dari pola penamaan skrip
# pipeline -- SEBELUM evaluasi dijalankan, cocokkan dulu dgn `TAG_ARM` yang benar2
# dicetak pada log pelatihan DGR di atas. Kalau beda, edit 2 baris ARM_* di bawah
# lalu jalankan ulang blok evaluasi ini secara terpisah (tak perlu ulang pelatihan).
ARM_PPO=master_pure_ppo_pure3_dgr_acc1
ARM_DDPG=master_pure_ddpg_pure3_dgr_acc1

echo "[$(waktu)] === Evaluasi PPO (arm=${ARM_PPO}) ==="
$PY $UJI $SEEDS 30d "$ARM_PPO" ppo

echo "[$(waktu)] === Evaluasi DDPG (arm=${ARM_DDPG}) ==="
$PY $UJI $SEEDS 30d "$ARM_DDPG" ddpg

echo "[$(waktu)] === SELESAI ==="
