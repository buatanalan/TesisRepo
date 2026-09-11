#!/bin/bash
# Latih + evaluasi PURE3 (wait/gini/acceptance) utk backbone PPO vs DDPG, VARIAN
# StationVectorHead (bukan MLP polos) -- lihat `MasterPurePPOActorV2`/`MasterPureActorV2`
# di `marl_spklu/rl/master_pure_ppo_policy.py`/`master_pure_policy.py`. Tag arm diberi
# sufiks `_svh` -- TIDAK menimpa checkpoint/hasil PURE3-MLP lama
# (`master_pure_ppo_pure3_dgr_acc1` dst., sudah ter-commit).
#
# Urutan WAJIB per backbone: 3 spesialis (wait/gini/acceptance) dulu, baru DGR --
# mode='dgr' membaca r_star/checkpoint ketiga spesialis otomatis lewat tag.
#
#   nohup bash Eksekusi_RL/jalankan_pure3_svh_backbone.sh > Eksekusi_RL/outputs/jalankan_pure3_svh_backbone.log 2>&1 &
#
# Pantau progres:   tail -f Eksekusi_RL/outputs/jalankan_pure3_svh_backbone.log
# Batalkan:          cari PID (`ps aux | grep jalankan_pure3_svh`), lalu `kill <PID>`
set -e

LOCK=/tmp/pure3_svh_backbone.lock
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

echo "[$(waktu)] === MULAI: PPO (StationVectorHead) spesialis wait (stream 0) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0

echo "[$(waktu)] === PPO spesialis gini (stream 1) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0

echo "[$(waktu)] === PPO spesialis acceptance (stream 2) ==="
$PY $PPO_PIPE --mode pretrain_specialist --stream-select 2 --alpha-accept 1.0

echo "[$(waktu)] === PPO DGR (gabungan 3 aliran) ==="
$PY $PPO_PIPE --mode dgr --alpha-accept 1.0

echo "[$(waktu)] === MULAI: DDPG (StationVectorHead) spesialis wait (stream 0) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 0 --alpha-accept 1.0

echo "[$(waktu)] === DDPG spesialis gini (stream 1) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 1 --alpha-accept 1.0

echo "[$(waktu)] === DDPG spesialis acceptance (stream 2) ==="
$PY $DDPG_PIPE --mode pretrain_specialist --stream-select 2 --alpha-accept 1.0

echo "[$(waktu)] === DDPG DGR (gabungan 3 aliran) ==="
$PY $DDPG_PIPE --mode dgr --alpha-accept 1.0

# Nama tag berikut mengikuti pola pipeline SVH (`_svh` disisipkan sblm `_dgr`) --
# cocokkan dgn `TAG_ARM` yang benar2 dicetak di log pelatihan DGR di atas sblm
# percaya hasil evaluasi ini. Kalau beda, edit 2 baris ARM_* di bawah lalu jalankan
# ulang HANYA blok evaluasi ini (tak perlu ulang pelatihan).
ARM_PPO=master_pure_ppo_pure3_svh_dgr_acc1
ARM_DDPG=master_pure_ddpg_pure3_svh_dgr_acc1

echo "[$(waktu)] === Evaluasi PPO (arm=${ARM_PPO}) ==="
$PY $UJI $SEEDS 30d "$ARM_PPO" ppo

echo "[$(waktu)] === Evaluasi DDPG (arm=${ARM_DDPG}) ==="
$PY $UJI $SEEDS 30d "$ARM_DDPG" ddpg

echo "[$(waktu)] === SELESAI ==="
