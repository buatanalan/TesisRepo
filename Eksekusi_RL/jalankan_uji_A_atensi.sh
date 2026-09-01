#!/bin/bash
# Uji A' (bobot attention antar-stasiun), 2026-09-01 -- menjawab kritik bahwa klaim
# Bab IV soal attention "menyadari persaingan antar-kandidat" belum diuji pd
# mekanismenya sendiri. Lihat docstring `_ekstrak_attensi_stasiun.py` utk tiga sinyal
# yang diuji (korelasi bobot-vs-utilisasi, collision rate, sensitivitas komposisi).
#
#   nohup bash Eksekusi_RL/jalankan_uji_A_atensi.sh \
#     > Eksekusi_RL/outputs/uji_A_atensi.log 2>&1 &
set -e

LOCK=/tmp/uji_A_atensi.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada uji atensi yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
B_PMASTER=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3
B_MASTER_ATENSI=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pure3
B_MASTER=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3
SEEDS=0,1,2

echo "===== Ekstraksi 1/4: P-MASTER (bobot + collision) ====="
$PY Eksekusi_RL/_ekstrak_attensi_stasiun.py $SEEDS 90d "$B_PMASTER"

echo
echo "===== Ekstraksi 2/4: MASTER+Atensi (bobot + collision) ====="
$PY Eksekusi_RL/_ekstrak_attensi_stasiun.py $SEEDS 90d "$B_MASTER_ATENSI"

echo
echo "===== Ekstraksi 3/4: MASTER (collision saja -- tak punya attention) ====="
$PY Eksekusi_RL/_ekstrak_attensi_stasiun.py $SEEDS 90d "$B_MASTER" --collision-saja

echo
echo "===== Ekstraksi 4/4: greedy (collision saja, pembanding) ====="
$PY Eksekusi_RL/_ekstrak_attensi_stasiun.py $SEEDS 90d greedy --collision-saja

echo
echo "===== Analisis Uji A' ====="
$PY Eksekusi_RL/_analisis_attensi_stasiun.py "$B_PMASTER" "$B_MASTER_ATENSI" "$B_MASTER" greedy

echo
echo "=== SELESAI (Uji A') ==="
