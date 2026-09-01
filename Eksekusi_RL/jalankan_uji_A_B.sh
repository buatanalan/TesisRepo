#!/bin/bash
# Uji A (representasi laten) + Uji B (intervensi trust terkontrol), 2026-09-01 --
# menjawab kritik "bagaimana Anda tahu LSTM benar-benar belajar memisahkan preferensi
# dari kepercayaan, bukan hanya menghafal pola urutan stasiun?" Lihat docstring di
# `_ekstrak_representasi_laten.py`, `_analisis_representasi_laten.py`,
# `_uji_intervensi_trust.py` untuk rincian tiap uji.
#
# Empat langkah SEKUENSIAL: ekstraksi P-MASTER, ekstraksi MASTER (pembanding negatif),
# analisis representasi (cepat, CPU-only, tak perlu checkpoint), intervensi trust.
#
#   nohup bash Eksekusi_RL/jalankan_uji_A_B.sh \
#     > Eksekusi_RL/outputs/uji_A_B.log 2>&1 &
set -e

LOCK=/tmp/uji_A_B.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada uji A/B yang berjalan (kunci: $LOCK)."
  echo "  Periksa : pgrep -af '_ekstrak_representasi_laten|_uji_intervensi_trust'"
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
B_PMASTER=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3
B_MASTER=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3
SEEDS=0,1,2

echo "===== Uji A, langkah 1/3: ekstraksi P-MASTER ====="
$PY Eksekusi_RL/_ekstrak_representasi_laten.py $SEEDS 90d "$B_PMASTER"

echo
echo "===== Uji A, langkah 2/3: ekstraksi MASTER (pembanding negatif) ====="
$PY Eksekusi_RL/_ekstrak_representasi_laten.py $SEEDS 90d "$B_MASTER"

echo
echo "===== Uji A, langkah 3/3: analisis representasi laten ====="
$PY Eksekusi_RL/_analisis_representasi_laten.py "$B_PMASTER" "$B_MASTER"

echo
echo "===== Uji B: intervensi trust terkontrol (P-MASTER) ====="
$PY Eksekusi_RL/_uji_intervensi_trust.py $SEEDS 90d "$B_PMASTER"

echo
echo "=== SELESAI (Uji A + Uji B) ==="
