#!/bin/bash
# Uji penilaian relatif antar-stasiun (atensi) pada keempat sel faktorial 2x2.
# Lihat docstring `_uji_penilaian_relatif_atensi.py` untuk definisi metrik M1-M5.
# MASTER (tanpa atensi) wajib menghasilkan M1 = M2 = M3 silang = 0 PERSIS -- itu
# pemeriksaan kebenaran kode. Kalau tidak nol, jangan pakai hasil lengan lain.
#
#   nohup bash Eksekusi_RL/jalankan_uji_penilaian_relatif.sh \
#     > Eksekusi_RL/outputs/uji_penilaian_relatif.log 2>&1 &
set -e

LOCK=/tmp/uji_penilaian_relatif.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada uji penilaian relatif yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_penilaian_relatif_atensi.py
B=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout
LANGKAH=10   # rekam 1 dari tiap 10 keputusan (~1.600 sampel per checkpoint pada 90 hari)

waktu() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(waktu)] === 1/4 MASTER (pemeriksaan kebenaran: metrik silang harus 0) ==="
$PY $UJI 90d "${B}_noattn_pure3" $LANGKAH

echo "[$(waktu)] === 2/4 MASTER+Atensi ==="
$PY $UJI 90d "${B}_pure3" $LANGKAH

echo "[$(waktu)] === 3/4 MASTER+P ==="
$PY $UJI 90d "${B}_pg0.1_noattn_pure3" $LANGKAH

echo "[$(waktu)] === 4/4 P-MASTER ==="
$PY $UJI 90d "${B}_pg0.1_pure3" $LANGKAH

echo "[$(waktu)] === Ringkasan ==="
$PY Eksekusi_RL/_ringkas_penilaian_relatif.py

echo "[$(waktu)] === SELESAI ==="
