#!/bin/bash
# Evaluasi faktorial 2x2 PURE3 pada rezim beban 6x YANG SEBENARNYA.
#
# LATAR: `eval_pure3_beban6x_metrik.sh` meneruskan TAG=90d, dan
# `_uji_master_pure_hybrid_ppo_metrik.py` mengunci `K.DS` dari TAG -- sehingga seluruh
# lengan `_load6x` selama ini dievaluasi pada dataset 4x. Terverifikasi lewat `served`:
# 16.676 pada lengan 6x, sedangkan dataset 6x berisi 25.106 permintaan.
#
# Hasil lama TIDAK dibuang: dilatih-6x/diuji-4x adalah uji TRANSFER yang sah dan sudah
# dipakai di Blok E. Skrip ini menambah sisi yang belum ada, dan menulis ke berkas
# BERBEDA (`..._metrik_90d6x.json`) sehingga hasil transfer tetap utuh.
#
#   nohup bash Eksekusi_RL/eval_pure3_beban6x_sejati.sh > Eksekusi_RL/outputs/eval_pure3_beban6x_sejati.log 2>&1 &
set -e

LOCK=/tmp/eval_pure3_6x_sejati.lock
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

# TAG '90d6x' -> scenario_dataset_klaster12_6x_90d.json (_uji_konsolidasi.py::HORIZON).
# Dgn TAG ini `_load6x` cocok antara latih & uji, jadi peringatan latih!=uji TIDAK muncul;
# kemunculannya berarti ada yang salah ketik.
$PY $UJI $SEEDS 90d6x "${B}_noattn_pure3"       cepat
$PY $UJI $SEEDS 90d6x "${B}_pure3"              cepat
$PY $UJI $SEEDS 90d6x "${B}_pg0.1_noattn_pure3" cepat
$PY $UJI $SEEDS 90d6x "${B}_pg0.1_pure3"        cepat

echo "=== SELESAI ==="
