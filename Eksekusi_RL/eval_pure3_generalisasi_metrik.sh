#!/bin/bash
# Evaluasi uji generalisasi PURE3 (retrain_pure3_generalisasi.sh) -- 4 kondisi x 4 sel.
# 10 seed eval (checkpoint bergilir 4/3/3), hanya tahap DGR, mode CEPAT (signed|dinamis).
#
# `initial_trust` TIDAK diteruskan sbg argumen: skrip uji MENURUNKANNYA dari tag
# (`_it0.3` dst), sehingga kebijakan selalu diuji di lingkungan yang SAMA dgn tempatnya
# dilatih. Ketidakcocokan latih/uji jadi mustahil terjadi karena lupa mengetik argumen.
#
#   nohup bash Eksekusi_RL/eval_pure3_generalisasi_metrik.sh > Eksekusi_RL/outputs/eval_pure3_generalisasi_metrik.log 2>&1 &
set -e

LOCK=/tmp/eval_pure3_gen.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada eval generalisasi yang berjalan (kunci: $LOCK)."
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SEEDS=0,1,2,3,4,5,6,7,8,9
B=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout

# akhiran 4 sel: sel1..sel4
SEL=("_noattn_pure3" "_pure3" "_pg0.1_noattn_pure3" "_pg0.1_pure3")
# akhiran 4 kondisi
KOND=("_it0.3" "_it0.7" "_g0.95" "_g0.999")
KOND_NAMA=("A. trust awal 0.3" "B. trust awal 0.7" "C. gamma 0.95" "D. gamma 0.999")

for k in 0 1 2 3; do
  echo "#################### ${KOND_NAMA[$k]} ####################"
  for s in 0 1 2 3; do
    $PY $UJI $SEEDS 90d "${B}${SEL[$s]}${KOND[$k]}" cepat
  done
done

echo "=== SELESAI ==="
