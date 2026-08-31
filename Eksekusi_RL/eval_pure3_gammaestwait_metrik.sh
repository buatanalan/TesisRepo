#!/bin/bash
# Evaluasi uji generalisasi PURE3 thd gamma_est_wait (retrain_pure3_gammaestwait.sh).
# 10 seed eval (checkpoint bergilir 4/3/3), hanya tahap DGR, mode CEPAT (signed|dinamis).
#
# `gamma_est_wait` TIDAK diteruskan sbg argumen: skrip uji MENURUNKANNYA dari tag
# (`_gw0.0279...` dst, lihat _uji_master_pure_hybrid_ppo_metrik.py), sehingga
# kebijakan selalu diuji di lingkungan yang SAMA dgn tempatnya dilatih.
#
#   nohup bash Eksekusi_RL/eval_pure3_gammaestwait_metrik.sh > Eksekusi_RL/outputs/eval_pure3_gammaestwait_metrik.log 2>&1 &
set -e

LOCK=/tmp/eval_pure3_gw.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada eval gamma_est_wait yang berjalan (kunci: $LOCK)."
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
# akhiran 2 kondisi (urutan argumen --gamma-est-wait -> tag "_gw{value:g}")
KOND=("_gw0.0279513" "_gw0.111805")
KOND_NAMA=("E. gamma_est_wait 0.02795135 (x0.5)" "F. gamma_est_wait 0.11180542 (x2)")

for k in 0 1; do
  echo "#################### ${KOND_NAMA[$k]} ####################"
  for s in 0 1 2 3; do
    $PY $UJI $SEEDS 90d "${B}${SEL[$s]}${KOND[$k]}" cepat
  done
done

echo "=== SELESAI ==="
