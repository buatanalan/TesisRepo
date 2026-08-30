#!/bin/bash
# Perluasan uji "kritik buta P" (2026-08-30) ke 3 arah, melengkapi
# retrain_hybrid_ppo_critpref.sh (yg sudah menutup Gabungan+kritikP):
#   A. Attn-saja  + kritik-ber-P -- gerbang P AKTOR tetap 0 (P aktor SENGAJA inert,
#      definisi asli "Attn-saja"), gerbang KRITIK 0.1 (aktif). Uji apakah kritik
#      sendiri diuntungkan P walau aktor tak memakainya sama sekali.
#   B. Pref-saja  + kritik-ber-P -- --no-station-attn, gerbang P AKTOR 0.1 (aktif,
#      spt "Pref-saja" asli), gerbang KRITIK 0.1.
# (Gabungan+kritikP sudah di retrain_hybrid_ppo_critpref.sh, TIDAK diulang di sini.)
#
# Sama pola pengurangan skala: 1 SEED LATIH, 90 hari saja, eval 3 seed (skrip eval
# terpisah). Kalau sinyal positif, ulangi 3 seed sebelum dikutip di tesis.
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_critpref_3arah.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_critpref_3arah.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
DS="--dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d"

run_arm () {
  local NAME=$1; local EXTRA=$2
  local COMMON="--n-train-seed 1 --n-updates 300 --rollout-steps 96 $DS $FAIL $EXTRA"
  echo "=== $NAME (90d, 1 seed latih) ==="
  $PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
  $PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
  $PY $PIPE --mode dgr $COMMON
}

# A. Attn-saja + kritik-ber-P: aktor TANPA --pref-gate-init (tetap 0.0, P aktor mati),
#    TAPI --pref-feature-mode --pref-pair-outcome WAJIB (assert kode) supaya bentuk
#    riwayat sama dgn varian P lain -- gerbangnya saja yg dijaga mati.
run_arm "A. Attn-saja + kritik-ber-P" \
  "--pref-feature-mode --pref-pair-outcome --critic-pref --critic-pref-gate-init 0.1"

# B. Pref-saja + kritik-ber-P: --no-station-attn + P aktor AKTIF (pg0.1)
run_arm "B. Pref-saja + kritik-ber-P" \
  "--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --no-station-attn --critic-pref --critic-pref-gate-init 0.1"

echo "=== SELESAI ==="
