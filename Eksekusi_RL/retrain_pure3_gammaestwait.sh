#!/bin/bash
# UJI GENERALISASI faktorial 2x2 PURE3 thd gamma_est_wait (2026-08-31).
#
# KOREKSI atas retrain_pure3_generalisasi.sh: "gamma" yang diuji di sana adalah
# diskon PPO/GAE (--gamma, hiperparameter ALGORITMA) -- BUKAN gamma pada P_rec =
# softmax(exp(-gamma*wait)) yang mengatur sensitivitas PENGGUNA terhadap estimasi
# waktu tunggu (user.py::GAMMA_DEFAULT, properti LINGKUNGAN/perilaku pengguna).
# Skrip ini menguji parameter yang benar via --gamma-est-wait
# (marl_spklu/experiments/ablations.py::gamma_est_wait).
#
# RANCANGAN: 2 kondisi non-baku dari user.py::GAMMA_SWEEP (titik baku 0.05590271
# sudah tercakup di faktorial 4x baku yang sudah ada, tak diulang):
#   E. gamma_est_wait rendah (x0.5): 0.02795135  -- pengguna longgar/kurang sensitif
#      thd selisih waktu tunggu antar-SPKLU yang direkomendasikan
#   F. gamma_est_wait tinggi (x2)  : 0.11180542  -- pengguna tajam/sangat sensitif,
#      mendekati keputusan hampir-deterministik ke SPKLU tercepat
#
# BIAYA: 2 kondisi x 4 sel x 4 tahap x 3 seed = 96 run (~2,4 jam di server).
#
# Jalankan dari root repo:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#     nohup bash Eksekusi_RL/retrain_pure3_gammaestwait.sh \
#     > Eksekusi_RL/outputs/retrain_pure3_gammaestwait.log 2>&1 &
set -e

LOCK=/tmp/retrain_pure3_gw.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada yang berjalan (kunci: $LOCK)."
  echo "  Periksa : pgrep -af '_run_master_pure_hybrid_ppo_pipeline'"
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
BASE="--n-train-seed 3 --n-updates 300 --rollout-steps 96 \
      --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d \
      --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
      --pure-streams --alpha-accept 1.0 \
      --pref-feature-mode --pref-pair-outcome"

# 4 sel faktorial; --pref-gate-init 0.1 = Modul P AKTIF, tanpanya gerbang 0.0 = INERT
SEL_NAMA=("sel1 attnOFF Poff" "sel2 attnON  Poff" "sel3 attnOFF PON " "sel4 attnON  PON ")
SEL_ARG=("--no-station-attn" "" "--no-station-attn --pref-gate-init 0.1" "--pref-gate-init 0.1")

jalankan_kondisi () {
  local NAMA=$1; local KOND=$2
  echo
  echo "#################### KONDISI: $NAMA ####################"
  for i in 0 1 2 3; do
    echo "===== ${SEL_NAMA[$i]} | $NAMA ====="
    local C="$BASE $KOND ${SEL_ARG[$i]}"
    $PY $PIPE --mode pretrain_specialist --stream-select 0 $C   # wait
    $PY $PIPE --mode pretrain_specialist --stream-select 1 $C   # gini
    $PY $PIPE --mode pretrain_specialist --stream-select 2 $C   # acceptance
    $PY $PIPE --mode dgr $C
  done
}

jalankan_kondisi "E. gamma_est_wait 0.02795135 (x0.5)" "--gamma-est-wait 0.02795135"
jalankan_kondisi "F. gamma_est_wait 0.11180542 (x2)"   "--gamma-est-wait 0.11180542"

echo
echo "=== SELESAI ==="
