#!/bin/bash
# UJI GENERALISASI PURE3 ke rezim beban LEBIH BERAT (2026-08-31): 4x -> 6x.
#
# BEDA JENIS dari uji trust/gamma sebelumnya: rezim 4x DIBEKUKAN di Tahap 1
# (`common.py::SUBSTRAT`, "efek performatif baru jelas mulai ~4x, rentang trust
# 34,0x") -- SELURUH eksperimen sesi ini (termasuk PURE3, trust, gamma) berdiri di
# atas keputusan itu. Ini bukan menambah satu titik data, tapi menguji apakah
# pembekuan itu sendiri masih berlaku pada beban lebih berat.
#
# DATASET: `scenario_dataset_klaster12_6x_90d.json` (BARU, dibuat 2026-08-31 via
# `common.generate_load_dataset(load_multiplier=6.0, seed=42, n_users=2636,
# horizon_days=90)`) -- SEMUA parameter lain (kalibrasi Jabodetabek 2024, w1-w5,
# freq_i, SoC, dst) IDENTIK dgn dataset 4x kanonik, hanya `load_multiplier` berbeda.
# 25.106 permintaan (vs 16.718 di 4x, rasio 1,50x sesuai proporsi beban).
#
# FAKTORIAL 2x2 PENUH (attention x Modul P), semua lain KONSTAN dari baku PURE3:
# PPO, cwtfail120/-2, alpha_accept=1.0, pref-feature-mode+pairout, 3 seed.
#
# BIAYA: 4 sel x 4 tahap x 3 seed = 48 run. Dataset lebih besar (1,5x permintaan)
# -> perkirakan waktu per run naik proporsional, ~1,5x dari baku (~71 menit -> ~107
# menit), TAPI rollout_steps/n_updates TIDAK diubah (tetap 96/300, spy anggaran
# pelatihan per lengan tetap sebanding dgn kondisi lain, bukan proporsional jumlah
# permintaan).
#
# Jalankan dari root repo:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#     nohup bash Eksekusi_RL/retrain_pure3_beban6x.sh \
#     > Eksekusi_RL/outputs/retrain_pure3_beban6x.log 2>&1 &
set -e

LOCK=/tmp/retrain_pure3_6x.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada yang berjalan (kunci: $LOCK)."
  echo "  Periksa : pgrep -af '_run_master_pure_hybrid_ppo_pipeline'"
  echo "  Sisa mati? hapus: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
# --dataset custom (bukan "4x") -> WAJIB --horizon eksplisit (assert di pipeline)
BASE="--n-train-seed 3 --n-updates 300 --rollout-steps 96 \
      --dataset scenario_dataset_klaster12_6x_90d.json --horizon 90d \
      --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
      --pure-streams --alpha-accept 1.0 \
      --pref-feature-mode --pref-pair-outcome"

run_sel () {
  local NAMA=$1; local EXTRA=$2
  echo
  echo "===== $NAMA (beban 6x) ====="
  local C="$BASE $EXTRA"
  $PY $PIPE --mode pretrain_specialist --stream-select 0 $C   # wait
  $PY $PIPE --mode pretrain_specialist --stream-select 1 $C   # gini
  $PY $PIPE --mode pretrain_specialist --stream-select 2 $C   # acceptance
  $PY $PIPE --mode dgr $C
}

run_sel "sel1 attnOFF Poff" "--no-station-attn"
run_sel "sel2 attnON  Poff" ""
run_sel "sel3 attnOFF PON " "--no-station-attn --pref-gate-init 0.1"
run_sel "sel4 attnON  PON " "--pref-gate-init 0.1"

echo
echo "=== SELESAI ==="
