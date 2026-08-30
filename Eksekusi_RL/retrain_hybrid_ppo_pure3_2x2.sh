#!/bin/bash
# EKSPERIMEN TERPISAH (2026-08-30): DGR tiga kepala dgn ALIRAN MURNI.
#
#   aliran 0 = wait (termasuk circuit-breaker CFR)   -- tertunda, individual
#   aliran 1 = gini                                   -- agregat, populasi
#   aliran 2 = acceptance                             -- segera, individual
#   `prox` & `flock` DIBUANG.
#
# ALASAN UTAMA -- menghapus `alpha` sepenuhnya. Dgn satu suku per aliran, bobot suku
# merosot jadi penskala SERAGAM aliran, dan penskalaan seragam lenyap di DUA tempat:
#   (i)  normalisasi advantage per-aliran : (c*adv - mean)/std == (adv - mean)/std
#   (ii) gap-ratio DGR                    : (c*r* - c*ret)/|c*r*| == (r* - ret)/|r*|
# Jadi TIDAK ADA proporsi internal yang perlu dikalibrasi atau dibenarkan; SELURUH
# penyeimbangan objektif ditangani `beta` (gap-ratio) yang dinamis. Ini membalik
# sifat scale-invariance yg dulu membuat `alpha_gini`/CMDP mati jadi properti desain.
#
# MOTIVASI LANGSUNG: pada eksperimen `alpha_accept` sebelumnya, acceptance dan gini
# berbagi SATU aliran, sehingga proporsinya ditentukan `alpha_accept` vs `alpha_gini`
# yg TAK PERNAH dikalibrasi. Estimasi ragam menunjukkan acceptance mendominasi ~90%
# aliran itu (gini LEVEL nyaris konstan -> hilang setelah normalisasi) -- praktis
# aliran pemerataan berubah jadi aliran kepatuhan. Mode ini menghapus masalah itu
# secara struktural, bukan dgn menyetel ulang bobot.
#
# PERUBAHAN DESAIN YG WAJIB DILAPORKAN: anti-herding jendela bergulir (Tahap 0.1)
# TIDAK aktif di mode ini -- perannya sbg objektif jaringan dianggap tumpang tindih
# dgn gini. `prox` juga dibuang (nyaris konstan, kontribusinya ke advantage
# ternormalisasi memang hampir nol).
#
# FAKTORIAL 2x2 (semua lain konstan: PPO, 90d, cwtfail120/-2, 3 seed):
#   faktor A = station attention : OFF / ON
#   faktor B = Modul P           : OFF (gerbang 0.0) / ON (gerbang 0.1)
# Catatan: acceptance kini bagian DEFINISI OBJEKTIF (hadir di semua sel), BUKAN lagi
# bagian "paket P" -- sehingga faktorial menguji pertanyaan yg lebih tajam:
# "kepatuhan sudah jadi objektif -- apakah memodelkan preferensi membantu mencapainya?"
#
# BIAYA: 4 tahap per sel (3 spesialis + 1 DGR) x 4 sel = 16 run.
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_pure3_2x2.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_pure3_2x2.log 2>&1 &
set -e

# --- PENJAGA PELUNCURAN GANDA ------------------------------------------------------
# Peluncuran ganda pd tahap LATIH jauh lebih merusak drpd di eval: dua proses menulis
# checkpoint & `training_results.json` yang SAMA, sehingga `r_star` bisa berpasangan
# dgn bobot dari run yang berbeda -- korupsi senyap, tanpa pesan error.
LOCK=/tmp/retrain_pure3_2x2.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "DITOLAK: sudah ada pelatihan pure3 yang berjalan (kunci: $LOCK)."
  echo "  Periksa : pgrep -af '_run_master_pure_hybrid_ppo_pipeline'"
  echo "  Bila itu proses mati/sisa, hapus kuncinya: rmdir $LOCK"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
# -----------------------------------------------------------------------------------

PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
BASE="--n-train-seed 3 --n-updates 300 --rollout-steps 96 \
      --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d \
      --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
      --pure-streams --alpha-accept 1.0 \
      --pref-feature-mode --pref-pair-outcome"

run_sel () {
  local NAMA=$1; local EXTRA=$2
  local C="$BASE $EXTRA"
  echo "############## $NAMA ##############"
  $PY $PIPE --mode pretrain_specialist --stream-select 0 $C   # wait
  $PY $PIPE --mode pretrain_specialist --stream-select 1 $C   # gini
  $PY $PIPE --mode pretrain_specialist --stream-select 2 $C   # acceptance
  $PY $PIPE --mode dgr $C
}

# --pref-gate-init 0.1 -> Modul P AKTIF ; tanpa flag itu -> gerbang 0.0 -> P INERT
run_sel "sel1: attn OFF, P OFF" "--no-station-attn"
run_sel "sel2: attn ON , P OFF" ""
run_sel "sel3: attn OFF, P ON " "--no-station-attn --pref-gate-init 0.1"
run_sel "sel4: attn ON , P ON " "--pref-gate-init 0.1"

echo "=== SELESAI ==="
