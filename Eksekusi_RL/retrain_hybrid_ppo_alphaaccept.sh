#!/bin/bash
# UJI TEORI RANTAI KEPATUHAN (2026-08-30) -- eksperimen paling terarah sesi ini.
#
# TEORI: Modul P tak punya jalur LANGSUNG ke gini. Pengaruhnya HARUS lewat rantai:
#   P (paham preferensi/trust) -> tahu SIAPA yg bisa digeser tanpa menolak
#     -> kepatuhan -> alokasi yg BENAR-BENAR terjadi -> distribusi beban -> gini
# Rantai itu PUTUS di link kedua: `alpha_accept=0.0` (MATI) sepanjang seluruh
# eksperimen Hybrid, shg kebijakan tak pernah diajari bahwa "membuat pengguna ini
# patuh" itu bernilai. `wait_reward` cuma aktif bila patuh TAPI tak menghukum
# penolakan; `Prox` sama saja patuh/tidak.
#
# PREDIKSI yg BISA GAGAL (falsifiable): begitu alpha_accept menyala, selisih gini
# antara varian BER-P (Pref-saja) vs TANPA-P (Attn-saja) MEMBESAR ke arah varian
# ber-P -- krn utk pertama kalinya P dan reward sejalan. Kalau gini tetap tak
# berbeda, teori rantai ini SALAH (atau kepatuhan di lingkungan ini didominasi
# faktor generik spt jarak/antrean, bukan personal) -- dan itu jawaban berharga juga.
#
# DESAIN SENGAJA MINIMAL: kritik LAMA (TANPA --critic-pref) supaya hasilnya TIDAK
# tercampur variabel kritik-ber-P yg terbukti tak konsisten arah. Satu variabel
# berubah saja: alpha_accept mati -> nyala.
#
# `--accept-stream global` (BAKU): acceptance masuk aliran gini+flock, BUKAN
# wait+prox. Alasan: diagnosis 2026-08-21 pd Kandidat A -- acceptance +-1 SEGERA
# menumpuk dgn wait (kecil & TERTUNDA) membuat r_bar aliran meledak 10-30x, diduga
# penyebab wait/gini liar. Sekaligus lebih selaras teori di atas (kepatuhan sbg
# PRASYARAT pemerataan, bukan kenyamanan individual).
#
# 3 SEED, 90 hari saja. alpha_accept=0.5 (mengikuti magnitudo yg dipakai Kandidat A).
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_alphaaccept.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_alphaaccept.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
ACC="--alpha-accept 0.5 --accept-stream global"
DS="--dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d"

run_arm () {
  local NAME=$1; local EXTRA=$2
  local COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 $DS $FAIL $ACC $EXTRA"
  echo "=== $NAME (90d, 3 seed, alpha_accept=0.5) ==="
  $PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
  $PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
  $PY $PIPE --mode dgr $COMMON
}

# Pref-saja + alpha_accept -- lengan UTAMA hipotesis (P aktif, attention MATI supaya
# kontribusi P terisolasi).
run_arm "Pref-saja + alpha_accept" \
  "--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --no-station-attn"

# Attn-saja + alpha_accept -- KONTROL. P aktor MATI (gerbang 0.0). Kalau lengan ini
# ikut membaik sebesar Pref-saja, berarti perbaikan datang dari alpha_accept SEMATA,
# BUKAN dari P -- pembanding yg WAJIB ada supaya klaim tak salah atribusi.
run_arm "Attn-saja + alpha_accept (kontrol)" \
  "--pref-feature-mode --pref-pair-outcome"

# Gabungan + alpha_accept -- konfigurasi terlengkap (attention DAN P aktif).
run_arm "Gabungan + alpha_accept" \
  "--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1"

echo "=== SELESAI ==="
