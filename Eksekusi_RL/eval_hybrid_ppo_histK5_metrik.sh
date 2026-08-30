#!/bin/bash
# Evaluasi CEPAT (hanya signed|dinamis, bukan 4 mode) utk retrain_hybrid_ppo_histK5.sh.
# Argumen ke-5 "cepat" pada _uji_master_pure_hybrid_ppo_metrik.py memotong waktu eval
# 4x -- dipakai krn ini eksperimen eksploratif (cek apakah K=5 layak dikejar lebih
# jauh), BUKAN pelaporan final. Kalau sinyalnya positif, jalankan ULANG tanpa "cepat"
# (4 mode penuh) sebelum dikutip di tesis -- lihat catatan di skrip eval itu sendiri.
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_histK5_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_histK5_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SUF_90=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_histK5

echo "=== 90 hari ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_90 cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_90 cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_90 cepat

echo "=== SELESAI ==="
