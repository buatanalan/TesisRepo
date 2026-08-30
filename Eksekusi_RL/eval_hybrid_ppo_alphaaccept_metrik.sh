#!/bin/bash
# Evaluasi utk retrain_hybrid_ppo_alphaaccept.sh (3 lengan x 3 seed, 90 hari).
# Mode CEPAT (signed|dinamis saja) -- konfirmasi arah sinyal dulu. Kalau prediksi
# teori rantai TERBUKTI (selisih gini Pref-saja vs Attn-saja membesar ke arah
# Pref-saja), jalankan ULANG tanpa "cepat" (4 mode penuh) sebelum dikutip di tesis.
#
# METRIK KUNCI yg harus dilihat pertama:
#   acc   -- apakah alpha_accept memang menaikkan kepatuhan (sanity check, WAJIB naik;
#            kalau tidak, suku rewardnya tak bekerja & sisanya tak bisa ditafsirkan)
#   gini  -- inti hipotesis: Pref-saja harus membaik LEBIH BANYAK drpd Attn-saja
#   jn_expost_frac_untung, rec_entropy, herding -- cek trade-off tersembunyi
#
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_alphaaccept_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_alphaaccept_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SUF_PREF=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_noattn_acc0.5
SUF_ATTN=90d_cwtfail120pen-2_preffeat_pairout_acc0.5
SUF_GAB=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_acc0.5

echo "=== Pref-saja + alpha_accept ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_PREF cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_PREF cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_PREF cepat

echo "=== Attn-saja + alpha_accept (kontrol) ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_ATTN cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_ATTN cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_ATTN cepat

echo "=== Gabungan + alpha_accept ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_GAB cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_GAB cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_GAB cepat

echo "=== SELESAI ==="
