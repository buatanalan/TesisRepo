#!/bin/bash
# Evaluasi 3-SEED SUNGGUHAN utk retrain_hybrid_ppo_critpref_3seed.sh (C, A, B).
# Masih mode CEPAT (signed|dinamis saja) -- ini konfirmasi arah sinyal dgn seed
# sungguhan, BUKAN pelaporan final. Begitu pola 1-seed terkonfirmasi bertahan di
# 3 seed, jalankan SEKALI LAGI tanpa "cepat" (4 mode penuh, lih. catatan di
# eval_hybrid_ppo_3arah_metrik.sh) sebelum dikutip di tesis.
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_critpref_3seed_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_critpref_3seed_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SUF_C=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_critpref
SUF_A=90d_cwtfail120pen-2_preffeat_pairout_critpref
SUF_B=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_noattn_critpref

echo "=== C. Gabungan + kritik-ber-P ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_C cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_C cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_C cepat

echo "=== A. Attn-saja + kritik-ber-P ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_A cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_A cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_A cepat

echo "=== B. Pref-saja + kritik-ber-P ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_B cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_B cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_B cepat

echo "=== SELESAI ==="
