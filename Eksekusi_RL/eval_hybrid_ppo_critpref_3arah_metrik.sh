#!/bin/bash
# Evaluasi CEPAT (signed|dinamis saja) utk retrain_hybrid_ppo_critpref_3arah.sh.
# 1 seed latih -> fallback ke checkpoint seed0 utk eval seed 1,2 (stokastisitas
# simulasi tetap 3 seed berbeda, lih. catatan di eval_hybrid_ppo_critpref_metrik.sh).
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_critpref_3arah_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_critpref_3arah_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SUF_A=90d_cwtfail120pen-2_preffeat_pairout_critpref
SUF_B=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_noattn_critpref

echo "=== A. Attn-saja + kritik-ber-P ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_A cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_A cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_A cepat

echo "=== B. Pref-saja + kritik-ber-P ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF_B cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF_B cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF_B cepat

echo "=== SELESAI ==="
