#!/bin/bash
# Evaluasi CEPAT (signed|dinamis saja) utk retrain_hybrid_ppo_critpref.sh.
# 1 seed LATIH -> checkpoint seed0 dipakai utk ketiga run eval (seed 0,1,2),
# stokastisitas simulasi tetap 3 seed berbeda (lih. fallback di _uji_master_pure_
# hybrid_ppo_metrik.py::muat_policy). Kalau sinyalnya positif, latih ULANG dgn
# 3 seed sungguhan sebelum dikutip di tesis -- rentang antar-CHECKPOINT (bukan
# cuma antar-simulasi) belum terukur di sini.
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_critpref_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_critpref_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SUF=90d_cwtfail120pen-2_preffeat_pairout_pg0.1_critpref

echo "=== 90 hari (kritik-ber-P) ==="
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist0_wait_$SUF cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_specialist1_gini_$SUF cepat
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_$SUF cepat

echo "=== SELESAI ==="
