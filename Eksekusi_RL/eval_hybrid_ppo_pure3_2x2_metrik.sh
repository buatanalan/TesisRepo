#!/bin/bash
# Evaluasi faktorial 2x2 mode aliran-murni (retrain_hybrid_ppo_pure3_2x2.sh).
# Mode CEPAT (signed|dinamis). Hanya tahap DGR yang dievaluasi -- ketiga spesialis
# adalah run PENGUKURAN (menetapkan r_star), bukan kandidat kebijakan.
#
# CARA BACA hasilnya (faktorial 2x2, semua lain konstan):
#   efek utama attention = rata(sel2,sel4) - rata(sel1,sel3)
#   efek utama Modul P   = rata(sel3,sel4) - rata(sel1,sel2)
#   interaksi            = (sel4-sel3) - (sel2-sel1)
# Baru dgn sel1 ADA, ketiga besaran itu terdefinisi -- itulah alasan sel1 wajib.
#
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_pure3_2x2_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_pure3_2x2_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
B=90d_cwtfail120pen-2_preffeat_pairout

$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_${B}_noattn_pure3        cepat   # sel1
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_${B}_pure3               cepat   # sel2
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_${B}_pg0.1_noattn_pure3  cepat   # sel3
$PY $UJI 0,1,2 90d master_hybrid_ppo_dgr_${B}_pg0.1_pure3         cepat   # sel4

echo "=== SELESAI ==="
