#!/bin/bash
# Evaluasi faktorial 2x2 mode aliran-murni (pure3) -- 10 SEED, HANYA TAHAP DGR.
#
# Spesialis SENGAJA TIDAK dievaluasi: ketiganya adalah run PENGUKURAN yang tugasnya
# hanya menetapkan `r_star` sebagai plafon acuan DGR, bukan kandidat kebijakan.
# Kebijakan yang dipakai dan dilaporkan selalu hasil tahap DGR.
#
# PRASYARAT: DGR sudah dilatih 10 seed (retrain_pure3_dgr_10seed.sh). Bila baru 3 seed,
# eval TETAP jalan tapi seed 3-9 jatuh ke *fallback* bobot seed 0 -- agregatnya jadi
# timpang (checkpoint 0 terpakai 8 dari 10 sampel). Waspadai baris "[fallback]" di log;
# kalau muncul, hentikan dan latih dulu sampai 10 seed.
#
# Mode CEPAT (signed|dinamis saja). Setelah polanya jelas, jalankan ulang tanpa "cepat"
# (4 mode: abs/signed x dinamis/beku) untuk angka yang dilaporkan di tesis.
#
# CARA BACA (faktorial 2x2, semua lain konstan):
#   efek utama attention = rata(sel2,sel4) - rata(sel1,sel3)
#   efek utama Modul P   = rata(sel3,sel4) - rata(sel1,sel2)
#   interaksi            = (sel4-sel3) - (sel2-sel1)
# Bandingkan tiap besaran itu terhadap simpangan baku ANTAR-SEED di dalam sel --
# efek yang lebih kecil dari simpangan baku bukan sinyal.
#
#   nohup bash Eksekusi_RL/eval_hybrid_ppo_pure3_2x2_metrik.sh > Eksekusi_RL/outputs/eval_hybrid_ppo_pure3_2x2_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_pure_hybrid_ppo_metrik.py
SEEDS=0,1,2,3,4,5,6,7,8,9
B=90d_cwtfail120pen-2_preffeat_pairout

$PY $UJI $SEEDS 90d master_hybrid_ppo_dgr_${B}_noattn_pure3        cepat   # sel1 attnOFF Poff
$PY $UJI $SEEDS 90d master_hybrid_ppo_dgr_${B}_pure3               cepat   # sel2 attnON  Poff
$PY $UJI $SEEDS 90d master_hybrid_ppo_dgr_${B}_pg0.1_noattn_pure3  cepat   # sel3 attnOFF PON
$PY $UJI $SEEDS 90d master_hybrid_ppo_dgr_${B}_pg0.1_pure3         cepat   # sel4 attnON  PON

echo "=== SELESAI ==="
echo "Periksa log: bila ada baris '[fallback]', berarti seed DGR belum lengkap 10."
