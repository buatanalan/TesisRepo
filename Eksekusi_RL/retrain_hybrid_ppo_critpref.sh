#!/bin/bash
# Uji hipotesis "kritik buta P" (2026-08-30): `MasterHybridPPOCritic` menerima
# pref_hist (param TERPISAH dari aktor), menggantikan `MasterPurePPOCritic` yg
# SELALU buta P di seluruh eksperimen Hybrid sebelumnya -- lih. docstring kelas
# itu. HANYA 90 hari (sama alasan spt histK5: 30 hari terlalu sedikit riwayat
# utk P bermakna sama sekali).
#
# PENGURANGAN SKALA (2026-08-30, instruksi user): 1 SEED LATIH (bukan 3) --
# eksperimen eksplorasi murah utk cek arah sinyal dulu, BUKAN pelaporan final.
# Eval tetap 3 seed (evaluasi jauh lebih murah drpd training, tak perlu dipotong).
# Kalau sinyalnya positif, ulangi dgn 3 seed latih sebelum dikutip di tesis.
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_critpref.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_critpref.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
PREF="--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1 --critic-pref"
COMMON="--n-train-seed 1 --n-updates 300 --rollout-steps 96 --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d $FAIL $PREF"

echo "=== Gabungan + kritik-ber-P (90d, 1 seed latih) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON
echo "=== SELESAI ==="
