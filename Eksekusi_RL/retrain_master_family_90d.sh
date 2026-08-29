#!/bin/bash
# Horizon 90 HARI utk keluarga MASTER: murni & Hybrid, PPO & DDPG (2026-08-29).
#
# MOTIVASI (temuan 30d yg memaksa uji ini):
#   tr_p25 DAN tr_p50 keduanya PERSIS 0,5000 -- artinya 50-75% pengguna TAK PERNAH
#   mengalami satu pun pembaruan kepercayaan dlm 30 hari (rata2 hanya 3,2 trip/pengguna,
#   dan `User.update_trust` hanya jalan pd trip PATUH). `trust_p10` bahkan tetap persis
#   0,5000 sampai hari ke-7. Akibatnya komponen KEPERCAYAAN pd modul laten KELAPARAN
#   SINYAL, dan seluruh bukti M1 (frac_trust_rendah turun, tr_p10 naik) hanya terjadi
#   di EKOR populasi. Pada 90 hari trip/pengguna naik ~3x (=9,6) sehingga median pengguna
#   benar-benar bergerak -- barulah manfaat modul laten bisa dinilai tanpa konfoun ini.
#
# Konfigurasi SENGAJA disamakan dgn run 30d masing-masing lengan supaya 30d<->90d
# langsung sebanding:
#   murni  : --wait-fail-threshold 120 --wait-fail-penalty -2.0
#   Hybrid : idem + --pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1
#
# Pembanding sudah tersedia: Kandidat A 90d (uji_master_ev_ppo_pref_feat_small_latectx_
# nohist_acc1_stattn_vwf_seimbang4x_K3_gap_sig1_90d_metrik_90d.json).
#
# PERINGATAN DURASI: 4 lengan x 3 tahap x 3 seed, tiap rollout ~3x lebih panjang drpd
# 30d. Perkirakan ~3x total waktu retrain 30d. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_family_90d.sh > Eksekusi_RL/outputs/retrain_master_family_90d.log 2>&1 &
set -e
PY=.venv/bin/python
DS90=scenario_dataset_klaster12_4x_90d.json
BUDGET="--n-train-seed 3 --n-updates 300 --rollout-steps 96 --dataset $DS90 --horizon 90d"
FAIL="--wait-fail-threshold 120 --wait-fail-penalty -2.0"
PREF="--pref-feature-mode --pref-pair-outcome --pref-gate-init 0.1"

MURNI="$BUDGET $FAIL"
HYBRID="$BUDGET $FAIL $PREF"

echo "=== [1/4] MASTER-DDPG murni 90d ==="
PIPE=Eksekusi_RL/_run_master_pure_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $MURNI
$PY $PIPE --mode pretrain_specialist --stream-select 1 $MURNI
$PY $PIPE --mode dgr $MURNI

echo "=== [2/4] MASTER-PPO murni 90d ==="
PIPE=Eksekusi_RL/_run_master_pure_ppo_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $MURNI
$PY $PIPE --mode pretrain_specialist --stream-select 1 $MURNI
$PY $PIPE --mode dgr $MURNI

echo "=== [3/4] Master-Hybrid DDPG 90d ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ddpg_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $HYBRID
$PY $PIPE --mode pretrain_specialist --stream-select 1 $HYBRID
$PY $PIPE --mode dgr $HYBRID

echo "=== [4/4] Master-Hybrid PPO 90d ==="
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
$PY $PIPE --mode pretrain_specialist --stream-select 0 $HYBRID
$PY $PIPE --mode pretrain_specialist --stream-select 1 $HYBRID
$PY $PIPE --mode dgr $HYBRID

echo "=== SELESAI ==="
