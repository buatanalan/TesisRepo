#!/bin/bash
# SEL-1 dari rancangan faktorial 2x2 (2026-08-30): attention OFF, P-paket OFF.
#
# Ini SATU-SATUNYA sel yang belum pernah dilatih, dan ia mengunci SELURUH rancangan:
# tanpa baseline "tanpa keduanya", efek utama attention maupun efek utama P-paket
# tidak dapat dihitung -- hanya selisih antar-sel yang terkonfound.
#
#   RANCANGAN 2 FAKTOR (semua lain KONSTAN: PPO, 90d, cwtfail120/-2, 3 seed):
#     faktor A = station attention        : OFF / ON
#     faktor B = P-paket                  : OFF / ON
#       (P-paket = pref_lstm mode fitur+hasil  +  pref_gate 0.1  +  alpha_accept 0.5,
#        DIPERLAKUKAN SEBAGAI SATU KESATUAN, bukan komponen terpisah)
#
#     sel1 attn OFF, P OFF  <- SKRIP INI (belum ada)
#     sel2 attn ON , P OFF  <- master_hybrid_ppo_dgr_90d_cwtfail120pen-2 (ADA)
#     sel3 attn OFF, P ON   <- ..._preffeat_pairout_pg0.1_noattn_acc0.5 (ADA)
#     sel4 attn ON , P ON   <- ..._preffeat_pairout_pg0.1_acc0.5        (ADA)
#
# Konfigurasi sel1: aktor = StationVectorHead(7->16->8) -> Linear(8,1) saja.
# TANPA --pref-feature-mode (jalur P tak diaktifkan sama sekali) dan
# TANPA --alpha-accept. Tag yang dihasilkan: ..._cwtfail120pen-2_noattn
#
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_hybrid_ppo_sel1_baseline.sh > Eksekusi_RL/outputs/retrain_hybrid_ppo_sel1_baseline.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_hybrid_ppo_pipeline.py
COMMON="--n-train-seed 3 --n-updates 300 --rollout-steps 96 \
        --dataset scenario_dataset_klaster12_4x_90d.json --horizon 90d \
        --wait-fail-threshold 120 --wait-fail-penalty -2.0 \
        --no-station-attn"

echo "=== SEL-1: attention OFF, P-paket OFF (90d, 3 seed) ==="
$PY $PIPE --mode pretrain_specialist --stream-select 0 $COMMON
$PY $PIPE --mode pretrain_specialist --stream-select 1 $COMMON
$PY $PIPE --mode dgr $COMMON
echo "=== SELESAI ==="
