#!/bin/bash
# "Kandidat A-Simple" (2026-08-29) -- Kandidat A (retrain_h6b_latectx_stattn.sh) TAPI
# TANPA --late-ctx (ctx_merge dihapus, context balik disuntik EARLY sblm MLP pertama,
# spt MasterEVPPOPrefPolicySmall polos) & TANPA --use-station-attn (self-attention
# antar-kandidat dihapus). TUJUAN: uji apakah dua lapisan pemrosesan tambahan itu
# (yg TIDAK ada padanannya di aktor MASTER murni -- lih. diskusi arsitektur
# 2026-08-29) benar2 esensial, atau P (PDQN) + AttentionPooling kritik (MASTER) saja
# sudah cukup begitu digabung ke unit-agen-EV + reward seimbang4x. SEMUA flag lain
# IDENTIK Kandidat A (--pref-feature-mode --pref-small --alpha-accept 1.0 --no-hist
# --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x
# --forecaster vwf) supaya perbandingan adil -- HANYA dua lapisan itu yg dicabut.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_h6b_simple.sh > Eksekusi_RL/outputs/retrain_h6b_simple.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
COMMON="--pref --pref-feature-mode --pref-small --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --n-eval-seed 3"

echo "=== h6b-simple (tanpa late-ctx, tanpa station-attn) 30d ==="
$PY $PIPE $COMMON

echo "=== SELESAI ==="
