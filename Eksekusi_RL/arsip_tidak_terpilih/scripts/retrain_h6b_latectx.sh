#!/bin/bash
# MasterEVPPOPrefPolicySmallLateCtx (2026-08-28) -- head DASAR (tanpa attention/
# concat/separate-critic apa pun): station_encoder aktor context_dim=0 (emb_pure
# MURNI fitur stasiun, TAK tersentuh preferensi) + ctx_merge terpisah bergerbang
# nol-awal SETELAH station_encoder. Uji apakah mempersempit zona konflik gradien
# preferensi-vs-representasi-stasiun (lih. diskusi arsitektur 2026-08-28) memperbaiki
# stabilitas h6b_utama SEBELUM attention stasiun ditambahkan. HANYA 30 HARI (sesuai
# permintaan -- validasi cepat dulu sebelum commit ke 90d). Jalankan dari root repo
# di server:
#   nohup bash Eksekusi_RL/retrain_h6b_latectx.sh > Eksekusi_RL/outputs/retrain_h6b_latectx.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_ev_ppo_pipeline.py
COMMON="--pref --pref-feature-mode --pref-small --late-ctx --alpha-accept 1.0 --no-hist --n-critics 3 --beta-mode gap_ratio --beta-sigma 1.0 --reward-preset seimbang4x --forecaster vwf --skip-greedy-eval --n-eval-seed 3"

echo "=== h6b-latectx (head dasar) 30d ==="
$PY $PIPE $COMMON

echo "=== SELESAI ==="
