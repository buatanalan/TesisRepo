#!/bin/bash
# Evaluasi greedy_util/greedy_queue metodologi kaya (30d & 90d) -- pembanding
# apple-to-apple utk hasil DGR gap_ratio (lihat eval_dgr_all_metrik.sh).
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/eval_greedy_metrik_both.sh > Eksekusi_RL/outputs/eval_greedy_metrik_both.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_greedy_metrik.py

echo "=== greedy 30d ==="
$PY $UJI 0,1,2 30d

echo "=== greedy 90d ==="
$PY $UJI 0,1,2 90d

echo "=== SELESAI ==="
