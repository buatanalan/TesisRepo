#!/bin/bash
# Evaluasi metrik kaya (acceptance/trust/wait/served, dst) utk semua 11 checkpoint
# _gap dari retrain_dgr_all.sh. Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/eval_dgr_all_metrik.sh > Eksekusi_RL/outputs/eval_dgr_all_metrik.log 2>&1 &
set -e
PY=.venv/bin/python
UJI=Eksekusi_RL/_uji_master_ev_ppo_metrik.py

echo "=== 1. K2 base ==="
$PY $UJI 0,1,2 30d master_ev_ppo_vwf_K2_gap

echo "=== 2. K3 base 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_vwf_K3_gap

echo "=== 3. K3 base 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_vwf_K3_gap_90d

echo "=== 4. pref_feat K2 ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_vwf_K2_gap

echo "=== 5. pref_feat K3 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_vwf_K3_gap

echo "=== 6. pref_feat_nohist K3 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_nohist_vwf_K3_gap

echo "=== 7. pref_feat_nohist K3 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_pref_feat_nohist_vwf_K3_gap_90d

echo "=== 8. pref_feat_nohist_acc1 K3 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_nohist_acc1_vwf_K3_gap

echo "=== 9. pref_feat_nohist_acc1 K3 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_pref_feat_nohist_acc1_vwf_K3_gap_90d

echo "=== 10. pref_feat_nohist_trust2 K3 30d ==="
$PY $UJI 0,1,2 30d master_ev_ppo_pref_feat_nohist_trust2_vwf_K3_gap

echo "=== 11. pref_feat_nohist_trust2 K3 90d ==="
$PY $UJI 0,1,2 90d master_ev_ppo_pref_feat_nohist_trust2_vwf_K3_gap_90d

echo "=== SEMUA EVALUASI METRIK DGR SELESAI ==="
