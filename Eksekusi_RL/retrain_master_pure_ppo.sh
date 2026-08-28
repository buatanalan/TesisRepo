#!/bin/bash
# Master-PPO (2026-08-28) -- MASTER murni, tulang punggung PPO menggantikan DDPG.
# SATU tahap saja (bukan 3 spt master_pure DDPG -- gap-ratio DGR genuine-spesialis
# tak berlaku utk V(s), diganti gap-ratio berbasis return, tak butuh pra-latih
# spesialis terpisah -- lih. master_pure_ppo_trainer.py). n_updates=300, 3 seed.
# Jalankan dari root repo di server:
#   nohup bash Eksekusi_RL/retrain_master_pure_ppo.sh > Eksekusi_RL/outputs/retrain_master_pure_ppo.log 2>&1 &
set -e
PY=.venv/bin/python
PIPE=Eksekusi_RL/_run_master_pure_ppo_pipeline.py

echo "=== Master-PPO 30d ==="
$PY $PIPE --n-train-seed 3 --n-updates 300 --rollout-steps 96 --horizon 30d

echo "=== SELESAI ==="
