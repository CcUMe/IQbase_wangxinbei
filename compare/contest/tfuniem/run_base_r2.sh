#!/bin/bash
# Round-2 clean baseline: per-length BN (LenBN), tune subset excluded from training, per-epoch selection on the
# tune subset, comparison set scored once at the end. Single GPU 1.
cd /mnt/disk1/suyang/contest/tfuniem
export CUDA_VISIBLE_DEVICES=1
/mnt/disk1/suyang/miniconda3/envs/contest/bin/python code/training/train.py \
  --config code/configs/base_r2.json --run-dir runs/base_r2 2> runs/base_r2.stderr
echo "==== TRAINING EXITED code $?" | tee -a runs/base_r2.stderr
