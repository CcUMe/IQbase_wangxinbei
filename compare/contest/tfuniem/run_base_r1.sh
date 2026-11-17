#!/bin/bash
# Round-1 supervised training of TF-UniEM Base, single GPU 4 (guide 6.3). Progress bar on the terminal,
# plain log in runs/base_r1/train.log, errors in runs/base_r1.stderr.
cd /mnt/disk1/suyang/contest/tfuniem
export CUDA_VISIBLE_DEVICES=4
/mnt/disk1/suyang/miniconda3/envs/contest/bin/python code/training/train.py \
  --config code/configs/base_round1.json --run-dir runs/base_r1 2> runs/base_r1.stderr
echo "==== TRAINING EXITED code $?" | tee -a runs/base_r1.stderr
