#!/bin/bash
# Round-3 comparison queue on GPU 1, run sequentially. Each run: 12 epochs, tune-set selection, comparison set once.
#   S2: baseline with seed 2027 (seed noise)   P1: pulse items loss weight x2   E1: patch16 time frontend
cd /mnt/disk1/suyang/contest/tfuniem
export CUDA_VISIBLE_DEVICES=1
for name in exp_s2_seed2027 exp_p1_pulse_w2 exp_e1_patch16; do
  echo; echo "########## $(date '+%F %T') START $name ##########"
  /mnt/disk1/suyang/miniconda3/envs/contest/bin/python code/training/train.py \
    --config code/configs/$name.json --run-dir runs/$name 2> runs/$name.stderr
  echo "==== $name EXITED code $?" | tee -a runs/$name.stderr
done
echo "########## $(date '+%F %T') QUEUE DONE ##########"
