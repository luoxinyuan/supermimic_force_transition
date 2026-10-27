#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source env.sh
export CUDA_VISIBLE_DEVICES="${CEER_GPUS:-2,3,4}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4
out="${CEER_OUT:-runs/ceer_full_seed0}"
mkdir -p "$out"
.venv/bin/torchrun --standalone --nproc_per_node=3 experiments/ceer_full/train.py \
 --command-interface ceer --motions data/processed/bones_seed_filtered_50hz.npz \
 --out "$out" --num-envs 4096 --buffer-steps 1000 --wm-batch 2048 --wm-batches 10 \
 --pi-batch 4096 --horizon 32 --iterations 320000 --wall-hours 48 \
 --eval-every 8000 --eval-clips 256 --eval-envs 256 --ckpt-every 4000 --log-every 100 "$@"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" .venv/bin/python experiments/ceer_full/evaluate_final.py --checkpoint "$out/last.pt"
.venv/bin/python experiments/ceer_full/export.py --checkpoint "$out/last.pt"
