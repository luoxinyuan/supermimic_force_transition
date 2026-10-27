# H200 planning for the CEER-style full tracking run

The local 3×RTX A6000 run uses 4096 environments, WM batch 2048 (10 batches), and policy batch 4096 **per GPU**. Its global sizes are 12288 environments, 6144 WM examples per minibatch, and 12288 policy rollouts. H=32, UTD=0.4, BF16 MLPs, base LR=2e-4 with sqrt(3) scaling.

Observed steady iteration time on this machine is approximately 0.28 s: about 25 hours for 320000 iterations, excluding evaluation and initial setup. This measures speed, not convergence or final tracking quality.

NVIDIA specifies H200 SXM at 141 GB HBM3e and 4.8 TB/s, versus 48 GB GDDR6 and 768 GB/s for RTX A6000. These are hardware specifications, **not** a measured SuperMimic speedup. Learned world-model/BPTT matrix operations may accelerate substantially, while MuJoCo contact solves, sequential horizon steps, small kernels, Python overhead and communication can limit the overall improvement. H200 NVL/SXM, power limits, CPU allocation and GPU interconnect also matter.

For a controlled comparison, preserve global batch and effective LR:

| Hardware | Envs/GPU | WM batch/GPU | Policy batch/GPU |
|---|---:|---:|---:|
| 3×A6000 | 4096 | 2048 | 4096 |
| 2×H200 | 6144 | 3072 | 6144 |
| 1×H200 | 12288 | 6144 | 12288 |

Use `--lr-sqrt-world false --lr 0.0003464101615` for all comparison runs. Keep data, seed, horizon, precision, WM batch count, critic settings and UTD equal. Changing GPU count while leaving per-GPU batches unchanged is a different training workload, so time per iteration alone is misleading.

Run a fresh 20–30 minute pilot on H200, excluding compilation/warmup from the steady timing sample. Record simulated env-steps/s, imagined rollout steps/s, iteration time, peak memory and reward-versus-total-samples. Do not promise a convergence-time multiplier from theoretical TFLOPS. One H200 is not guaranteed to beat three A6000s end-to-end; two H200s are more likely to improve throughput, but must be measured.

Official specification sources:
- https://www.nvidia.com/en-us/data-center/h200/
- https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a6000-datasheet.pdf
