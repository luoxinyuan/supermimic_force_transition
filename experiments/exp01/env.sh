# Source from any directory; GPU index refers to the host before CUDA remapping.
EXP01_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$EXP01_ROOT"
export SMIMIC_PY="$EXP01_ROOT/.venv/bin/python"
source "$EXP01_ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="${EXP01_GPU:-1}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=4
