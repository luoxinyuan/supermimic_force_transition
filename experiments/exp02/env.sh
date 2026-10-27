EXP02_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$EXP02_ROOT"
export SMIMIC_PY="$EXP02_ROOT/.venv/bin/python"
source "$EXP02_ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="${EXP02_GPU:-1}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4
