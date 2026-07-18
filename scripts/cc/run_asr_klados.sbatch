#!/bin/bash
#SBATCH --job-name=asr_klados
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${KLADOS_DATA_DIR:?export KLADOS_DATA_DIR containing Pure_Data.mat and Contaminated_Data.mat}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/mne-denoise-neuroimage-v1/asr_klados}"
mkdir -p "${OUTPUT_ROOT}" logs

git diff --quiet && git diff --cached --quiet || {
    echo "refusing benchmark execution from a dirty worktree" >&2
    exit 3
}

python scripts/run_aasr_klados_validation.py \
    --data-dir "${KLADOS_DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}"
