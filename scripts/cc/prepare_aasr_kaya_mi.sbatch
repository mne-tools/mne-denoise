#!/bin/bash
#SBATCH --job-name=kaya_prepare
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${KAYA_DATA_DIR:?export the verified Figshare collection root}"
: "${KAYA_INVENTORY:?export the frozen Kaya inventory JSON}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH}/mne-denoise-neuroimage-v1/asr_tsai_kaya_mi/cache}"
mkdir -p "${CACHE_ROOT}" logs

python scripts/run_aasr_kaya_mi.py prepare \
    --config configs/benchmarks/asr_tsai_kaya_mi.yaml \
    --dataset-root "${KAYA_DATA_DIR}" \
    --inventory "${KAYA_INVENTORY}" \
    --output-dir "${CACHE_ROOT}" \
    --locked
