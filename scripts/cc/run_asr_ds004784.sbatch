#!/bin/bash
#SBATCH --job-name=asr_phantom
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${ASR_PHANTOM_DATA_DIR:?export the verified ds004784 root}"
: "${ASR_PHANTOM_MANIFEST:?export the repeat-specific manifest}"
: "${ASR_PHANTOM_CAMPAIGN:?published_reference or family_replication}"
CONFIG="${CONFIG:-configs/benchmarks/asr_ds004784_replication.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/mne-denoise-neuroimage-v1/asr_ds004784}"
REFERENCE_MANIFEST="${ASR_PHANTOM_REFERENCE_MANIFEST:-}"
mkdir -p "${OUTPUT_ROOT}" logs

git diff --quiet && git diff --cached --quiet || {
    echo "refusing benchmark execution from a dirty worktree" >&2
    exit 3
}

args=(
    run
    --config "${CONFIG}"
    --campaign "${ASR_PHANTOM_CAMPAIGN}"
    --dataset-root "${ASR_PHANTOM_DATA_DIR}"
    --dataset-manifest "${ASR_PHANTOM_MANIFEST}"
    --output-root "${OUTPUT_ROOT}"
    --index "${SLURM_ARRAY_TASK_ID}"
)
if [[ -n "${REFERENCE_MANIFEST}" ]]; then
    args+=(--reference-manifest "${REFERENCE_MANIFEST}")
fi
python scripts/run_asr_ds004784_replication.py "${args[@]}"
