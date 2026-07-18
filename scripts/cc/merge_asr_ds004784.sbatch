#!/bin/bash
#SBATCH --job-name=asr_ph_merge
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${ASR_PHANTOM_CAMPAIGN:?published_reference or family_replication}"
CONFIG="${CONFIG:-configs/benchmarks/asr_ds004784_replication.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/mne-denoise-neuroimage-v1/asr_ds004784}"

python scripts/run_asr_ds004784_replication.py merge \
    --config "${CONFIG}" \
    --campaign "${ASR_PHANTOM_CAMPAIGN}" \
    --output-root "${OUTPUT_ROOT}"

