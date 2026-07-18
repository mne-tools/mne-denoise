#!/bin/bash
#SBATCH --job-name=kaya_merge
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${PREPARED_MANIFEST:?export the prepared Kaya manifest JSON}"
: "${RESULTS_ROOT:?export the directory containing terminal cell JSON files}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_ROOT}/../merged}"
mkdir -p "${OUTPUT_ROOT}" logs

python scripts/run_aasr_kaya_mi.py merge \
    --config configs/benchmarks/asr_tsai_kaya_mi.yaml \
    --prepared-manifest "${PREPARED_MANIFEST}" \
    --results-dir "${RESULTS_ROOT}" \
    --output-dir "${OUTPUT_ROOT}"
