#!/bin/bash
#SBATCH --job-name=asrv_merge
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${RESULTS_ROOT:?export the asr_variant_sensitivity results directory}"
: "${CELL_START:?export the first zero-based sensitivity cell}"
: "${CELL_STOP:?export the last zero-based sensitivity cell}"
SHARD_COUNT="${SHARD_COUNT:-24}"

if ! [[ "${CELL_START}" =~ ^[0-9]+$ && "${CELL_STOP}" =~ ^[0-9]+$ ]]; then
    echo "CELL_START and CELL_STOP must be non-negative integers" >&2
    exit 2
fi
if [ "${CELL_START}" -gt "${CELL_STOP}" ]; then
    echo "CELL_START must not exceed CELL_STOP" >&2
    exit 2
fi

for index in $(seq "${CELL_START}" "${CELL_STOP}"); do
    cell=$(printf "%02d" "${index}")
    python scripts/merge_benchmark_shards.py \
        "${RESULTS_ROOT}/cell-${cell}/shards" \
        --expected-shards "${SHARD_COUNT}"
done
