#!/bin/bash
#SBATCH --job-name=kaya_aasr
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${PREPARED_MANIFEST:?export the prepared Kaya manifest JSON}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH}/mne-denoise-neuroimage-v1/asr_tsai_kaya_mi/results}"
CONFIG="${CONFIG:-configs/benchmarks/asr_tsai_kaya_mi_v2.yaml}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as a zero-based array}"
SUBJECTS=(A B C E F G H I J K L M)
VARIANTS=(init mw psp psw)
CUTOFFS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
EXPECTED_TASKS=$((${#SUBJECTS[@]} * ${#VARIANTS[@]} * ${#CUTOFFS[@]}))
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${EXPECTED_TASKS}" ]; then
    echo "invalid Kaya task ${TASK_ID}/${EXPECTED_TASKS}" >&2
    exit 2
fi

CUTOFF_INDEX=$((TASK_ID % ${#CUTOFFS[@]}))
VARIANT_INDEX=$(((TASK_ID / ${#CUTOFFS[@]}) % ${#VARIANTS[@]}))
SUBJECT_INDEX=$((TASK_ID / (${#CUTOFFS[@]} * ${#VARIANTS[@]})))
SUBJECT="${SUBJECTS[${SUBJECT_INDEX}]}"
VARIANT="${VARIANTS[${VARIANT_INDEX}]}"
CUTOFF="${CUTOFFS[${CUTOFF_INDEX}]}"
OUTPUT="${RESULTS_ROOT}/subject-${SUBJECT}_${VARIANT}_cutoff-$(printf '%02d' "${CUTOFF}").json"
mkdir -p "${RESULTS_ROOT}" logs

python scripts/run_aasr_kaya_mi.py run \
    --config "${CONFIG}" \
    --prepared-manifest "${PREPARED_MANIFEST}" \
    --subject "${SUBJECT}" \
    --variant "${VARIANT}" \
    --cutoff "${CUTOFF}" \
    --output "${OUTPUT}" \
    --locked
