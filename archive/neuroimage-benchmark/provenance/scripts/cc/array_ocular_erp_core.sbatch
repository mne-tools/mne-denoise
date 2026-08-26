#!/bin/bash
#SBATCH --job-name=ocular_erp_core
#SBATCH --account=rrg-kjerbi
#SBATCH --array=0-39%16
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Ocular ERP-CORE N170 arm: one task per N170 subject (all 40). EOG-DSS / EOG-regression /
# ICA+ICLabel / SSP-EOG clean blinks on held-out trials; scored on EOG coupling vs N170
# preservation. Aggregate after:
#   python scripts/run_ocular_arm.py --config configs/benchmarks/ocular_erp_core.yaml \
#       --group-only --deriv-root $SCRATCH/mne-denoise/results/ocular_erp_core
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
ECROOT=/project/rrg-kjerbi/datasets/osf/erp-core
mapfile -t SUBJECTS < <(find "$ECROOT" -name "*ses-N170_task-N170_eeg.set" 2>/dev/null \
                          | sed -E 's#.*/(sub-[0-9]+)/ses.*#\1#' | sort -u)
SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "task ${SLURM_ARRAY_TASK_ID} -> ${SUBJ} (of ${#SUBJECTS[@]} N170 subjects)"
python scripts/run_ocular_arm.py \
  --config configs/benchmarks/ocular_erp_core.yaml \
  --subject "${SUBJ}" \
  --deriv-root "${SCRATCH}/mne-denoise/results/ocular_erp_core"
