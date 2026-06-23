#!/bin/bash
#SBATCH --job-name=evoked_ds000117
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-16%8
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# M170 evoked-MEG arm (ds000117 faces): one task per subject (sub-01..sub-16). Raw FIF,
# magnetometers, faces (Famous+Unfamiliar) vs Scrambled across 6 runs; DSS vs none/PCA on
# the held-out M170 (train-derived face-selective template) + split-half reliability.
# Heavy (6 MEG runs/subject at 1100 Hz). Aggregate after:
#   python scripts/run_evoked_arm.py --config configs/benchmarks/evoked_ds000117.yaml \
#       --group-only --deriv-root $SCRATCH/mne-denoise/results/evoked_ds000117
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
SUBJ=$(printf "sub-%02d" "${SLURM_ARRAY_TASK_ID}")
echo "task ${SLURM_ARRAY_TASK_ID} -> ${SUBJ}"
python scripts/run_evoked_arm.py \
  --config configs/benchmarks/evoked_ds000117.yaml \
  --subject "${SUBJ}" \
  --deriv-root "${SCRATCH}/mne-denoise/results/evoked_ds000117"
