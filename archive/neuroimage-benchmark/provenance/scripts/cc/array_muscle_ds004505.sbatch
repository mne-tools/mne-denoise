#!/bin/bash
#SBATCH --job-name=muscle_ds004505
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-25%10
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Muscle/reference-coupled arm (ds004505 dual-layer table-tennis EEG): one task per
# subject (sub-01..sub-25). Reference-free (none/ICA/ASR/rASR) + reference-aware
# (regression/TSPCA/dual-layer/iCanClean) tiers; fit on the noise-layer, score EMG
# contamination on the INDEPENDENT neck-EMG. Heavy (120 EEG + 120 noise + 8 EMG).
# Aggregate after:
#   python scripts/run_muscle_arm.py --config configs/benchmarks/muscle_ds004505.yaml \
#       --group-only --deriv-root $SCRATCH/mne-denoise/results/muscle_ds004505
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
SUBJ=$(printf "sub-%02d" "${SLURM_ARRAY_TASK_ID}")
echo "task ${SLURM_ARRAY_TASK_ID} -> ${SUBJ}"
python scripts/run_muscle_arm.py \
  --config configs/benchmarks/muscle_ds004505.yaml \
  --subject "${SUBJ}" \
  --deriv-root "${SCRATCH}/mne-denoise/results/muscle_ds004505"
