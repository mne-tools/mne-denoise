#!/bin/bash
#SBATCH --job-name=line_ds003620
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-44%24
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Full line_ds003620 arm: one task per subject (sub-01..sub-44), each runs every
# method + the ZapLine n_remove sweep on identical held-out data. Aggregate after:
#   python scripts/run_line_arm.py --config configs/benchmarks/line_noise_ds003620.yaml \
#       --group-only --deriv-root $SCRATCH/mne-denoise/results/line_ds003620
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
python scripts/run_line_arm.py \
  --config configs/benchmarks/line_noise_ds003620.yaml \
  --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/line_ds003620"
