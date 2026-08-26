#!/bin/bash
#SBATCH --job-name=cardiac_cap
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-8%8
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
python scripts/run_cardiac_arm.py --config configs/benchmarks/cardiac_cap.yaml --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/cardiac_cap"
