#!/bin/bash
#SBATCH --job-name=line_hcp_meg
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-20%8
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
python scripts/run_line_arm.py --config configs/benchmarks/line_hcp_meg.yaml --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/line_hcp_meg"
