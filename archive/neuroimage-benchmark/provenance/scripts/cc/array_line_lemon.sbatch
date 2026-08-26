#!/bin/bash
#SBATCH --job-name=line_lemon
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-40%16
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets TMPDIR="${SLURM_TMPDIR:-/tmp}"
python scripts/run_line_arm.py --config configs/benchmarks/line_lemon.yaml --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/line_lemon"
