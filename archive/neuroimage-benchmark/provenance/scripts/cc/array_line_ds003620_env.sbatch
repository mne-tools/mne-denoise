#!/bin/bash
#SBATCH --job-name=line_ds003620_env
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-44%16
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Tier-1 line arm, env-segmented (ds003620): one task per subject (sub-01..sub-44). Each runs
# 3 mobility environments (lab/oval/campus) x {none,notch,non-spatial,ZapLine+}, R(f0)+overclean.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
python scripts/run_line_arm.py \
  --config configs/benchmarks/line_ds003620_env.yaml \
  --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/line_ds003620_env"
