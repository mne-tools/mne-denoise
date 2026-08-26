#!/bin/bash
#SBATCH --job-name=line_meg_masc
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-11%11
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Tier-B MEG-scaling (MEG-MASC, NYU KIT): one task per subject (sub-01..sub-11).
# ZapLine/ZapLine+ vs none/notch/non-spatial on the 50 Hz line, magnetometers. Aggregate:
#   grep peak_attenuation_db across results/line_meg_masc/sub-*/.../metrics.tsv
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
python scripts/run_line_arm.py \
  --config configs/benchmarks/line_meg_masc.yaml \
  --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/line_meg_masc"
