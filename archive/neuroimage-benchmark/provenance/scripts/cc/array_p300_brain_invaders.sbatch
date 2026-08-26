#!/bin/bash
#SBATCH --job-name=p300_brain_invaders
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-25%16
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Tier-B P300 generalization (BrainInvaders 2012): one task per subject (sub-01..sub-25).
# DSS vs none/PCA on target vs non-target P300 (held-out), at Pz. Aggregate after:
#   python scripts/run_evoked_arm.py --config configs/benchmarks/p300_brain_invaders.yaml \
#       --group-only --deriv-root $SCRATCH/mne-denoise/results/p300_brain_invaders
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
python scripts/run_evoked_arm.py \
  --config configs/benchmarks/p300_brain_invaders.yaml \
  --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/p300_brain_invaders"
