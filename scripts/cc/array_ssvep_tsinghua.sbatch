#!/bin/bash
#SBATCH --job-name=ssvep_tsinghua
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-35%16
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Tier-B SSVEP (Tsinghua 40-target): one task per subject (S1..S35). comb-DSS harmonic-SNR
# enhancement + CCA/DSS-CCA decoding. ~30 s/subject.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export SSVEP_ROOT=/scratch/sesma/ssvep_extracted
python scripts/run_ssvep_arm.py \
  --config configs/benchmarks/ssvep_tsinghua.yaml \
  --slurm-array \
  --deriv-root "${SCRATCH}/mne-denoise/results/ssvep_tsinghua"
