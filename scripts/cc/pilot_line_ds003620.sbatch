#!/bin/bash
#SBATCH --job-name=pilot_line_ds003620
#SBATCH --account=rrg-kjerbi
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Pilot: run the line-noise runner on a few real ds003620 subjects (Tier-A line arm).
# Usage:  sbatch scripts/cc/pilot_line_ds003620.sbatch sub-001 sub-002 sub-003
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
DERIV="${SCRATCH}/mne-denoise/results/line_ds003620"
echo "host=$(hostname) python=$(which python)"
for sub in "$@"; do
  echo "=========== $sub ==========="
  python scripts/run_line_arm.py \
    --config configs/benchmarks/line_noise_ds003620.yaml \
    --subject "$sub" --deriv-root "$DERIV"
done
echo "=========== group aggregate ==========="
python scripts/run_line_arm.py \
  --config configs/benchmarks/line_noise_ds003620.yaml \
  --group-only --deriv-root "$DERIV"
echo "DONE -> $DERIV"
