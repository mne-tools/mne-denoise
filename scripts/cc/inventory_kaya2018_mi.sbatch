#!/bin/bash
#SBATCH --job-name=kaya_inventory
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${KAYA_DATA_DIR:?export KAYA_DATA_DIR containing figshare_manifest.json}"
OUTPUT="${OUTPUT:-${SCRATCH}/mne-denoise-neuroimage-v1/kaya2018_inventory.json}"
mkdir -p "$(dirname "${OUTPUT}")" logs

python scripts/inventory_kaya2018_mi.py \
    --dataset-root "${KAYA_DATA_DIR}" \
    --output "${OUTPUT}" \
    --locked
