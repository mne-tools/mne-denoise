#!/bin/bash
#SBATCH --job-name=ssvep_extract
#SBATCH --account=rrg-kjerbi
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Decompress the SSVEP benchmark datasets to scratch: Tsinghua (.7z per subject) + BETA (.tar.gz
# in 10-subject batches). One-off prep so the loaders read plain .mat. ~15 GB extracted.
set -euo pipefail
DST=/scratch/sesma/ssvep_extracted
mkdir -p "$DST/tsinghua" "$DST/beta"
cd "$DST/tsinghua"
for f in /project/rrg-kjerbi/datasets/ssvep/tsinghua-benchmark/S*.mat.7z; do
  7z x -y "$f" >/dev/null
done
cp /project/rrg-kjerbi/datasets/ssvep/tsinghua-benchmark/Freq_Phase.mat "$DST/tsinghua/" 2>/dev/null || true
cd "$DST/beta"
for f in /project/rrg-kjerbi/datasets/ssvep/beta/S*.tar.gz; do
  tar -xzf "$f"
done
echo "DONE: tsinghua=$(ls $DST/tsinghua/S*.mat 2>/dev/null | wc -l) beta=$(find $DST/beta -name '*.mat' 2>/dev/null | wc -l)"
