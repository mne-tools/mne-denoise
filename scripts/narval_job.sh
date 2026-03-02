#!/bin/bash
#SBATCH --job-name=mnedenoise
#SBATCH --account=def-XXXXX          # ← replace with your allocation
#SBATCH --array=1-44                 # one task per subject (sub-01 … sub-44)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00              # 3 h per subject (conservative)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=you@example.com  # ← uncomment & set

# ═══════════════════════════════════════════════════════════════════════════════
#  Narval batch job — mne-denoise ds003620 pipeline
#
#  Submit from the repo root on Narval:
#    mkdir -p logs
#    sbatch scripts/narval_job.sh
#
#  Or run a single subject interactively:
#    salloc --cpus-per-task=4 --mem=16G --time=1:00:00
#    bash scripts/narval_job.sh        # uses SLURM_ARRAY_TASK_ID=1 default
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023
module load python/3.11 scipy-stack/2024a

# ── Environment ───────────────────────────────────────────────────────────────
REPO_ROOT="${PROJECT}/mne-denoise"
VENV="${REPO_ROOT}/denoise_env"
export DATA_DIR="${HOME}/scratch/mnedenoise/data"

source "${VENV}/bin/activate"

# ── Copy data to fast local SSD ($SLURM_TMPDIR) ──────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SUB=$(printf "sub-%02d" "${TASK_ID}")
DATASET_ID="ds003620"
SRC="${DATA_DIR}/${DATASET_ID}"
DST="${SLURM_TMPDIR}/${DATASET_ID}"

echo "=== Copying ${SUB} data to \$SLURM_TMPDIR ==="
mkdir -p "${DST}"

# Top-level BIDS metadata
for f in dataset_description.json participants.tsv participants.json; do
    [ -f "${SRC}/${f}" ] && cp -n "${SRC}/${f}" "${DST}/${f}" || true
done

# Subject EEG + derivatives
rsync -a "${SRC}/${SUB}/" "${DST}/${SUB}/"
if [ -d "${SRC}/derivatives/trigger_corrected/${SUB}" ]; then
    mkdir -p "${DST}/derivatives/trigger_corrected/"
    rsync -a "${SRC}/derivatives/trigger_corrected/${SUB}/" \
             "${DST}/derivatives/trigger_corrected/${SUB}/"
fi

echo "=== Data copied ($(du -sh "${DST}/${SUB}" | cut -f1)) ==="

# Override DATA_DIR so config.py resolves to SLURM_TMPDIR copy
export DATA_DIR="${SLURM_TMPDIR}"

# ── Run pipeline ──────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
echo "=== Running pipeline for ${SUB} ==="
python scripts/run_batch.py --slurm-array

echo "=== ${SUB} complete ==="
