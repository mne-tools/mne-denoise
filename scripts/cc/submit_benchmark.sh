#!/bin/bash
#SBATCH --job-name=mnedn_bench
#SBATCH --account=rrg-kjerbi             # CPU allocation (no GPU used)
#SBATCH --array=1-44%20                  # one task per subject (sub-01 … sub-44), ≤20 at once
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00                  # per subject — shorten once real timings are known
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=you@example.com    # ← uncomment & set
# ==============================================================================
#  mne-denoise CPU benchmark array job — Fir (Compute Canada)
#
#  Build the env ONCE inside an allocation (see scripts/cc/README.md):
#      salloc --account=rrg-kjerbi --time=0:45:00 --cpus-per-task=4 --mem=16G
#      source scripts/cc/fir_env.sh        # builds venv_fir on first run
#      exit
#
#  Then submit the array from a login node:
#      cd /scratch/$USER/mne-denoise
#      mkdir -p logs
#      sbatch scripts/cc/submit_benchmark.sh
#
#  Smoke-test a single subject first:
#      sbatch --array=1 scripts/cc/submit_benchmark.sh
# ==============================================================================

set -euo pipefail

# ── Environment (modules + venv activation; see fir_env.sh) ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/fir_env.sh"

# Fail fast if the env is not usable (build was skipped / broken)
python -c "import mne_denoise, mne_denoise.asr" \
    || { echo "ERROR: mne-denoise/ASR import failed — build the venv in salloc first." >&2; exit 1; }

# ── Resolve this task's subject ───────────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SUB=$(printf "sub-%02d" "${TASK_ID}")
DATASET_ID="ds003620"
echo "=== SLURM array task ${TASK_ID} → ${SUB} ==="

# ── Stage this subject to fast node-local SSD ($SLURM_TMPDIR), if present ──────
SRC="${DATA_DIR}/${DATASET_ID}"
if [ -d "${SRC}/${SUB}" ]; then
    DST="${SLURM_TMPDIR}/${DATASET_ID}"
    mkdir -p "${DST}"
    for f in dataset_description.json participants.tsv participants.json; do
        [ -f "${SRC}/${f}" ] && cp -n "${SRC}/${f}" "${DST}/${f}" || true
    done
    rsync -a "${SRC}/${SUB}/" "${DST}/${SUB}/"
    [ -d "${SRC}/derivatives" ] && rsync -a --include="*/" \
        --include="*${SUB}*" --exclude="*" "${SRC}/derivatives/" "${DST}/derivatives/" || true
    export DATA_DIR="${SLURM_TMPDIR}"     # config.py resolves DATASET_ROOT from here
    echo "=== staged ${SUB} ($(du -sh "${DST}/${SUB}" 2>/dev/null | cut -f1)) to \$SLURM_TMPDIR ==="
else
    echo "NOTE: ${SRC}/${SUB} not found — runner will fall back to its smoke path."
fi

# ── Run the benchmark for this subject ────────────────────────────────────────
cd "${REPO_ROOT}"
python scripts/cc/run_benchmark.py --slurm-array

echo "=== ${SUB} complete ==="
