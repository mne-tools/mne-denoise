#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  One-time environment setup for mne-denoise on Narval
#
#  Run from a login node (or short interactive session):
#    bash scripts/setup_env.sh
#
#  This script:
#    1. Loads the required modules
#    2. Creates a Python virtualenv under $PROJECT/mne-denoise/denoise_env
#    3. Installs all dependencies + mne-denoise in editable mode
#    4. Downloads the ds003620 dataset (all 44 subjects)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_ROOT="${PROJECT}/mne-denoise"
VENV="${REPO_ROOT}/denoise_env"
export DATA_DIR="${HOME}/scratch/mnedenoise/data"

echo "=== mne-denoise Narval setup ==="
echo "Repo root : ${REPO_ROOT}"
echo "Virtualenv: ${VENV}"
echo "Data dir  : ${DATA_DIR}"
echo

# ── Step 1: Modules ──────────────────────────────────────────────────────────
echo "Loading modules..."
module purge
module load StdEnv/2023
module load python/3.11 scipy-stack/2024a

# ── Step 2: Virtual environment ──────────────────────────────────────────────
if [ ! -d "${VENV}" ]; then
    echo "Creating virtualenv at ${VENV} ..."
    python -m venv --system-site-packages "${VENV}"
else
    echo "Virtualenv already exists at ${VENV}"
fi

source "${VENV}/bin/activate"
pip install --upgrade pip

# ── Step 3: Install dependencies ─────────────────────────────────────────────
echo "Installing requirements..."
pip install -r "${REPO_ROOT}/scripts/requirements_narval.txt"

echo "Installing mne-denoise in editable mode..."
pip install -e "${REPO_ROOT}"

# Quick sanity check
python -c "import mne_denoise; print(f'mne-denoise version: {mne_denoise.__version__}')"
python -c "import mne; print(f'MNE version: {mne.__version__}')"

# ── Step 4: Download data ────────────────────────────────────────────────────
echo
echo "=== Downloading ds003620 (all 44 subjects) ==="
echo "This may take a while on first run..."
echo

SUBJECTS=$(printf "sub-%02d " $(seq 1 44))
python "${REPO_ROOT}/scripts/download_openneuro.py" \
    --dataset ds003620 \
    --task oddball \
    --subjects ${SUBJECTS}

echo
echo "=== Setup complete ==="
echo
echo "Next steps:"
echo "  1. Verify:  source ${VENV}/bin/activate"
echo "  2. Test:    python scripts/run_batch.py --subject sub-01"
echo "  3. Submit:  mkdir -p logs && sbatch scripts/narval_job.sh"
