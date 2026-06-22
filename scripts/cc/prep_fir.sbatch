#!/bin/bash
#SBATCH --job-name=mnedn_prep
#SBATCH --account=rrg-kjerbi
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ==============================================================================
#  One-time Fir prep: build venv_fir (offline, Alliance wheelhouse) + verify the
#  P1/P2 foundation on the cluster + stage/validate ds004505 (already on /project).
#  No internet needed (no downloads). Submit from the repo root on a login node:
#      cd /scratch/$USER/mne-denoise-bench && mkdir -p logs
#      sbatch scripts/cc/prep_fir.sbatch
# ==============================================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== [1/4] build + activate env (offline wheelhouse) ==="
source scripts/cc/fir_env.sh

echo "=== [2/4] import smoke ==="
python - <<'PY'
import mne, openneuro, pooch, mne_denoise
from mne_denoise.benchmarks import datasets, comparators, simulation, parity, config
from mne_denoise.qa import ground_truth, preservation, coupling, structural, computational
print("imports OK | mne-denoise", mne_denoise.__version__, "| mne", mne.__version__)
print("comparators:", comparators.available())
PY

echo "=== [3/4] benchmark unit tests on Fir ==="
python -m pytest tests/benchmarks -q -o addopts="" -p no:cacheprovider

echo "=== [4/4] stage + validate ds004505 (resolves to /project raw_bids) ==="
python scripts/cc/stage_dataset.py ds004505

echo "=== prep done (rc summary above) ==="
