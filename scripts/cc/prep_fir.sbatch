#!/bin/bash
#SBATCH --job-name=mnedn_prep
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ==============================================================================
#  One-time Fir prep: build the revision-specific environment + verify the
#  P1/P2 foundation on the cluster + stage/validate ds004505 (already on /project).
#  No internet needed (no downloads). Submit from the repo root on a login node:
#      cd /scratch/$USER/mne-denoise-bench && mkdir -p logs
#      sbatch scripts/cc/prep_fir.sbatch
# ==============================================================================
set -o pipefail

# In an sbatch job BASH_SOURCE is Slurm's spooled copy, not the repo — use SLURM_SUBMIT_DIR.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}" || { echo "cannot cd to ${REPO_ROOT}"; exit 1; }
echo "repo: ${REPO_ROOT}  host: $(hostname)"

echo "=== [1/4] build + activate env (offline wheelhouse) ==="
source scripts/cc/fir_env.sh || { echo "ERROR: fir_env.sh failed"; exit 1; }
python -c "import mne_denoise, mne" || { echo "ERROR: env unusable (mne/mne_denoise import failed)"; exit 1; }
echo "=== [2/5] import smoke ==="
python - <<'PY'
import numpy, scipy, mne, openneuro, pooch, mne_denoise
from mne_denoise.benchmarks import datasets, comparators, simulation, parity, config
from mne_denoise.qa import ground_truth, preservation, coupling, structural, computational
print("imports OK | mne-denoise", mne_denoise.__version__, "| mne", mne.__version__,
      "| numpy", numpy.__version__, "| scipy", scipy.__version__)
print("comparators:", comparators.available())
PY

echo "=== [3/5] config validator probe (diagnose Fir-only config-test failures) ==="
python - <<'PY'
import pathlib, yaml
from mne_denoise.benchmarks import config as C
print("pyyaml", yaml.__version__)
p = pathlib.Path("configs/benchmarks/muscle_ds004505.yaml")
print("exists:", p.exists(), "size:", p.stat().st_size if p.exists() else None)
cfg = C.load_arm_config(p)
print("type(cfg):", type(cfg).__name__, "| arm:", cfg.get("arm"), "| runner:", cfg.get("runner"))
iss = C.validate_for_submission(cfg)
print("n_issues:", len(iss), "| first:", [str(i) for i in iss[:4]])
PY

echo "=== [4/5] benchmark unit tests on Fir (non-fatal; show reasons) ==="
python -m pytest tests/benchmarks -o addopts="" -p no:cacheprovider --tb=short -rf -q || echo "[pytest had failures]"

echo "=== [5/5] stage + validate ds004505 (resolves to /project raw_bids) ==="
python scripts/cc/stage_dataset.py ds004505 || echo "[stage rc=$?]"

echo "=== prep done ==="
