#!/bin/bash
# ==============================================================================
#  Fir (Digital Research Alliance / Compute Canada) environment for mne-denoise
#  CPU-only benchmarking.  SOURCE this file (do not execute it):
#
#      source scripts/cc/fir_env.sh
#
#  It (1) loads the StdEnv/2023 + python + scipy-stack modules, (2) creates the
#  virtualenv on first use and installs mne-denoise editable + extra deps, and
#  (3) activates the venv and exports thread / cache / data env vars.
#
#  IMPORTANT (Alliance rule): the FIRST source builds the venv (pip install) and
#  must run inside an `salloc` allocation, NEVER on a login node.  Subsequent
#  sources (e.g. from the sbatch job) only activate — they are cheap and safe.
#  See scripts/cc/README.md.
# ==============================================================================

# ── Locate repo root (this file lives at <repo>/scripts/cc/fir_env.sh) ─────────
_FIR_SELF="${BASH_SOURCE[0]:-$0}"
_FIR_DIR="$(cd "$(dirname "${_FIR_SELF}")" && pwd)"
export MNE_DENOISE_REPO="${MNE_DENOISE_REPO:-$(cd "${_FIR_DIR}/../.." && pwd)}"
export MNE_DENOISE_VENV="${MNE_DENOISE_VENV:-${MNE_DENOISE_REPO}/venv_fir}"

# Persistent data + results live on /scratch (matches scripts/config.py defaults)
export DATA_DIR="${DATA_DIR:-${HOME}/scratch/mnedenoise/data}"

echo "=== mne-denoise Fir environment ==="
echo "  repo  : ${MNE_DENOISE_REPO}"
echo "  venv  : ${MNE_DENOISE_VENV}"
echo "  data  : ${DATA_DIR}"
echo "  host  : $(hostname)"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge 2>/dev/null
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 scipy-stack/2024a 2>/dev/null \
    || module load python/3.11 scipy-stack 2>/dev/null \
    || echo "WARNING: could not load python/3.11 + scipy-stack modules"

# ── Build the venv on first use (Alliance: must be inside salloc, not login) ───
if [ ! -d "${MNE_DENOISE_VENV}" ]; then
    case "$(hostname)" in
        *login*)
            echo "REFUSING to build the venv on a login node." >&2
            echo "Get an allocation first, then re-source:" >&2
            echo "  salloc --account=rrg-kjerbi --time=0:45:00 --cpus-per-task=4 --mem=16G" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac

    echo "--- Creating virtualenv (one-time build) ---"
    # --system-site-packages reuses numpy/scipy/matplotlib/pandas/sklearn from scipy-stack
    python -m venv --system-site-packages "${MNE_DENOISE_VENV}"
    # shellcheck disable=SC1091
    source "${MNE_DENOISE_VENV}/bin/activate"

    python -m pip install --no-index --upgrade pip

    # mne-denoise + its scientific deps (mne, numpy, scipy, sklearn, matplotlib)
    # are all available as Alliance wheels — install offline.  ASR is native
    # (numpy/scipy), so asrpy is NOT required.
    python -m pip install --no-index -e "${MNE_DENOISE_REPO}[test]" \
        || python -m pip install --no-index -e "${MNE_DENOISE_REPO}"

    # Extra deps — all confirmed in the Alliance wheelhouse (2026-06-21), so the
    # whole env builds offline (compute nodes have no internet). openneuro-py is
    # only used on login nodes (downloads); pooch for OSF/Zenodo fetchers.
    python -m pip install --no-index pandas seaborn pytest openneuro-py pooch
else
    echo "--- Activating existing virtualenv ---"
    # shellcheck disable=SC1091
    source "${MNE_DENOISE_VENV}/bin/activate"
fi

# ── Thread tuning — respect the cores Slurm actually gave us ───────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ── Keep caches off $HOME (quota) ─────────────────────────────────────────────
export XDG_CACHE_HOME="${SCRATCH:-${HOME}/scratch}/.cache"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${DATA_DIR}" 2>/dev/null

# ── Sanity line (non-fatal) ───────────────────────────────────────────────────
python -c "import mne_denoise, mne; print(f'  mne-denoise {mne_denoise.__version__} | mne {mne.__version__} | python {__import__(\"platform\").python_version()}')" \
    || echo "WARNING: 'import mne_denoise' failed — check the build above."
echo "=== environment ready ==="
