#!/bin/bash
# Fir (Digital Research Alliance / Compute Canada) CPU environment.
# Source this file from an allocation; the first source builds the environment.

_FIR_SELF="${BASH_SOURCE[0]:-$0}"
_FIR_DIR="$(cd "$(dirname "${_FIR_SELF}")" && pwd)"
export MNE_DENOISE_REPO="${MNE_DENOISE_REPO:-$(cd "${_FIR_DIR}/../.." && pwd)}"
_FIR_REVISION="$(git -C "${MNE_DENOISE_REPO}" rev-parse --short=12 HEAD 2>/dev/null || basename "${MNE_DENOISE_REPO}")"
_FIR_VENV_ROOT="${SCRATCH:-${HOME}/scratch}/mne-denoise-venvs"
export MNE_DENOISE_VENV="${MNE_DENOISE_VENV:-${_FIR_VENV_ROOT}/${_FIR_REVISION}-py311}"
_FIR_VENV_READY="${MNE_DENOISE_VENV}/.mne-denoise-ready"
_FIR_VENV_LOCK="${MNE_DENOISE_VENV}.build-lock"
_FIR_VENV_FAILED="${MNE_DENOISE_VENV}.build-failed"

export DATA_DIR="${DATA_DIR:-${HOME}/scratch/mnedenoise/data}"

echo "=== mne-denoise Fir environment ==="
echo "  repo  : ${MNE_DENOISE_REPO}"
echo "  venv  : ${MNE_DENOISE_VENV}"
echo "  data  : ${DATA_DIR}"
echo "  host  : $(hostname)"

module purge 2>/dev/null
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 scipy-stack/2024a 2>/dev/null \
    || module load python/3.11 scipy-stack 2>/dev/null \
    || echo "WARNING: could not load python/3.11 + scipy-stack modules"

if [ ! -f "${_FIR_VENV_READY}" ]; then
    case "$(hostname)" in
        *login*)
            echo "REFUSING to build the venv on a login node." >&2
            echo "Get an allocation first, then re-source:" >&2
            echo "  salloc --account=def-kjerbi_cpu --time=0:45:00 --cpus-per-task=4 --mem=16G" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac

    mkdir -p "${_FIR_VENV_ROOT}"
    if mkdir "${_FIR_VENV_LOCK}" 2>/dev/null; then
        echo "--- Creating virtualenv (one-time build) ---"
        rm -f "${_FIR_VENV_FAILED}"
        _fir_build_status=0
        if [ -e "${MNE_DENOISE_VENV}" ]; then
            _fir_incomplete="${MNE_DENOISE_VENV}.incomplete.$(date -u +%Y%m%dT%H%M%SZ).$$"
            echo "Moving incomplete environment to ${_fir_incomplete}"
            mv "${MNE_DENOISE_VENV}" "${_fir_incomplete}" || _fir_build_status=$?
        fi
        if [ "${_fir_build_status}" -eq 0 ]; then
            python -m venv --system-site-packages "${MNE_DENOISE_VENV}" \
                || _fir_build_status=$?
        fi
        if [ "${_fir_build_status}" -eq 0 ]; then
            # shellcheck disable=SC1091
            source "${MNE_DENOISE_VENV}/bin/activate" || _fir_build_status=$?
        fi
        if [ "${_fir_build_status}" -eq 0 ]; then
            python -m pip install --no-index --upgrade pip || _fir_build_status=$?
        fi
        if [ "${_fir_build_status}" -eq 0 ]; then
            python -m pip install --no-index -e "${MNE_DENOISE_REPO}[test]" \
                || python -m pip install --no-index -e "${MNE_DENOISE_REPO}" \
                || _fir_build_status=$?
        fi
        if [ "${_fir_build_status}" -eq 0 ]; then
            python -m pip install --no-index pandas seaborn pytest pyyaml \
                openneuro-py pooch pymatreader==0.0.32 \
                || _fir_build_status=$?
        fi
        if [ "${_fir_build_status}" -eq 0 ]; then
            # These comparator-only dependencies are not required by ASR jobs.
            python -m pip install --no-index python-picard==0.8.2 amica==0.0.1 \
                || echo "WARNING: Picard/AMICA wheels unavailable; do not submit BSS arrays."
            touch "${_FIR_VENV_READY}"
        else
            printf '%s\n' "${_fir_build_status}" > "${_FIR_VENV_FAILED}"
        fi
        rmdir "${_FIR_VENV_LOCK}" 2>/dev/null || true
        if [ "${_fir_build_status}" -ne 0 ]; then
            echo "Fir environment build failed with status ${_fir_build_status}." >&2
            return "${_fir_build_status}" 2>/dev/null || exit "${_fir_build_status}"
        fi
    else
        echo "--- Waiting for another task to finish the virtualenv build ---"
        _fir_waited=0
        while [ ! -f "${_FIR_VENV_READY}" ] && [ "${_fir_waited}" -lt 900 ]; do
            if [ -f "${_FIR_VENV_FAILED}" ]; then
                echo "Concurrent Fir environment build failed." >&2
                return 1 2>/dev/null || exit 1
            fi
            sleep 5
            _fir_waited=$((_fir_waited + 5))
        done
        if [ ! -f "${_FIR_VENV_READY}" ]; then
            echo "Timed out waiting for the Fir environment build." >&2
            return 1 2>/dev/null || exit 1
        fi
        # shellcheck disable=SC1091
        source "${MNE_DENOISE_VENV}/bin/activate"
    fi
else
    echo "--- Activating existing virtualenv ---"
    # shellcheck disable=SC1091
    source "${MNE_DENOISE_VENV}/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export XDG_CACHE_HOME="${SCRATCH:-${HOME}/scratch}/.cache"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${DATA_DIR}" 2>/dev/null

for _try in 1 2 3 4 5 6; do
    python -c "import mne_denoise, mne_denoise.asr, yaml, mne, numpy, scipy, pymatreader" 2>/dev/null && break
    echo "  import warmup attempt ${_try} failed; backing off..."
    sleep $(( (RANDOM % 8) + 2 ))
done
python -c "import importlib.metadata, mne_denoise, mne, platform; print(f'  mne-denoise {mne_denoise.__version__} | mne {mne.__version__} | pymatreader {importlib.metadata.version(\"pymatreader\")} | python {platform.python_version()}')" \
    || echo "WARNING: 'import mne_denoise' failed after warmup - check the build above."
echo "=== environment ready ==="
