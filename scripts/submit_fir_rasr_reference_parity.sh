#!/bin/bash
# Generate rASRMatlab reference fixtures and run the legacy rASR parity gates.
set -euo pipefail

package_commit="${PACKAGE_COMMIT:-229232675e1286681c147822d5170a3d69e29c28}"
rasr_commit="${RASR_COMMIT:-0e1ead42028bf18d023e9ace96e234ca12afd437}"
clean_rawdata_commit="${CLEAN_RAWDATA_COMMIT:-d4b143f2a7719cf12d46c9b3e15aa827edb05614}"
source_repo="${SOURCE_REPO:-/scratch/sesma/mne-denoise-neuroimage-asr-sensitivity-2292326}"
work_repo="${WORK_REPO:-/scratch/sesma/mne-denoise-asr-rasr-parity-2292326}"
venv="${MNE_DENOISE_VENV:-/scratch/sesma/venvs/mne-denoise-asr}"
account="${SLURM_ACCOUNT:-def-kjerbi_cpu}"
matlab_module="${MATLAB_MODULE:-matlab/2025b.1}"

if [[ ! -d "${work_repo}/.git" ]]; then
    git clone --quiet "${source_repo}" "${work_repo}"
fi
git -C "${work_repo}" checkout --quiet "${package_commit}"

mkdir -p "${work_repo}/refs/asr/repos" "${work_repo}/slurm"
if [[ ! -d "${work_repo}/refs/asr/repos/rASRMatlab/.git" ]]; then
    git clone --quiet https://github.com/s4rify/rASRMatlab.git \
        "${work_repo}/refs/asr/repos/rASRMatlab"
fi
git -C "${work_repo}/refs/asr/repos/rASRMatlab" fetch --quiet origin
git -C "${work_repo}/refs/asr/repos/rASRMatlab" checkout --quiet "${rasr_commit}"

# rASRMatlab relies on Manopt but does not vendor it. Use the dependency
# revision pinned by the audited clean_rawdata oracle.
if [[ ! -d "${work_repo}/refs/asr/repos/clean_rawdata/.git" ]]; then
    git clone --quiet https://github.com/sccn/clean_rawdata.git \
        "${work_repo}/refs/asr/repos/clean_rawdata"
fi
git -C "${work_repo}/refs/asr/repos/clean_rawdata" fetch --quiet origin
git -C "${work_repo}/refs/asr/repos/clean_rawdata" checkout --quiet \
    "${clean_rawdata_commit}"
git -C "${work_repo}/refs/asr/repos/clean_rawdata" submodule update \
    --init --recursive

job_id=$(sbatch --parsable \
    --account="${account}" \
    --time=00:45:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --output="${work_repo}/slurm/rasr-parity-%j.out" \
    --error="${work_repo}/slurm/rasr-parity-%j.err" \
    --wrap="set -euo pipefail; cd '${work_repo}'; '${venv}/bin/python' tests/parity/matlab_reference/generate_asr_input.py; module load '${matlab_module}'; matlab -batch \"addpath(genpath('${work_repo}/refs/asr/repos/clean_rawdata/manopt')); addpath(genpath('${work_repo}/refs/asr/repos/rASRMatlab/rASRToolbox')); run('${work_repo}/tests/parity/matlab_reference/generate_rasr_reference.m')\"; '${venv}/bin/python' -m pytest tests/parity/test_asr_parity.py -q -k 'rasr_calibration or rasr_process'")
job_id="${job_id%%;*}"

printf 'job=%s\n' "${job_id}"
printf 'package_repo=%s\n' "${work_repo}"
printf 'package_commit=%s\n' "$(git -C "${work_repo}" rev-parse HEAD)"
printf 'rasr_commit=%s\n' \
    "$(git -C "${work_repo}/refs/asr/repos/rASRMatlab" rev-parse HEAD)"
printf 'manopt_commit=%s\n' \
    "$(git -C "${work_repo}/refs/asr/repos/clean_rawdata/manopt" rev-parse HEAD)"
