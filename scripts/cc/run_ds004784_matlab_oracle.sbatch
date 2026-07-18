#!/bin/bash
#SBATCH --job-name=asr_matlab
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load matlab/2023b.2

: "${ASR_PHANTOM_DATA_DIR:?export the verified ds004784 root}"
: "${ASR_MATLAB_ORACLE_ROOT:?export the EEGLAB and clean_rawdata parent}"
: "${ASR_MATLAB_CONDITION:?export the condition name}"
: "${ASR_MATLAB_USE_EXTERNAL:?export 0 or 1}"
: "${ASR_MATLAB_CUTOFF:?export the burst cutoff}"
: "${ASR_MATLAB_OUTPUT:?export the .mat output path}"

mkdir -p "$(dirname "${ASR_MATLAB_OUTPUT}")" logs
matlab -batch "addpath('scripts/matlab'); run_ds004784_asr_oracle(string(getenv('ASR_PHANTOM_DATA_DIR')), string(getenv('ASR_MATLAB_ORACLE_ROOT')), string(getenv('ASR_MATLAB_CONDITION')), logical(str2double(getenv('ASR_MATLAB_USE_EXTERNAL'))), str2double(getenv('ASR_MATLAB_CUTOFF')), string(getenv('ASR_MATLAB_OUTPUT')));"
