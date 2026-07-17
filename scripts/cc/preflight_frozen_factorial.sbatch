#!/bin/bash
#SBATCH --job-name=mnedn_preflight
#SBATCH --account=rrg-kjerbi
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Read-only scientific preflight. Submit one job before any corresponding array:
#   sbatch --export=ALL,ARM=asr_core scripts/cc/preflight_frozen_factorial.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
: "${ARM:?export ARM as accepted by frozen_factorial_array.sbatch}"

export SHARD_COUNT=1
export SLURM_ARRAY_TASK_ID=0
export SCRATCH="${SCRATCH:?SCRATCH is required}"

# Reuse the exact array dispatch with a smoke flag injected through the runner
# wrapper only for campaign names with a direct smoke mode.
case "${ARM}" in
    bss_generic|bss_amica|bss_forward)
        CONFIG=configs/benchmarks/ground_truth_generic.yaml
        [ "${ARM}" = bss_forward ] && CONFIG=configs/benchmarks/ground_truth_forward.yaml
        METHODS=pca,fastica,iterative_dss_tanh,oracle
        [ "${ARM}" = bss_amica ] && METHODS=amica
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_ground_truth_arm.py --config "${CONFIG}" --methods "${METHODS}" \
            --smoke --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    asr_core)
        CONFIG=configs/benchmarks/asr_transient_ground_truth.yaml
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_asr_transient_ground_truth.py --config "${CONFIG}" --smoke \
            --campaign core --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    asr_variants)
        CONFIG=configs/benchmarks/asr_variant_ground_truth.yaml
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_asr_transient_ground_truth.py --config "${CONFIG}" --smoke \
            --campaign core --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    asr_sensitivity)
        CONFIG=configs/benchmarks/asr_transient_ground_truth.yaml
        : "${SENSITIVITY_INDEX:?export SENSITIVITY_INDEX for the sensitivity preflight}"
        if ! [[ "${SENSITIVITY_INDEX}" =~ ^[0-9]+$ ]]; then
            echo "SENSITIVITY_INDEX must be a non-negative integer" >&2
            exit 2
        fi
        CELL="cell-$(printf '%02d' "${SENSITIVITY_INDEX}")"
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_asr_transient_ground_truth.py --config "${CONFIG}" --smoke \
            --campaign sensitivity --sensitivity-index "${SENSITIVITY_INDEX}" \
            --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}/${CELL}"
        ;;
    asr_juggler)
        CONFIG=configs/benchmarks/asr_juggler_kim2025.yaml
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_asr_juggler_kim2025.py --config "${CONFIG}" --smoke \
            --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    reference)
        CONFIG=configs/benchmarks/icanclean_reference_ground_truth.yaml
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_icanclean_reference_ground_truth.py --config "${CONFIG}" --smoke \
            --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    sensor_forward)
        CONFIG=configs/benchmarks/sensor_forward_ground_truth.yaml
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_sensor_forward_ground_truth.py --config "${CONFIG}" --smoke \
            --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    temporal)
        CONFIG=configs/benchmarks/temporal_ground_truth.yaml
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_temporal_ground_truth.py --config "${CONFIG}" --smoke \
            --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    streaming)
        CONFIG=configs/benchmarks/streaming_replay.yaml
        python -m mne_denoise.benchmarks validate "${CONFIG}"
        python scripts/run_streaming_replay.py --config "${CONFIG}" --smoke \
            --output-root "${SCRATCH}/mne-denoise-neuroimage-v1/preflight/${ARM}"
        ;;
    *) echo "unknown ARM=${ARM}" >&2; exit 2 ;;
esac
