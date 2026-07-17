#!/bin/bash
#SBATCH --job-name=mnedn_factorial
#SBATCH --account=rrg-kjerbi
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Submit a 0-based array after the one-shard preflight succeeds, for example:
#   sbatch --array=0-47%24 --export=ALL,ARM=bss_generic,SHARD_COUNT=48 \
#     scripts/cc/frozen_factorial_array.sbatch
# Merge only after every array element has a terminal Slurm state:
#   python scripts/merge_benchmark_shards.py \
#     "$SCRATCH/mne-denoise-neuroimage-v1/bss_generic/shards" --expected-shards 48
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh

: "${ARM:?export ARM (bss_generic, bss_amica, bss_forward, asr_core, asr_variants, asr_sensitivity, asr_juggler, reference, sensor_forward, temporal, or streaming)}"
: "${SHARD_COUNT:?export SHARD_COUNT matching the array length}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?submit as a zero-based array}"

if [ "${SHARD_INDEX}" -lt 0 ] || [ "${SHARD_INDEX}" -ge "${SHARD_COUNT}" ]; then
    echo "invalid shard ${SHARD_INDEX}/${SHARD_COUNT}" >&2
    exit 2
fi

RESULT_ARM="${ARM}"
if [ "${ARM}" = "asr_sensitivity" ]; then
    : "${SENSITIVITY_INDEX:?export SENSITIVITY_INDEX for this sensitivity array}"
    if ! [[ "${SENSITIVITY_INDEX}" =~ ^[0-9]+$ ]]; then
        echo "SENSITIVITY_INDEX must be a non-negative integer" >&2
        exit 2
    fi
    # Each one-factor-at-a-time sensitivity cell is an independent factorial.
    # Keep its shards isolated so separately submitted arrays cannot overwrite
    # shard-00000 ... shard-N or their merged evidence.
    RESULT_ARM="${ARM}/cell-$(printf '%02d' "${SENSITIVITY_INDEX}")"
fi

ROOT="${SCRATCH}/mne-denoise-neuroimage-v1/${RESULT_ARM}/shards"
OUT="${ROOT}/shard-$(printf '%05d' "${SHARD_INDEX}")"
mkdir -p "${OUT}" logs

COMMON=(--output-root "${OUT}" --shard-index "${SHARD_INDEX}" --shard-count "${SHARD_COUNT}")
case "${ARM}" in
    bss_generic)
        CONFIG=configs/benchmarks/ground_truth_generic.yaml
        METHODS=iterative_dss_tanh,iterative_dss_robust_tanh,iterative_dss_gauss,iterative_dss_skew,iterative_dss_smooth_tanh,iterative_dss_kurtosis,iterative_dss_wiener_mask,iterative_dss_variance_mask,iterative_dss_dct,iterative_dss_quasi_periodic,iterative_dss_spectrogram,pca,fastica,infomax,picard,sobi,jade,oracle
        RUNNER=(python scripts/run_ground_truth_arm.py --config "${CONFIG}" --methods "${METHODS}")
        ;;
    bss_amica)
        CONFIG=configs/benchmarks/ground_truth_generic.yaml
        RUNNER=(python scripts/run_ground_truth_arm.py --config "${CONFIG}" --methods amica)
        ;;
    bss_forward)
        CONFIG=configs/benchmarks/ground_truth_forward.yaml
        RUNNER=(python scripts/run_ground_truth_arm.py --config "${CONFIG}")
        ;;
    asr_core)
        CONFIG=configs/benchmarks/asr_transient_ground_truth.yaml
        RUNNER=(python scripts/run_asr_transient_ground_truth.py --config "${CONFIG}" --campaign core)
        ;;
    asr_variants)
        CONFIG=configs/benchmarks/asr_variant_ground_truth.yaml
        RUNNER=(python scripts/run_asr_transient_ground_truth.py --config "${CONFIG}" --campaign core)
        ;;
    asr_sensitivity)
        CONFIG=configs/benchmarks/asr_transient_ground_truth.yaml
        RUNNER=(python scripts/run_asr_transient_ground_truth.py --config "${CONFIG}" --campaign sensitivity --sensitivity-index "${SENSITIVITY_INDEX}")
        ;;
    asr_juggler)
        CONFIG=configs/benchmarks/asr_juggler_kim2025.yaml
        RUNNER=(python scripts/run_asr_juggler_kim2025.py --config "${CONFIG}")
        ;;
    reference)
        CONFIG=configs/benchmarks/icanclean_reference_ground_truth.yaml
        RUNNER=(python scripts/run_icanclean_reference_ground_truth.py --config "${CONFIG}")
        ;;
    sensor_forward)
        CONFIG=configs/benchmarks/sensor_forward_ground_truth.yaml
        RUNNER=(python scripts/run_sensor_forward_ground_truth.py --config "${CONFIG}")
        ;;
    temporal)
        CONFIG=configs/benchmarks/temporal_ground_truth.yaml
        RUNNER=(python scripts/run_temporal_ground_truth.py --config "${CONFIG}")
        ;;
    streaming)
        CONFIG=configs/benchmarks/streaming_replay.yaml
        RUNNER=(python scripts/run_streaming_replay.py --config "${CONFIG}")
        ;;
    *)
        echo "unknown ARM=${ARM}" >&2
        exit 2
        ;;
esac

python -m mne_denoise.benchmarks validate "${CONFIG}"
git diff --quiet && git diff --cached --quiet || {
    echo "refusing benchmark execution from a dirty worktree" >&2
    exit 3
}
"${RUNNER[@]}" "${COMMON[@]}"
if grep -q 'unavailable_dependency' "${OUT}/raw_metrics.tsv"; then
    echo "required method dependency unavailable in shard ${SHARD_INDEX}" >&2
    exit 4
fi
