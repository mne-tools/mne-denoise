#!/bin/bash
# Sequentially stage OpenNeuro benchmark datasets on a Fir LOGIN node (datasets need
# internet; compute nodes are offline). Downloads are I/O-bound — run detached:
#
#   cd /scratch/$USER/mne-denoise-bench && mkdir -p logs
#   nohup bash scripts/cc/stage_all.sh ds003620 ds004784 ds000117 \
#         > logs/stage_all.log 2>&1 &
#
# Each dataset is downloaded (openneuro-py) to its resolved scratch root, then
# validated, then a sentinel is written (see scripts/cc/stage_dataset.py).
set -o pipefail
# Stage datasets under the shared /project root (firm user preference), overridable.
export DATASETS_ROOT="${DATASETS_ROOT:-/project/rrg-kjerbi/datasets}"
cd "$(dirname "$0")/../.." || { echo "cannot find repo root"; exit 1; }
echo "repo: $(pwd)  host: $(hostname)  start: $(date '+%F %T')  DATASETS_ROOT=${DATASETS_ROOT}"
source scripts/cc/fir_env.sh

rc_all=0
for ds in "$@"; do
    echo "============================================================"
    echo "=== $(date '+%F %T') staging ${ds} ==="
    python scripts/cc/stage_dataset.py "${ds}"
    rc=$?
    echo "=== $(date '+%F %T') ${ds} finished rc=${rc} ==="
    [ "${rc}" -ne 0 ] && rc_all=1
done
echo "ALL_DONE rc=${rc_all} $(date '+%F %T')"
exit "${rc_all}"
