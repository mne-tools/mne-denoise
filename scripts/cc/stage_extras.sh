#!/bin/bash
# Stage the non-OpenNeuro datasets on a Fir LOGIN node (needs internet):
#   ERP CORE  -> OSF (osfclient: osf -p thsqg clone)
#   EEGdenoiseNet -> G-Node/GIN (git clone + git annex get)
# Run detached:
#   cd /scratch/$USER/mne-denoise-bench && mkdir -p logs
#   nohup bash scripts/cc/stage_extras.sh erp_core_n170 eegdenoisenet \
#         > logs/stage_extras.log 2>&1 &
set -o pipefail
export DATASETS_ROOT="${DATASETS_ROOT:-/project/rrg-kjerbi/datasets}"
cd "$(dirname "$0")/../.." || { echo "cannot find repo root"; exit 1; }
echo "repo: $(pwd)  host: $(hostname)  start: $(date '+%F %T')  DATASETS_ROOT=${DATASETS_ROOT}"
source scripts/cc/fir_env.sh

# Tools required by the OSF + GIN downloaders (login node has internet).
module load git-annex 2>/dev/null && echo "git-annex: $(command -v git-annex)" || echo "WARN: no git-annex module"
command -v osf >/dev/null 2>&1 || pip install --quiet osfclient || echo "WARN: could not install osfclient"

rc_all=0
for ds in "$@"; do
    echo "============================================================"
    echo "=== $(date '+%F %T') staging ${ds} ==="
    python scripts/cc/stage_dataset.py "${ds}"
    rc=$?
    echo "=== $(date '+%F %T') ${ds} rc=${rc} ==="
    [ "${rc}" -ne 0 ] && rc_all=1
done
echo "EXTRAS_DONE rc=${rc_all} $(date '+%F %T')"
exit "${rc_all}"
