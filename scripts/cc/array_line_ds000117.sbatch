#!/bin/bash
#SBATCH --job-name=line_ds000117
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1-16%8
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# MEG line arm (ds000117 faces): one task per subject (sub-01..sub-16). Uses the RAW
# run-01 FIF (line noise intact; SSS strips it), magnetometers only. ZapLine / ZapLine+
# / notch / non-spatial on identical held-out data; scores 50 Hz attenuation. Heavy
# (adaptive ZapLine+ segments a ~8 min, 1100 Hz, 102-ch recording). Aggregate after:
#   python scripts/run_line_arm.py --config configs/benchmarks/line_noise_ds000117.yaml \
#       --group-only --deriv-root $SCRATCH/mne-denoise/results/line_ds000117
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/cc/fir_env.sh
export DATASETS_ROOT=/project/rrg-kjerbi/datasets
SUBJ=$(printf "sub-%02d" "${SLURM_ARRAY_TASK_ID}")
echo "task ${SLURM_ARRAY_TASK_ID} -> ${SUBJ}"
python scripts/run_line_arm.py \
  --config configs/benchmarks/line_noise_ds000117.yaml \
  --subject "${SUBJ}" \
  --deriv-root "${SCRATCH}/mne-denoise/results/line_ds000117"
