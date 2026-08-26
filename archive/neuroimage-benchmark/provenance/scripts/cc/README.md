# mne-denoise CPU benchmarking on Fir (Compute Canada)

## Submission controller

Full paper runs must pass the configuration and provenance gate. The older generic
registry smoke test remains useful for environment diagnosis, but it is not a paper
benchmark and must never substitute synthetic data when a real dataset is missing.

```bash
python -m mne_denoise.benchmarks validate configs/benchmarks/<arm>.yaml
python scripts/cc/submission_controller.py preflight \
    --config configs/benchmarks/<arm>.yaml
```

Every array element is wrapped by `submission_controller.py run`, which writes an
atomic `terminal_status.json` before computation and updates it on success or failure.
The wrapper requires a dataset manifest, a clean Git commit, and a frozen config.

Scaffolding to run the mne-denoise denoisers (ASR, ZapLine, DSS, ICanClean) on the
**Fir** cluster, **CPU-only**, charged to **`rrg-kjerbi`**. The actual benchmark
metrics/datasets are wired in a follow-up step; what's here makes the branch
*runnable* and proves the merged code (notably the native ASR) works on a CPU node.

| File | Purpose |
|------|---------|
| `fir_env.sh` | Source to load modules + build (first time) and activate `venv_fir`. |
| `requirements_fir.txt` | Extra PyPI deps (openneuro-py, pandas, seaborn). Core stack comes from `scipy-stack` + editable install. |
| `submit_benchmark.sh` | `sbatch` array (1–44 subjects), `rrg-kjerbi`, 4 CPU / 16G / 3h. |
| `run_benchmark.py` | Thin runner: `--smoke`, `--subject`, `--slurm-array`, `--all`. Reuses `scripts/config.py`. |

> Paths/dataset (`ds003620`, 44 subjects) come from `scripts/config.py`, which already
> auto-detects Compute Canada via `$CC_CLUSTER`. ASR here is a **native** numpy/scipy
> implementation — `asrpy` is **not** required.

## Alliance guardrails (do not skip)
- **Never build the env or run Python on a login node.** Module loads + `pip install`
  + imports + the smoke test all run inside an `salloc` allocation. `fir_env.sh`
  refuses to build the venv on a `*login*` host.
- Long runs go through `sbatch`, not `salloc`. Poll with `sacct`/`seff`, **≤ once a minute**.
- CPU work uses **`rrg-kjerbi`** (no `_gpu`/`_cpu` suffix, no `--gres`).

## One-time setup (inside an allocation)
```bash
ssh fir
cd /scratch/$USER && git clone git@github.com:mne-tools/mne-denoise.git 2>/dev/null || true
cd mne-denoise && git fetch origin && git checkout benchmark/compute-canada && git pull --ff-only

salloc --account=rrg-kjerbi --time=0:45:00 --cpus-per-task=4 --mem=16G
srun --pty bash -l
hostname                                   # must be a compute node (fc#####), NOT loginN
source scripts/cc/fir_env.sh               # builds venv_fir on first source (~few min)
python scripts/cc/run_benchmark.py --smoke # synthetic Raw → all denoisers; ASR must pass
exit                                        # release the allocation
```

## Get the data (optional until benchmarks are wired)
```bash
# inside an allocation, env active:
python scripts/download_openneuro.py --dataset ds003620 --task oddball \
    --subjects $(printf "sub-%02d " $(seq 1 44))
# data lands under $DATA_DIR/ds003620 (default: $HOME/scratch/mnedenoise/data)
```

## Submit the array (from a login node — submission is login-safe)
```bash
cd /scratch/$USER/mne-denoise && mkdir -p logs
sbatch --array=1 scripts/cc/submit_benchmark.sh     # smoke one subject first
sbatch scripts/cc/submit_benchmark.sh               # full 1–44%20
```

## Monitor + retrieve
```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem,ExitCode -P | head
seff <jobid>
# results land in $SCRATCH/mne_denoise_results/cc_benchmark/ (see scripts/config.py OUTPUT_DIR)
rsync -av fir:/scratch/$USER/mne_denoise_results/ ./results/
```

## Tuning notes
- Drop `--time`/`--mem` once a real subject's `Elapsed`/`MaxRSS` are known (over-requests hurt scheduling priority).
- `--array=1-44%20` caps concurrency at 20; raise/lower to taste.
- `MNE_DENOISE_REPO` / `MNE_DENOISE_VENV` / `DATA_DIR` override the defaults in `fir_env.sh`.
