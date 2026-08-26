FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gfortran git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mne-denoise
COPY requirements-neuroimage-lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-neuroimage-lock.txt

COPY . .
RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pytest -q tests/benchmarks/test_sharding.py

ENTRYPOINT ["python", "-m", "mne_denoise.benchmarks"]
CMD ["validate", "configs/benchmarks"]
