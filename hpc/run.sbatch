#!/bin/bash
#SBATCH --job-name=bosonic_ldc
#SBATCH --output=results/%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ldc-matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
LDC_PYTHON="${LDC_PYTHON:-.venv/bin/python}"
"$LDC_PYTHON" -m ldc_mean_field.solver "$@"
