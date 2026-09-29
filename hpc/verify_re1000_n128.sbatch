#!/bin/bash
#SBATCH --job-name=ldc128_re1000_dns
#SBATCH --output=results/re1000_n128/%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ldc-matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
LDC_PYTHON="${LDC_PYTHON:-.venv/bin/python}"
LDC_CASE="${LDC_CASE:-results/re1000_n128}"
"$LDC_PYTHON" -m validation.run_reference "$LDC_CASE" --max-time 1000
"$LDC_PYTHON" -m validation.case "$LDC_CASE"
