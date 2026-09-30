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
# Check plotting dependencies on the allocated node before spending time on DNS.
"$LDC_PYTHON" -c 'from PIL import Image; import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot; import matplotlib.backends.backend_pdf'
if [[ ! -f "$LDC_CASE/dns.npz" ]]; then
    "$LDC_PYTHON" -m validation.run_reference "$LDC_CASE" --max-time 1000
fi
# This independently verifies both saved solutions before reusing any DNS output.
"$LDC_PYTHON" -m validation.case "$LDC_CASE"
