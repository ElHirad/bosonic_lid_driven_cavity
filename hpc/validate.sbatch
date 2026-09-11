#!/bin/bash
#SBATCH --job-name=ldc_validation
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
LDC_RESULTS="${LDC_RESULTS:-results}"
"$LDC_PYTHON" -m validation.verify "$LDC_RESULTS/mf32.npz"
"$LDC_PYTHON" -m ldc_mean_field.solver --cutoff 16 --output "$LDC_RESULTS/mf32_cutoff16.npz"
"$LDC_PYTHON" -m ldc_mean_field.solver --dt 0.00125 --output "$LDC_RESULTS/mf32_half_dt.npz"
"$LDC_PYTHON" -m ldc_mean_field.solver --psi-scale 50 --omega-scale 5000 --output "$LDC_RESULTS/mf32_half_scales.npz"
"$LDC_PYTHON" -m ldc_mean_field.solver --relaxation 0.05 --output "$LDC_RESULTS/mf32_relax005.npz"
for n in 32 63 125; do
    "$LDC_PYTHON" -m validation.dns --n "$n" --output "$LDC_RESULTS/dns${n}.npz"
done
"$LDC_PYTHON" -m validation.compare --results "$LDC_RESULTS"
