#!/bin/bash
#SBATCH --job-name=ldc128_re1000
#SBATCH --output=results/re1000_n128/%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ldc-matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
LDC_PYTHON="${LDC_PYTHON:-.venv/bin/python}"
LDC_CASE="${LDC_CASE:-results/re1000_n128}"
mkdir -p "$LDC_CASE"
resume_arguments=()
if [[ -f "$LDC_CASE/checkpoint.npz" ]]; then
    resume_arguments=(--resume)
fi
# The Python runner checkpoints actual kets on a stop signal.
request_checkpoint() { kill -USR1 "$solver_pid" 2>/dev/null || true; }
trap request_checkpoint USR1 TERM
"$LDC_PYTHON" -m ldc_mean_field.run_case --directory "$LDC_CASE" \
    --n 128 --reynolds 1000 --cutoff 12 --dt 0.00125 --relaxation 0.01 \
    --psi-scale 1000 --omega-scale 100000 --max-time 400 \
    "${resume_arguments[@]}" &
solver_pid=$!
solver_status=0
while kill -0 "$solver_pid" 2>/dev/null; do
    wait "$solver_pid" && solver_status=0 || solver_status=$?
done
[[ "$solver_status" -eq 0 ]] || exit "$solver_status"
"$LDC_PYTHON" -m validation.verify "$LDC_CASE/mean_field.npz"
