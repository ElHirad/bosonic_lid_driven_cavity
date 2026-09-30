# Launch: 128×128, Re=1000

Submitted on **2026-09-29**. This document records the launch configuration, not a completed result.

**Completion update, 2026-09-30:** the mean-field job completed successfully in 3 h 27 min
52 s, converging after 186,900 iterations. DNS also converged in 559.45 s. Its original
verification job was marked failed because the compute node lacked the Pillow plotting
dependency, after saving the complete DNS solution. The environment was repaired and the
saved solutions passed the comparison checks; neither fluid calculation needed rerunning.
Postprocessing recovery job **11486253** completed successfully on a compute node in 13 s,
regenerating the report and plots from those saved solutions.
See the [completed report](REPORT.md) for errors and plots.

| Setting | Value |
|---|---|
| Grid | 128×128 including walls; h=1/127 |
| Reynolds number | 1000 (ν=0.001, U=L=1) |
| Formulation | Streamfunction–vorticity, steady artificial-time relaxation |
| Quantum state | Product of 31,752 local Fock vectors |
| Boson occupation cutoff | N_b=12, occupations 0,…,12; 13 coefficients per site |
| Initial condition | Vacuum interior kets, prescribed unit moving lid |
| RK4 artificial timestep | 0.00125 |
| Streamfunction relaxation rate | 0.01 |
| Observable scales | ψ=1000⟨aψ⟩, ω=100000⟨aω⟩ |
| Steady tolerance | Both physical residual infinity norms ≤1e-7 |
| Quantum quality gates | Coherent defect <1e-5; ceiling probability <1e-9; norm error <1e-12 |
| Diagnostic interval | 100 steps |
| Checkpoint interval | 1000 steps, plus stop signals |
| Maximum artificial time | 400; an unconverged run exits unsuccessfully and retains its checkpoint |
| Production Slurm job | **11480040**, HTC, 1 CPU, 4 GB, 8-hour allocation |
| Independent DNS / plotting job | **11480049**, starts only after successful job 11480040 |

The pilot (job 11480028) advanced 1,000 steps to artificial time 1.25 in 37.59 seconds.
It passed all sampled quantum-quality gates: worst coherent-state defect 2.2124e-13,
ceiling probability 4.7977e-81, and norm error 2.2204e-16. The pilot deliberately stopped
before steady convergence and is not a validated solution. Production timing depends
on the allocated node and the number of iterations required; the first production
1,000 steps took about 85 seconds. Allow several hours for this calculation.

The larger grid requires smaller integration steps than the original 32×32 case.
The changed relaxation rate and observable scales affect artificial-time conditioning,
not Re, viscosity, lid speed, or the target steady equations. The core normally ordered
operator assembly and Fock-vector RK4 implementations are unchanged. No DNS field is
used for initialization, relaxation, correction, or restart.

## Run and resume

From the repository root, with the Python environment installed:

```bash
mkdir -p results/re1000_n128
sbatch --clusters=htc hpc/re1000_n128.sbatch
# Substitute the new production job ID when reproducing:
sbatch --clusters=htc --dependency=afterok:PRODUCTION_JOB_ID hpc/verify_re1000_n128.sbatch
```

The production batch script automatically adds `--resume` when `checkpoint.npz` exists.
Resume refuses changed configuration or production-source fingerprints. File locking
prevents two writers from using the same case directory. The resume test checks exact
equality of saved kets against an uninterrupted trajectory.

For a fresh calculation, set `LDC_CASE` to a new directory for both batch submissions.
The scripts use `.venv/bin/python`; set `LDC_PYTHON` to override it. A completed
`mean_field.npz` is never overwritten.

## Outputs and validation

- `status.json`: local live status, latest sampled residuals, quantum quality, and elapsed time.
- `checkpoint.npz`: local actual Fock states and history for restarting; excluded from Git.
- `mean_field.npz`: written only when the mean-field convergence criteria pass.
- `mean_field.verification.json`: independent checks from saved kets and cavity stencils.
- `dns.npz`: independent rest-start DNS at the same n and Re, using SSPRK3 and its own DST Poisson inverse.
- `comparison.json`, `REPORT.md`, and figures: generated only after both results pass verification.

The comparison will report absolute discrete L2, mesh-weighted spatial L2, and relative
L2 errors for ψ, ω, u, v, and the velocity vector. It will also show both methods'
streamfunction/vorticity fields and compare centerlines against the **Re=1000** values
in Ghia, Ghia & Shin (1982), Tables I–II. DNS stays entirely within `validation/`.

The original Re=100 validation files remain in the parent results directory. This launch
does not claim that their cutoff/timestep/refinement studies establish convergence of
the new Re=1000 case; its own measured residuals and DNS comparison must pass first.
