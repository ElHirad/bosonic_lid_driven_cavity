# Bosonic mean-field lid-driven cavity

A **32×32, Re=100** steady lid-driven cavity solver using the **streamfunction–vorticity formulation** and a product of explicit local bosonic Fock states. Both unknown fields are obtained by mean-field operator evolution. The production package uses no DNS calculation, direct Poisson inverse, pressure projection, MAC grid, MPS, or TDVP.

The unit-square cavity has a right-moving lid of speed 1 and three stationary, no-slip walls. Here **32×32 means 32 points per direction including the walls**, with spacing 1/31. There are 30×30 independent sites for each field, giving **1,800 local Fock vectors**, each of dimension 13 at the default occupation cutoff 12. Boundary values are eliminated analytically into the operator coefficients.

This solver computes a **steady solution**. Its iteration coordinate is artificial time, not the physical transient after the lid starts moving. It is a classical simulation of bosonic mean-field states; no quantum-hardware speedup is claimed.

## Validated results

See **[the numerical report](results/REPORT.md)** for the saved-state verification, independent DNS comparison, cutoff/timestep/scale/relaxation checks, and published Re=100 benchmarks. The complete result files, local Fock states, parameters, histories, plots, and source hashes are included in [results](results/).

![Mean-field and DNS centerlines](results/centerlines.png)

![Cavity fields and DNS error](results/cavity_fields.png)

The mean-field result is checked against an independently written DNS on the same grid, and DNS on nested 63×63 and 125×125 grids estimates spatial error. Centerline velocities and the primary streamfunction minimum are also checked against [Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4). Agreement with the matched DNS verifies the bosonic implementation; the 32×32 solution still has finite spatial error.

## Run the mean-field solver

Requires Python 3.9 or newer. The recorded calculations used Python 3.9.21, NumPy 1.23.5, and SciPy 1.11.2.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python3 -m unittest discover -s tests -v
python3 -m ldc_mean_field.solver --output new_results/mf32.npz
python3 -m validation.verify new_results/mf32.npz
```

The first calculation command runs only the mean-field solver. The second independently checks its saved kets and physical residuals; it does not evolve a DNS trajectory. Existing result files are never overwritten. A run that reaches its iteration limit without converging saves a partial NPZ with `converged=false` and exits unsuccessfully.

Default settings are occupation cutoff 12, RK4 artificial step 0.0025, streamfunction relaxation coefficient 0.1, and absolute infinity-norm tolerance 1e-7 for **each** steady equation. Observables are ψ=100⟨aψ⟩ and ω=10000⟨aω⟩. These encoding scales improve numerical conditioning without changing the physical equations. Their sensitivity is checked by halving both scales. Small-scale exploratory runs failed the coherent-state quality gate and were rejected before comparison.

The maximum coherent-state defect must stay below 1e-5, ceiling probability below 1e-9, and normalization error below 1e-12 at every diagnostic sample (100 steps). The recorded production values are much smaller; the report gives the measured values. Kets are normalized after RK4 steps, and are **never reset to coherent states** during evolution.

## Reproduce the full comparison

After the production run and saved-state verification above:

```bash
python3 -m ldc_mean_field.solver --cutoff 16 --output new_results/mf32_cutoff16.npz
python3 -m ldc_mean_field.solver --dt 0.00125 --output new_results/mf32_half_dt.npz
python3 -m ldc_mean_field.solver --psi-scale 50 --omega-scale 5000 --output new_results/mf32_half_scales.npz
python3 -m ldc_mean_field.solver --relaxation 0.05 --output new_results/mf32_relax005.npz

python3 -m validation.dns --n 32 --output new_results/dns32.npz
python3 -m validation.dns --n 63 --output new_results/dns63.npz
python3 -m validation.dns --n 125 --output new_results/dns125.npz
export MPLCONFIGDIR=/tmp/ldc-matplotlib
python3 -m validation.compare --results new_results
```

To verify the included artifacts without recomputing the solutions:

```bash
python3 -m validation.compare --results results
```

On Pitt CRC, [hpc/run.sbatch](hpc/run.sbatch) runs production on one CPU core and [hpc/validate.sbatch](hpc/validate.sbatch) performs the validation sequence. Submit from this repository root after installing the environment above. The scripts use `.venv/bin/python` (override with `LDC_PYTHON`), target the HTC cluster, and are specific to that environment. Change output paths when recomputing included results.

```bash
sbatch --clusters=htc hpc/run.sbatch --output new_results/mf32.npz
# After the production job completes successfully:
LDC_RESULTS=new_results sbatch --clusters=htc hpc/validate.sbatch
```

## Code and derivation

| Location | Purpose |
|---|---|
| [ldc_mean_field/bosons.py](ldc_mean_field/bosons.py) | Local a, a†, I matrices, normalized Fock-vector RK4 |
| [ldc_mean_field/operators.py](ldc_mean_field/operators.py) | Normally ordered cavity monomials, wall elimination, full mean-field factorization |
| [ldc_mean_field/solver.py](ldc_mean_field/solver.py) | Quiescent initialization, ket evolution, convergence gates, saved states |
| [DERIVATION.md](DERIVATION.md) | Equations, signs, boundary conditions, bosonic reduction, numerical interpretation |
| [validation/dns.py](validation/dns.py) | Separate classical DNS reference, SSPRK3 and DST Poisson inverse |
| [validation/verify.py](validation/verify.py) | Independent observable/PDE checks from saved Fock states |
| [validation/compare.py](validation/compare.py) | DNS and published benchmark comparisons, plots, numerical report |
| [tests/test_numerics.py](tests/test_numerics.py) | Stencils, full factorization, coherent tangent, manufactured solution, dependency separation |

`ldc_mean_field` does not import `validation`, read reference files, or use a field-space inverse. Its SciPy sparse arrays store monomial coefficients only. The reference solver does not import the mean-field package or use its solution as an initial condition. The comparison combines completed outputs in one direction only.

The generic local-boson machinery was adapted from [ElHirad/bosonic_mixing_layer](https://github.com/ElHirad/bosonic_mixing_layer), commit `7e0b97a`. This is a separate repository with a new history and its own origin. The original mixing-layer repository and local uncommitted work were preserved.
