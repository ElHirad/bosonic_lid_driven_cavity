"""Report and plot a verified matched-grid cavity case at Re=100 or Re=1000."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .compare import read, difference, ghia_values, benchmark_error, plots
from .dns import laplacian, transport, velocity, walls
from .verify import verify


def compare_case(directory):
    directory = Path(directory)
    validation = verify(directory/"mean_field.npz")
    mf, dns = read(directory/"mean_field.npz"), read(directory/"dns.npz")
    meta = json.loads(str(mf["metadata_json"]))
    config = meta["config"]
    n, reynolds, cutoff = config["n"], config["reynolds"], config["cutoff"]
    reference = json.loads(str(dns["metadata_json"]))
    if not reference["converged"] or reference["n"] != n or reference["reynolds"] != reynolds:
        raise ValueError("DNS and mean field must be converged and match n and Re")
    if reference["source_sha256"] != hashlib.sha256(Path(__file__).with_name("dns.py").read_bytes()).hexdigest():
        raise ValueError("DNS source fingerprint differs")
    h = 1/(n-1)
    for key in ("psi", "omega", "u", "v"):
        if dns[key].shape != (n, n) or not np.all(np.isfinite(dns[key])):
            raise ValueError("invalid DNS fields")
    for key in ("x", "y"):
        np.testing.assert_array_equal(dns[key], mf[key])
    omega = walls(dns["psi"], dns["omega"][1:-1, 1:-1], h)
    u, v = velocity(dns["psi"], h)
    for key, value in dict(omega=omega, u=u, v=v).items():
        np.testing.assert_allclose(dns[key], value, atol=1e-12, rtol=0)
    poisson_residual = float(np.max(np.abs(laplacian(dns["psi"], h)+omega[1:-1, 1:-1])))
    vorticity_residual = float(np.max(np.abs(transport(dns["psi"], omega, h, reynolds))))
    if poisson_residual > 1e-9 or vorticity_residual > reference["tolerance"]*1.001:
        raise ValueError("DNS residual verification failed")
    errors = difference(mf, dns)
    if max(errors[key]["relative_l2"] for key in ("psi", "omega", "velocity")) >= 1e-5:
        raise ValueError("mean-field/DNS relative L2 error exceeds 1e-5")
    benchmark = ghia_values(reynolds)
    reference_min = {100: -0.103423, 1000: -0.117929}[reynolds]
    benchmarks = {"mean_field": benchmark_error(mf, benchmark, reference_min),
                  "dns": benchmark_error(dns, benchmark, reference_min)}
    # Same absolute centerline criterion as the 32x32 benchmark comparison.
    if max(benchmarks["mean_field"][component]["maximum_absolute"] for component in ("u", "v")) >= 0.02:
        raise ValueError("published centerline comparison exceeds 0.02 lid speed")
    metrics = dict(passed=True, n=n, reynolds=reynolds, boson_cutoff=cutoff,
                   local_fock_dimension=cutoff+1, verification=validation,
                   dns_verification=dict(poisson_residual=poisson_residual, vorticity_residual=vorticity_residual),
                   **{f"mean_field_vs_dns{n}": errors}, benchmark=benchmarks,
                   error_norm_definition=dict(domain="interior nodes; both components for velocity",
                                              absolute_l2="sqrt(sum(abs(error)**2))",
                                              spatial_l2="h*absolute_l2", relative_l2="absolute_l2/norm(reference)"),
                   artifact_sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                    for name in ("mean_field.npz", "dns.npz")},
                   postprocess_source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                              for name in ("case.py", "compare.py", "verify.py")})
    plots(directory, mf, {n: dns}, benchmark, metrics)
    (directory/"comparison.json").write_text(json.dumps(metrics, indent=2)+"\n")
    lines = [f"# {n}×{n} lid-driven cavity, Re={reynolds:g}", "",
             f"The grid includes walls, h=1/{n-1}. The production state has {2*(n-2)**2:,} local "
             f"Fock vectors with occupation cutoff **N_b={cutoff}** and **{cutoff+1} coefficients per site**. "
             "The mean-field solver evolves streamfunction and vorticity kets from vacuum. "
             "DNS is a separate verification calculation, starts from rest, and supplies no solution data to the mean-field run.", "",
             "## Parameters and convergence", "",
             f"- Re={reynolds:g}; unit square; unit right-moving lid; stationary no-slip side and bottom walls.",
             f"- Artificial RK4 step: {config['dt']}; streamfunction relaxation rate: {config['relaxation']}.",
             f"- Observable scales: ψ={config['psi_scale']:g}⟨aψ⟩, ω={config['omega_scale']:g}⟨aω⟩.",
             f"- Iterations: {meta['steps']:,}; final artificial time: {meta['final']['pseudo_time']:g}.",
             f"- Mean-field residuals: streamfunction {validation['poisson_residual']:.6e}, "
             f"vorticity {validation['vorticity_residual']:.6e}; tolerance {config['tolerance']:.1e}.",
             f"- Maximum discrete divergence: {validation['divergence_max']:.6e}.",
             f"- Worst sampled coherent-state defect: {meta['worst_sampled_quality']['coherent_defect']:.6e}; "
             f"ceiling probability: {meta['worst_sampled_quality']['ceiling_probability']:.6e}.",
             f"- Measured runtime on one CPU: mean field {meta['elapsed_seconds']/3600:.3f} hours; DNS "
             f"{reference['elapsed_seconds']/3600:.3f} hours.", "",
             "The iteration coordinate is artificial time, not physical startup time. "
             "The mean-field calculation advances streamfunction and vorticity kets together with coupled RK4; "
             "it does not fully relax streamfunction after each individual vorticity update. "
             "Only residual-converged states are reported. The DNS integrates physical time with SSPRK3 and "
             "its own discrete sine-transform Poisson inverse.", "",
             "## L2 error against independent DNS", "",
             f"The sum covers the {(n-2)}×{(n-2)} interior nodes. Absolute discrete L2=√Σ|e|²; "
             f"spatial L2=h√Σ|e|² with h=1/{n-1}; relative L2=||e||₂/||DNS||₂. "
             "The vector norm includes both velocity components.", "",
             "| Field | Absolute discrete L2 | Spatial L2 | Relative L2 | Max absolute error |",
             "|---|---:|---:|---:|---:|"]
    for key, label in (("psi", "Streamfunction ψ"), ("omega", "Vorticity ω"),
                       ("u", "u"), ("v", "v"), ("velocity", "Velocity vector")):
        e = errors[key]
        lines.append(f"| {label} | {e['absolute_l2']:.6e} | {e['spatial_l2']:.6e} | "
                     f"{e['relative_l2']:.6e} | {e['maximum_absolute']:.6e} |")
    lines += ["", "## Published benchmark", "",
              "Comparison against [Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4), "
              "Tables I–II (centerline velocities) and Table V (primary streamfunction minimum). "
              "Centerlines and printed sample coordinates are linearly interpolated.", "",
              "| Method | Max u error / lid speed | Max v error / lid speed | ψ minimum |",
              "|---|---:|---:|---:|"]
    for name, b in benchmarks.items():
        lines.append(f"| {name} | {b['u']['maximum_absolute']:.6e} | {b['v']['maximum_absolute']:.6e} | {b['psi_min']:.9f} |")
    lines += ["", "This matched-grid comparison checks the bosonic calculation at this grid and Re. "
              "It does not by itself establish spatial convergence or cutoff/timestep convergence for this new case. "
              "The separate Re=100 refinement and sensitivity runs remain in the parent results directory.", "",
              "![Mean-field and DNS fields](cavity_fields.png)", "",
              "![Centerlines](centerlines.png)", "", "![Convergence](convergence.png)", "",
              "[Field figure as PDF](cavity_fields.pdf). MF and DNS use shared field color scales; "
              "difference panels have independent error scales. Walls are plotted but excluded from norms.", "",
              "## Reproduction", "", "See `hpc/re1000_n128.sbatch` and `hpc/verify_re1000_n128.sbatch`. "
              "To regenerate this report from completed saved states:", "", "```bash",
              f"python3 -m validation.case {directory.as_posix()}", "```", ""]
    (directory/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    compare_case(parser.parse_args().directory)
