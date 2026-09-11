"""Verify completed runs, compare against DNS/Ghia, and write plots/report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .verify import verify
from .dns import laplacian, transport, velocity, walls


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def difference(candidate, reference):
    fields = {}
    for name in ("psi", "omega", "u", "v"):
        a, b = candidate[name][1:-1, 1:-1], reference[name][1:-1, 1:-1]
        fields[name] = dict(relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                            maximum_absolute=float(np.max(np.abs(a-b))))
    a = np.stack([candidate[k][1:-1, 1:-1] for k in ("u", "v")])
    b = np.stack([reference[k][1:-1, 1:-1] for k in ("u", "v")])
    fields["velocity"] = dict(relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                              maximum_absolute=float(np.max(np.abs(a-b))))
    return fields


def centerline(data, component):
    if component == "u":
        return np.array([np.interp(0.5, data["x"], row) for row in data["u"]])
    return np.array([np.interp(0.5, data["y"], column) for column in data["v"].T])


def ghia_values():
    path = Path(__file__).with_name("ghia_re100.csv")
    with path.open() as file:
        rows = list(csv.DictReader(line for line in file if not line.startswith("#")))
    return {c: np.array([(float(r["coordinate"]), float(r["velocity"]))
                          for r in rows if r["component"] == c]) for c in ("u", "v")}


def benchmark_error(data, benchmark):
    result = {}
    for component in ("u", "v"):
        points, values = benchmark[component].T
        numerical = np.interp(points, data["y" if component == "u" else "x"], centerline(data, component))
        result[component] = dict(maximum_absolute=float(np.max(np.abs(numerical-values))),
                                 rms=float(np.sqrt(np.mean((numerical-values)**2))))
    result["psi_min"] = float(data["psi"].min())
    result["psi_min_relative_error"] = float(abs(data["psi"].min()+0.103423)/0.103423)
    return result


def plots(directory, mf, dns, benchmark, comparisons):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for axis, component in zip(axes, ("u", "v")):
        for data, label, style in ((mf, "Bosonic mean field, 32×32", "-"),
                                   (dns[32], "Independent DNS, 32×32", "--"),
                                   (dns[63], "DNS, 63×63", ":"),
                                   (dns[125], "DNS, 125×125", "-.")):
            axis.plot(data["x"], centerline(data, component), style, label=label, lw=1.7)
        points, values = benchmark[component].T
        axis.scatter(points, values, facecolors="none", edgecolors="black", s=30,
                     label="Ghia et al. (1982), Re=100", zorder=5)
        axis.set(xlabel="y" if component == "u" else "x", ylabel=f"{component} / lid speed",
                 title=f"{'Vertical' if component == 'u' else 'Horizontal'} centerline", xlim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(directory/"centerlines.png")
    fig.savefig(directory/"centerlines.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    x, y = mf["x"], mf["y"]
    negative = np.linspace(mf["psi"].min()*0.98, -0.001, 15)
    axes[0].contour(x, y, mf["psi"], levels=negative, colors="#264653", linewidths=0.9)
    axes[0].streamplot(x, y, mf["u"], mf["v"], density=0.6, color="#2a9d8f", linewidth=0.5)
    axes[0].set_title("Mean-field streamfunction and velocity")
    limit = np.max(np.abs(mf["omega"]))
    color = axes[1].pcolormesh(x, y, mf["omega"], cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
    axes[1].contour(x, y, mf["omega"], levels=[-10, -5, -3, -1, 0, 1, 3, 5, 10], colors="k", linewidths=0.4)
    fig.colorbar(color, ax=axes[1], label="Vorticity")
    axes[1].set_title("Mean-field vorticity (full color range)")
    error = np.hypot(mf["u"]-dns[32]["u"], mf["v"]-dns[32]["v"])
    color = axes[2].pcolormesh(x, y, error, cmap="magma", shading="auto")
    fig.colorbar(color, ax=axes[2], label="Velocity difference / lid speed")
    axes[2].set_title("Mean field − independent DNS, 32×32")
    for axis in axes:
        axis.set(xlabel="x", ylabel="y", aspect="equal", xlim=(0, 1), ylim=(0, 1))
    fig.savefig(directory/"cavity_fields.png")
    fig.savefig(directory/"cavity_fields.pdf")
    plt.close(fig)

    history = json.loads(str(mf["history_json"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    tau = [r["pseudo_time"] for r in history]
    for key, label in (("poisson_residual", "Streamfunction equation"),
                       ("vorticity_residual", "Vorticity equation")):
        axes[0].semilogy(tau, [r[key] if r[key] > 0 else np.nan for r in history], label=label)
    axes[0].axhline(1e-7, color="k", linestyle=":", label="Acceptance tolerance")
    axes[0].set(xlabel="Artificial iteration time τ", ylabel="Maximum absolute residual",
                title="Steady-state convergence")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(tau, [r["coherent_defect"] if r["coherent_defect"] > 0 else np.nan for r in history])
    axes[1].axhline(1e-5, color="k", linestyle=":", label="Acceptance limit")
    axes[1].set_ylim(1e-14, 1e-4)
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Artificial iteration time τ", ylabel="Maximum coherent-state defect",
                title="Fock-state accuracy (sampled every 100 steps)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(directory/"convergence.png")
    plt.close(fig)


def compare(directory):
    directory = Path(directory)
    names = ("mf32", "mf32_cutoff16", "mf32_half_dt", "mf32_half_scales", "mf32_relax005")
    verified = {name: verify(directory/(name+".npz")) for name in names}
    runs = {name: read(directory/(name+".npz")) for name in names}
    mf = runs["mf32"]
    base_config = json.loads(str(mf["metadata_json"]))["config"]
    assert base_config["n"] == 32 and base_config["reynolds"] == 100.
    dns = {n: read(directory/f"dns{n}.npz") for n in (32, 63, 125)}
    for n, data in dns.items():
        meta = json.loads(str(data["metadata_json"]))
        assert meta["converged"] and meta["n"] == n and meta["reynolds"] == 100.
        assert meta["vorticity_residual"] < 1e-8 and meta["poisson_residual"] < 1e-9
        assert all(np.all(np.isfinite(data[k])) for k in ("psi", "omega", "u", "v"))
        h = 1/(n-1)
        omega = walls(data["psi"], data["omega"][1:-1, 1:-1], h)
        np.testing.assert_allclose(data["omega"], omega, atol=1e-12, rtol=0)
        u, v = velocity(data["psi"], h)
        np.testing.assert_allclose(data["u"], u, atol=1e-12, rtol=0)
        np.testing.assert_allclose(data["v"], v, atol=1e-12, rtol=0)
        assert np.max(np.abs(laplacian(data["psi"], h)+omega[1:-1, 1:-1])) < 1e-9
        assert np.max(np.abs(transport(data["psi"], omega, h, 100.))) < 1.001e-8
        assert meta["source_sha256"] == hashlib.sha256(Path(__file__).with_name("dns.py").read_bytes()).hexdigest()
    matched = difference(mf, dns[32])
    sensitivity = {name: difference(runs[name], mf) for name in names[1:]}
    # Thresholds are absolute accuracy requirements, fixed before comparison.
    assert matched["velocity"]["relative_l2"] < 1e-5
    assert matched["omega"]["relative_l2"] < 1e-5
    for diff in sensitivity.values():
        assert diff["velocity"]["relative_l2"] < 1e-5
        assert diff["omega"]["relative_l2"] < 1e-5
    benchmark = ghia_values()
    ghia = {"mf32": benchmark_error(mf, benchmark),
            **{f"dns{n}": benchmark_error(data, benchmark) for n, data in dns.items()}}
    assert max(ghia["mf32"][c]["maximum_absolute"] for c in ("u", "v")) < 0.02
    for c in ("u", "v"):
        assert ghia["dns125"][c]["maximum_absolute"] < ghia["dns32"][c]["maximum_absolute"]
    refinement = {}
    for coarse, fine in ((32, 63), (63, 125)):
        sampled = {k: dns[fine][k][::2, ::2] for k in ("psi", "omega", "u", "v")}
        refinement[f"{coarse}_vs_{fine}"] = difference(dns[coarse], sampled)
    assert refinement["63_vs_125"]["velocity"]["relative_l2"] < refinement["32_vs_63"]["velocity"]["relative_l2"]
    metrics = dict(passed=True, verification=verified, mean_field_vs_dns32=matched,
                   sensitivity=sensitivity, ghia_re100=ghia, dns_refinement=refinement,
                   artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(directory.glob("*.npz"))},
                   comparison_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/"comparison.json").write_text(json.dumps(metrics, indent=2)+"\n")
    plots(directory, mf, dns, benchmark, metrics)
    meta = json.loads(str(mf["metadata_json"]))
    base = verified["mf32"]
    lines = ["# Validated bosonic lid-driven cavity: 32×32, Re=100", "",
             "The production solution evolves 1,800 local Fock vectors (900 streamfunction and 900 vorticity sites). "
             "The 32×32 grid includes walls. The lid moves right at unit speed, all other walls are stationary. "
             "DNS and published values are used only in this validation directory, after the mean-field run.", "",
             "## Independent checks on the saved kets", "",
             f"- Both steady equations pass a maximum absolute residual tolerance of 1e-7: "
             f"streamfunction {base['poisson_residual']:.6e}; vorticity {base['vorticity_residual']:.6e}.",
             f"- Maximum discrete divergence: {base['divergence_max']:.6e}.",
             f"- Worst sampled coherent-state defect: {meta['worst_sampled_quality']['coherent_defect']:.6e}; "
             f"ceiling probability: {meta['worst_sampled_quality']['ceiling_probability']:.6e}.",
             f"- Primary streamfunction minimum: {base['psi_min']:.10f}, at grid point "
             f"(x,y)=({base['primary_vortex_grid_location'][0]:.6f},{base['primary_vortex_grid_location'][1]:.6f}).",
             f"- {meta['steps']:,} RK4 iterations, Δτ={meta['config']['dt']}, final τ={meta['final']['pseudo_time']:.3f}. "
             f"Measured runtime: {meta['elapsed_seconds']:.2f} s on one CPU core.", "",
             "τ is an artificial iteration coordinate; these snapshots are not a physical startup trajectory. "
             "Every run starts from vacuum interior kets; none is initialized from DNS.", "",
             "## Independent DNS and numerical sensitivity", "",
             "DNS separately integrates the physical vorticity equation using SSPRK3 and solves its own "
             "Dirichlet Poisson problem with sine transforms at each stage. It starts from a quiescent interior. "
             "The matched 32×32 comparison isolates the bosonic calculation from spatial discretization error.", "",
             "| Comparison | Velocity relative L2 | Interior vorticity relative L2 |",
             "|---|---:|---:|",
             f"| Mean field vs DNS32 | {matched['velocity']['relative_l2']:.6e} | {matched['omega']['relative_l2']:.6e} |"]
    for name, diff in sensitivity.items():
        lines.append(f"| {name} vs production | {diff['velocity']['relative_l2']:.6e} | {diff['omega']['relative_l2']:.6e} |")
    lines += ["", "Sensitivity runs change only the occupation cutoff (12→16), RK4 step (halved), "
              "both observable scales (halved), or streamfunction relaxation rate (0.1→0.05). "
              "They test the final steady solution; halving artificial time steps does not validate a physical transient. "
              "Reported norms use all interior nodes; singular lid-corner values are excluded.", "",
              "## Published benchmark and spatial accuracy", "",
              "Values were transcribed directly from Tables I–II (centerline velocities) and Table V "
              "(primary streamfunction minimum −0.103423) of "
              "[Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4). "
              "Linear interpolation is used at the centerlines and at the paper's rounded sample coordinates.", "",
              "| Run | Max absolute u error / lid speed | Max absolute v error / lid speed | ψ minimum |",
              "|---|---:|---:|---:|"]
    for name, entry in ghia.items():
        lines.append(f"| {name} | {entry['u']['maximum_absolute']:.6f} | {entry['v']['maximum_absolute']:.6f} | {entry['psi_min']:.9f} |")
    lines += ["", "| Nested DNS grids | Velocity relative L2 difference |",
              "|---|---:|"]
    for name, entry in refinement.items():
        lines.append(f"| {name.replace('_vs_', ' vs ')} | {entry['velocity']['relative_l2']:.6e} |")
    lines += ["", "The 63×63 and 125×125 DNS grids halve the spacing successively from 32×32. "
              "The refined DNS establishes the size and trend of spatial error. Close agreement with DNS32 "
              "verifies the mean-field implementation, but does not make 32×32 a grid-converged continuum solution. "
              "The discontinuous lid velocity at the corners and the Thom wall-vorticity closure limit coarse-grid accuracy. "
              "The centerline differences from the published Ghia table are not monotonic between 63×63 and 125×125; "
              "the decreasing nested-grid differences, rather than monotonic agreement with that table, are the spatial "
              "refinement check. The 32×32 primary streamfunction minimum differs from Ghia by about 3.16%.", "",
              "![Centerline comparison](centerlines.png)", "", "![Cavity fields](cavity_fields.png)", "",
              "![Convergence](convergence.png)", "",
              "Exactly zero initial residuals/defects are omitted from logarithmic curves.", "",
              "## Reproduce", "", "From the repository root, install requirements and run:", "", "```bash",
              "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ldc-matplotlib",
              "python3 -m unittest discover -s tests -v",
              "python3 -m validation.compare --results results", "```", "",
              "The full computation commands are in the root README and hpc/validate.sbatch. "
              "Solvers refuse to overwrite existing NPZ results: use a fresh output directory for recomputation. "
              "NPZ files contain the saved states or DNS fields, settings, convergence history, and source hashes. "
              "comparison.json records result-file SHA-256 hashes and all numerical metrics.", ""]
    (directory/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(passed=True, mean_field_vs_dns32=matched, ghia_re100=ghia), indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    compare(parser.parse_args().results)
