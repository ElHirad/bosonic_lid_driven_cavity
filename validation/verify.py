"""Recompute physical and quantum checks from saved local kets, independently."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .dns import laplacian, transport, velocity, walls


def verify(path):
    with np.load(path, allow_pickle=False) as result:
        meta = json.loads(str(result["metadata_json"]))
        config = meta["config"]
        source_dir = Path(__file__).resolve().parents[1]/"ldc_mean_field"
        for name, digest in meta["source_sha256"].items():
            assert hashlib.sha256((source_dir/name).read_bytes()).hexdigest() == digest, "production source changed"
        if not meta["converged"]:
            raise AssertionError("unconverged result")
        n, cutoff = config["n"], config["cutoff"]
        h, m = 1/(n-1), (n-2)**2
        states, saved = result["states"], result["snapshot_states"]
        assert states.shape == (2*m, cutoff+1)
        assert saved.shape[1:] == states.shape
        assert np.all(np.isfinite(saved))
        np.testing.assert_array_equal(saved[-1], states)
        vacuum = np.zeros_like(states)
        vacuum[:, 0] = 1
        np.testing.assert_array_equal(saved[0], vacuum)
        assert result["snapshot_steps"][0] == 0
        assert result["snapshot_steps"][-1] == meta["steps"]
        np.testing.assert_allclose(result["x"], np.linspace(0, 1, n), atol=0, rtol=0)
        np.testing.assert_allclose(result["y"], np.linspace(0, 1, n), atol=0, rtol=0)
        assert np.all(np.diff(result["snapshot_steps"]) > 0)
        norms = np.sum(np.abs(saved)**2, axis=-1)
        weights = np.sqrt(np.arange(1, cutoff+1))
        amplitudes = np.sum(saved[..., :-1].conj()*saved[..., 1:]*weights, axis=-1)/norms
        assert np.max(np.abs(norms-1)) < 1e-12
        assert np.max(np.abs(amplitudes.imag)) < 1e-12
        lowered = np.zeros_like(saved)
        lowered[..., :-1] = saved[..., 1:]*weights
        defect = float(np.max(np.linalg.norm(lowered-amplitudes[..., None]*saved, axis=-1)))
        ceiling = float(np.max(np.abs(saved[..., -1])**2))
        assert defect < 1e-5
        assert ceiling < 1e-9
        alpha = amplitudes[-1].real
        psi = np.zeros((n, n))
        psi[1:-1, 1:-1] = config["psi_scale"]*alpha[:m].reshape(n-2, n-2)
        interior = config["omega_scale"]*alpha[m:].reshape(n-2, n-2)
        omega = walls(psi, interior, h)
        u, v = velocity(psi, h)
        for key, value in dict(psi=psi, omega=omega, u=u, v=v).items():
            np.testing.assert_allclose(result[key], value, atol=1e-12, rtol=1e-12)
        poisson = float(np.max(np.abs(laplacian(psi, h)+interior)))
        vorticity = float(np.max(np.abs(transport(psi, omega, h, config["reynolds"]))))
        assert max(poisson, vorticity) <= config["tolerance"]*1.001 + 1e-10
        divergence = ((u[1:-1, 2:]-u[1:-1, :-2])+(v[2:, 1:-1]-v[:-2, 1:-1]))/(2*h)
        divergence_max = float(np.max(np.abs(divergence)))
        assert divergence_max < 1e-12
        j, i = np.unravel_index(np.argmin(psi), psi.shape)
        assert psi[j, i] < 0  # Clockwise primary vortex with the stated sign convention.
        history = json.loads(str(result["history_json"]))
        assert history[-1]["step"] == meta["steps"]
        assert len(history) >= 2
        for row in history:
            assert all(np.isfinite(value) for value in row.values())
            assert row["coherent_defect"] < 1e-5
            assert row["ceiling_probability"] < 1e-9
            assert row["norm_error"] < 1e-12
        report = dict(verified=True, n=n, reynolds=config["reynolds"],
                      poisson_residual=poisson, vorticity_residual=vorticity,
                      divergence_max=divergence_max, snapshot_coherent_defect=defect,
                      snapshot_ceiling_probability=ceiling,
                      psi_min=float(psi[j, i]), primary_vortex_grid_location=[i*h, j*h],
                      checks="saved kets, initialization, observables, walls, PDE residuals, divergence, quantum quality")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    report = verify(args.result)
    args.result.with_suffix(".verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
