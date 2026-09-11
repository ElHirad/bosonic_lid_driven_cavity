"""Solve the steady cavity by evolving only local bosonic Fock vectors.

The iteration coordinate is artificial time, not the physical startup history.
Neither DNS, a direct Poisson solver, nor benchmark data is used here.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from .bosons import LocalBosons
from .operators import CavityOperators


@dataclass(frozen=True)
class Config:
    n: int = 32
    reynolds: float = 100.
    cutoff: int = 12
    dt: float = 0.0025
    max_time: float = 150.
    tolerance: float = 1e-7
    relaxation: float = 0.1
    psi_scale: float = 100.
    omega_scale: float = 10000.
    check_every: int = 100

    def validate(self):
        for key in ("n", "cutoff", "check_every"):
            if int(getattr(self, key)) != getattr(self, key):
                raise ValueError(f"{key} must be an integer")
        if self.n < 5 or self.cutoff < 2 or self.check_every < 1:
            raise ValueError("invalid grid, cutoff or check interval")
        for key in ("reynolds", "dt", "max_time", "tolerance", "relaxation", "psi_scale", "omega_scale"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.dt > 0.25/((self.n-1)**2*max(self.relaxation, 1/self.reynolds)):
            raise ValueError("dt exceeds the conservative explicit diffusion bound")


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def run(config, output):
    config.validate()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the result before computing; never replace an existing run.
    with output.open("xb"):
        pass
    bosons = LocalBosons(config.cutoff)
    op = CavityOperators(config.n, config.reynolds, config.psi_scale,
                         config.omega_scale, config.relaxation)
    states = bosons.coherent_states(np.zeros(op.size))
    snapshots, snapshot_steps, history = [states.copy()], [0], []
    worst_quality = {k: 0. for k in bosons.quality(states)}
    started, converged = time.perf_counter(), False
    requested_snapshots = {int(t/config.dt) for t in (1, 5, 10, 20, 40, 80)}
    steps = int(np.ceil(config.max_time/config.dt))
    try:
        for step in range(steps+1):
            if step % config.check_every == 0 or step == steps:
                alpha = bosons.amplitudes(states)
                residuals, quality = op.residuals(alpha), bosons.quality(states)
                for key, value in quality.items():
                    worst_quality[key] = max(worst_quality[key], value)
                for key, limit in {"coherent_defect": 1e-5, "ceiling_probability": 1e-9,
                                   "norm_error": 1e-12, "imaginary_amplitude": 1e-12}.items():
                    if not np.isfinite(quality[key]) or quality[key] > limit:
                        raise FloatingPointError(f"step {step}: {key}={quality[key]:.4e} exceeds {limit:.4e}")
                row = dict(step=step, pseudo_time=step*config.dt, **residuals, **quality)
                history.append(row)
                if step % (10*config.check_every) == 0:
                    print(json.dumps(row), flush=True)
                if max(residuals.values()) <= config.tolerance:
                    converged = True
                    break
            if step == steps:
                break
            states = bosons.advance(states, config.dt, op.generator)
            if step+1 in requested_snapshots:
                snapshots.append(states.copy())
                snapshot_steps.append(step+1)
        if snapshot_steps[-1] != step:
            snapshots.append(states.copy())
            snapshot_steps.append(step)
        metadata = dict(config=asdict(config), converged=converged, steps=step,
                        elapsed_seconds=time.perf_counter()-started,
                        method="single-site Fock-vector RK4; coupled artificial-time relaxation",
                        source_sha256=source_hashes(), python=platform.python_version(),
                        numpy=np.__version__, scipy=scipy.__version__,
                        worst_sampled_quality=worst_quality, final=history[-1])
        fields = op.fields(bosons.amplitudes(states))
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, states=states, snapshot_states=np.stack(snapshots),
                            snapshot_steps=snapshot_steps, history_json=np.array(json.dumps(history)),
                            metadata_json=np.array(json.dumps(metadata)),
                            x=np.linspace(0, 1, config.n), y=np.linspace(0, 1, config.n), **fields)
        temporary.replace(output)
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
        if not converged:
            raise RuntimeError(f"did not converge by pseudo-time {step*config.dt}; partial result saved")
        print(json.dumps(metadata, indent=2), flush=True)
        return metadata
    except BaseException:
        if output.exists() and output.stat().st_size == 0:
            output.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/mf32.npz"))
    for key, field in Config.__dataclass_fields__.items():
        parser.add_argument("--"+key.replace("_", "-"), type=field.type, default=field.default)
    args = vars(parser.parse_args())
    output = args.pop("output")
    run(Config(**args), output)


if __name__ == "__main__":
    main()
