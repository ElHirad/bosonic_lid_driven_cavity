"""Checkpointed mean-field runs for larger cavities; no DNS dependency.

The spatial operators and local Fock-vector RK4 are the same as solver.py.
Checkpoints contain actual kets and never reconstruct them from field arrays.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import time
import numpy as np
import scipy
from .bosons import LocalBosons
from .operators import CavityOperators
from .solver import Config, source_hashes


def atomic_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path, **data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(config, directory, *, resume=False, checkpoint_every=1000, steps_per_run=None):
    config.validate()
    if checkpoint_every < 1 or (steps_per_run is not None and steps_per_run < 1):
        raise ValueError("checkpoint and run step limits must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, directory, resume, checkpoint_every, steps_per_run)


def _run(config, directory, resume, checkpoint_every, steps_per_run):
    checkpoint, output = directory/"checkpoint.npz", directory/"mean_field.npz"
    if output.exists():
        raise FileExistsError(f"completed output already exists: {output}")
    if checkpoint.exists() and not resume:
        raise FileExistsError("checkpoint exists; use --resume or a fresh directory")
    hashes = source_hashes()
    bosons = LocalBosons(config.cutoff)
    op = CavityOperators(config.n, config.reynolds, config.psi_scale,
                         config.omega_scale, config.relaxation)
    if resume:
        with np.load(checkpoint, allow_pickle=False) as data:
            previous = json.loads(str(data["metadata_json"]))
            if previous["config"] != asdict(config) or previous["source_sha256"] != hashes:
                raise ValueError("checkpoint configuration or production source differs")
            states = data["states"].copy()
            snapshots = list(data["snapshot_states"].copy())
            snapshot_steps = list(data["snapshot_steps"].astype(int))
            history = json.loads(str(data["history_json"]))
        step = int(previous["steps"])
        elapsed = previous["elapsed_seconds"]
        worst = previous["worst_sampled_quality"]
    else:
        states = bosons.coherent_states(np.zeros(op.size))
        snapshots, snapshot_steps, history = [states.copy()], [0], []
        step, elapsed = 0, 0.
        worst = {key: 0. for key in bosons.quality(states)}
    initial_step = step
    started = time.perf_counter()
    last_checkpoint = step
    stop_requested = []
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        old_handlers[signum] = signal.signal(signum, lambda signum, frame: stop_requested.append(signum))
    targets = {round(t/config.dt) for t in (1, 5, 10, 20, 40, 80, 160, 320)}
    max_steps = int(np.ceil(config.max_time/config.dt))

    def metadata(converged):
        return dict(config=asdict(config), converged=converged, steps=int(step),
                    elapsed_seconds=elapsed+time.perf_counter()-started,
                    method="single-site Fock-vector RK4; coupled artificial-time relaxation",
                    source_sha256=hashes, python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    worst_sampled_quality=worst, final=history[-1])

    def save(converged, state, reason):
        meta = metadata(converged)
        saved_states, saved_steps = list(snapshots), list(snapshot_steps)
        if saved_steps[-1] != step:
            saved_states.append(states.copy())
            saved_steps.append(step)
        # Checkpoints retain all saved kets, including the current accepted state.
        atomic_npz(output if converged else checkpoint, states=states,
                   snapshot_states=np.stack(saved_states), snapshot_steps=np.asarray(saved_steps),
                   history_json=np.array(json.dumps(history)), metadata_json=np.array(json.dumps(meta)),
                   x=np.linspace(0, 1, config.n), y=np.linspace(0, 1, config.n),
                   **op.fields(bosons.amplitudes(states)))
        if converged:
            atomic_json(output.with_suffix(".json"), meta)
        atomic_json(directory/"status.json", dict(state=state, reason=reason, **meta))
        return meta

    try:
        while True:
            stopping = (bool(stop_requested) or step >= max_steps
                        or (steps_per_run is not None and step-initial_step >= steps_per_run))
            if step % config.check_every == 0 or stopping:
                quality = bosons.quality(states)
                residuals = op.residuals(bosons.amplitudes(states))
                for key, limit in {"coherent_defect": 1e-5, "ceiling_probability": 1e-9,
                                   "norm_error": 1e-12, "imaginary_amplitude": 1e-12}.items():
                    if not np.isfinite(quality[key]) or quality[key] > limit:
                        raise FloatingPointError(f"step {step}: {key}={quality[key]} exceeds {limit}")
                    worst[key] = max(worst[key], quality[key])
                if not all(np.isfinite(value) for value in residuals.values()):
                    raise FloatingPointError("nonfinite steady residual")
                row = dict(step=int(step), pseudo_time=step*config.dt, **residuals, **quality)
                if history and history[-1]["step"] == step:
                    history[-1] = row
                else:
                    history.append(row)
                if step % (10*config.check_every) == 0 or stopping:
                    print(json.dumps(dict(**row, elapsed_seconds=elapsed+time.perf_counter()-started)), flush=True)
                if max(residuals.values()) <= config.tolerance:
                    return save(True, "converged", "both steady residuals met tolerance")
                if stopping:
                    reason = "signal received" if stop_requested else "iteration limit reached"
                    return save(False, "stopped", reason)
            if step-last_checkpoint >= checkpoint_every:
                save(False, "running", "periodic checkpoint")
                last_checkpoint = step
            states = bosons.advance(states, config.dt, op.generator)
            step += 1
            if step in targets:
                snapshots.append(states.copy())
                snapshot_steps.append(step)
    except BaseException as error:
        atomic_json(directory/"status.json", dict(state="failed", step=int(step), reason=str(error)))
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--steps-per-run", type=int)
    defaults = dict(n=128, reynolds=1000., dt=0.00125, relaxation=0.01,
                    psi_scale=1000., omega_scale=100000., max_time=400.)
    for key, field in Config.__dataclass_fields__.items():
        parser.add_argument("--"+key.replace("_", "-"), type=field.type,
                            default=defaults.get(key, field.default))
    args = vars(parser.parse_args())
    controls = {key: args.pop(key) for key in ("directory", "resume", "checkpoint_every", "steps_per_run")}
    result = run(Config(**args), **controls)
    print(json.dumps(result, indent=2), flush=True)
    if not result["converged"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
