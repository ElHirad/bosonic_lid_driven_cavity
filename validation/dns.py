"""Independent classical DNS reference, used only after the MF result exists.

Array-based vorticity transport, SSPRK3 physical time integration, and a
Dirichlet Poisson inverse using discrete sine transforms at every stage.
No code or initialization is imported from the mean-field solver.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.fft import dstn, idstn


def laplacian(a, h):
    return (a[2:, 1:-1]+a[:-2, 1:-1]+a[1:-1, 2:]+a[1:-1, :-2]-4*a[1:-1, 1:-1])/h**2


def walls(psi, interior, h):
    omega = np.zeros_like(psi)
    omega[1:-1, 1:-1] = interior
    omega[0, 1:-1] = -2*psi[1, 1:-1]/h**2
    omega[-1, 1:-1] = -2*psi[-2, 1:-1]/h**2-2/h
    omega[1:-1, 0] = -2*psi[1:-1, 1]/h**2
    omega[1:-1, -1] = -2*psi[1:-1, -2]/h**2
    for j, jj in ((0, 1), (-1, -2)):
        for i, ii in ((0, 1), (-1, -2)):
            omega[j, i] = (omega[jj, i]+omega[j, ii])/2
    return omega


def velocity(psi, h):
    u, v = np.zeros_like(psi), np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[2:, 1:-1]-psi[:-2, 1:-1])/(2*h)
    v[1:-1, 1:-1] = (psi[1:-1, :-2]-psi[1:-1, 2:])/(2*h)
    u[-1, 1:-1] = 1
    return u, v


def transport(psi, omega, h, reynolds):
    u = (psi[2:, 1:-1]-psi[:-2, 1:-1])/(2*h)
    v = (psi[1:-1, :-2]-psi[1:-1, 2:])/(2*h)
    return (laplacian(omega, h)/reynolds
            - u*(omega[1:-1, 2:]-omega[1:-1, :-2])/(2*h)
            - v*(omega[2:, 1:-1]-omega[:-2, 1:-1])/(2*h))


class Reference:
    def __init__(self, n, reynolds=100.):
        self.n, self.h, self.reynolds = n, 1/(n-1), reynolds
        mode = np.arange(1, n-1)
        eigen = 4*np.sin(np.pi*mode/(2*(n-1)))**2/self.h**2
        self.eigenvalues = eigen[:, None]+eigen[None, :]

    def poisson(self, interior):
        psi = np.zeros((self.n, self.n))
        psi[1:-1, 1:-1] = idstn(dstn(interior, type=1, norm="ortho")/self.eigenvalues,
                               type=1, norm="ortho")
        return psi

    def rhs(self, interior):
        psi = self.poisson(interior)
        return transport(psi, walls(psi, interior, self.h), self.h, self.reynolds)

    def step(self, omega, dt):
        first = omega + dt*self.rhs(omega)
        second = 0.75*omega+0.25*(first+dt*self.rhs(first))
        return omega/3+2*(second+dt*self.rhs(second))/3


def run(n, output, reynolds=100., tolerance=1e-8, max_time=100., dt=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    reference = Reference(n, reynolds)
    h = reference.h
    if dt is None:
        dt = min(0.005, 0.2*reynolds*h*h, 0.25*h)
    if not np.isfinite(dt) or dt <= 0 or dt > min(0.25*reynolds*h*h, 0.5*h):
        raise ValueError("invalid DNS timestep")
    omega = np.zeros((n-2, n-2))
    history, converged = [], False
    started = time.perf_counter()
    for step in range(int(np.ceil(max_time/dt))+1):
        if step % 100 == 0:
            residual = float(np.max(np.abs(reference.rhs(omega))))
            if not np.isfinite(residual):
                raise FloatingPointError("nonfinite DNS reference")
            history.append(dict(step=step, time=step*dt, residual=residual))
            if step % 2000 == 0:
                print(json.dumps(dict(n=n, **history[-1])), flush=True)
            if residual < tolerance:
                converged = True
                break
        omega = reference.step(omega, dt)
    if not converged:
        raise RuntimeError("DNS reference failed to converge")
    psi = reference.poisson(omega)
    full_omega = walls(psi, omega, h)
    u, v = velocity(psi, h)
    metadata = dict(n=n, reynolds=reynolds, dt=dt, steps=step, time=step*dt,
                    tolerance=tolerance, converged=converged,
                    vorticity_residual=residual,
                    poisson_residual=float(np.max(np.abs(laplacian(psi, h)+omega))),
                    elapsed_seconds=time.perf_counter()-started,
                    method="independent DNS: centered vorticity, SSPRK3, DST Poisson solve",
                    initialization="quiescent interior; unit lid; no mean-field state input",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        np.savez_compressed(file, psi=psi, omega=full_omega, u=u, v=v,
                            x=np.linspace(0, 1, n), y=np.linspace(0, 1, n),
                            metadata_json=np.array(json.dumps(metadata)),
                            history_json=np.array(json.dumps(history)))
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/dns32.npz"))
    parser.add_argument("--reynolds", type=float, default=100.)
    parser.add_argument("--dt", type=float)
    args = parser.parse_args()
    run(**vars(args))
