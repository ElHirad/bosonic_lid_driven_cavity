"""Normalized local Fock-vector evolution; no scalar field integrator.

Adapted from ElHirad/bosonic_mixing_layer, commit 7e0b97a.
"""
import numpy as np


class LocalBosons:
    def __init__(self, cutoff):
        if int(cutoff) != cutoff or cutoff < 2:
            raise ValueError("cutoff must be an integer >= 2")
        self.dimension = int(cutoff) + 1
        self.annihilation = np.diag(np.sqrt(np.arange(1, self.dimension)), 1)
        self.creation = self.annihilation.T.copy()
        self.identity = np.eye(self.dimension)

    @staticmethod
    def normalize(states):
        norms = np.sqrt(np.sum(np.abs(states)**2, axis=1))
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise FloatingPointError("invalid local state norm")
        return states / norms[:, None]

    def coherent_states(self, alpha):
        """Used only at initialization, never during evolution."""
        states = np.ones((len(alpha), self.dimension), dtype=np.result_type(alpha, float))
        for m in range(1, self.dimension):
            states[:, m] = states[:, m-1] * alpha / np.sqrt(m)
        return self.normalize(states)

    @staticmethod
    def expectation(states, action):
        return np.sum(states.conj()*action, axis=1) / np.sum(np.abs(states)**2, axis=1)

    def amplitudes(self, states):
        return self.expectation(states, states @ self.annihilation.T)

    def derivative(self, states, generator):
        lowered = states @ self.annihilation.T
        alpha = self.expectation(states, lowered)
        f, b, c = generator.local_coefficients(alpha)
        action = (f[:, None] * (states @ self.creation.T)
                  + b[:, None] * lowered + c[:, None] * states)
        return action - self.expectation(states, action).real[:, None] * states

    def advance(self, states, dt, generator):
        """RK4 advances Fock vectors and remeasures couplings at every stage."""
        k1 = self.derivative(states, generator)
        k2 = self.derivative(states + dt*k1/2, generator)
        k3 = self.derivative(states + dt*k2/2, generator)
        k4 = self.derivative(states + dt*k3, generator)
        return self.normalize(states + dt*(k1+2*k2+2*k3+k4)/6)

    def quality(self, states):
        alpha = self.amplitudes(states)
        defect = states @ self.annihilation.T - alpha[:, None]*states
        return {
            "coherent_defect": float(np.max(np.linalg.norm(defect, axis=1))),
            "ceiling_probability": float(np.max(np.abs(states[:, -1])**2)),
            "norm_error": float(np.max(np.abs(np.sum(np.abs(states)**2, axis=1)-1))),
            "imaginary_amplitude": float(np.max(np.abs(alpha.imag))),
        }
