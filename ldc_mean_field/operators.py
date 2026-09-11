"""Normally ordered streamfunction/vorticity bosonic generator.

Sparse arrays store polynomial monomial coefficients, never a field inverse.
Wall conditions are eliminated into those coefficients before ket evolution.
"""
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix


class PolynomialGenerator:
    def __init__(self, size, terms):
        self.size = size
        self.constant = np.zeros(size)
        linear, quadratic = [], []
        for (r, sources), value in terms.items():
            if not value:
                continue
            if not sources:
                self.constant[r] += value
            elif len(sources) == 1:
                linear.append((r, sources[0], value))
            else:
                quadratic.append((r, *sources, value))
        self.linear = coo_matrix(([v for r, s, v in linear],
                                  ([r for r, s, v in linear], [s for r, s, v in linear])),
                                 shape=(size, size)).tocsr()
        self.transpose = self.linear.T.tocsr()
        self.target = np.array([r for r, s, t, v in quadratic], dtype=int)
        self.source = np.array([s for r, s, t, v in quadratic], dtype=int)
        self.other = np.array([t for r, s, t, v in quadratic], dtype=int)
        self.coefficient = np.array([v for r, s, t, v in quadratic])

    def scatter(self, indices, values):
        if np.iscomplexobj(values):
            return (np.bincount(indices, weights=values.real, minlength=self.size)
                    + 1j*np.bincount(indices, weights=values.imag, minlength=self.size))
        return np.bincount(indices, weights=values, minlength=self.size)

    def __call__(self, alpha):
        return (self.constant + self.linear @ alpha
                + self.scatter(self.target, self.coefficient*alpha[self.source]*alpha[self.other]))

    def local_coefficients(self, alpha):
        """Full first-order factorization: f=F, b=J(F)^T alpha*, c=-alpha b."""
        weights = self.coefficient * alpha[self.target].conj()
        b = (self.transpose @ alpha.conj()
             + self.scatter(self.source, weights*alpha[self.other])
             + self.scatter(self.other, weights*alpha[self.source]))
        return self(alpha), b, -alpha*b


class CavityOperators:
    def __init__(self, n=32, reynolds=100., psi_scale=1., omega_scale=100., relaxation=0.1):
        self.n, self.h, self.reynolds = n, 1/(n-1), reynolds
        self.m = (n-2)**2
        self.size = 2*self.m
        self.scales = {"psi": psi_scale, "omega": omega_scale}
        self.relaxation = relaxation
        terms = defaultdict(float)
        h = self.h

        def add(target, coefficient, *factors):
            products = [((), coefficient / self.scales[target[0]])]
            for factor in factors:
                products = [(tuple(sorted(sources + (() if index is None else (index,)))), value*c)
                            for sources, value in products
                            for index, c in self.expression(*factor)]
            for sources, value in products:
                terms[(self.index(*target), sources)] += value

        for j in range(1, n-1):
            for i in range(1, n-1):
                p, w = ("psi", j, i), ("omega", j, i)
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    add(p, relaxation/h**2, ("psi", j+dj, i+di))
                    add(w, 1/(reynolds*h**2), ("omega", j+dj, i+di))
                add(p, -4*relaxation/h**2, p)
                add(p, relaxation, w)
                add(w, -4/(reynolds*h**2), w)
                # -psi_y omega_x + psi_x omega_y, both centered.
                for a in (-1, 1):
                    for b in (-1, 1):
                        add(w, -a*b/(4*h*h), ("psi", j+a, i), ("omega", j, i+b))
                        add(w, a*b/(4*h*h), ("psi", j, i+a), ("omega", j+b, i))
        self.generator = PolynomialGenerator(self.size, terms)

    def index(self, field, j, i):
        if not (1 <= j < self.n-1 and 1 <= i < self.n-1):
            raise IndexError("only interior sites have independent kets")
        return (self.m if field == "omega" else 0) + (j-1)*(self.n-2) + i-1

    def expression(self, field, j, i):
        """Physical field = affine expression in annihilation operators."""
        n, h = self.n, self.h
        if 1 <= j < n-1 and 1 <= i < n-1:
            return [(self.index(field, j, i), self.scales[field])]
        if field == "psi":
            return []  # Homogeneous Dirichlet streamfunction, no gauge mode.
        if j in (0, n-1) and i in (0, n-1):
            raise IndexError("corner vorticity does not enter an interior stencil")
        jj, ii = min(max(j, 1), n-2), min(max(i, 1), n-2)
        result = [(self.index("psi", jj, ii), -2*self.scales["psi"]/h**2)]
        if j == n-1:
            result.append((None, -2/h))  # Right-moving unit lid, omega = -laplacian(psi).
        return result

    def fields(self, alpha):
        """Measure fields and evaluate eliminated wall observables."""
        n, h = self.n, self.h
        psi, omega = np.zeros((n, n)), np.zeros((n, n))
        psi[1:-1, 1:-1] = self.scales["psi"] * alpha[:self.m].real.reshape(n-2, n-2)
        omega[1:-1, 1:-1] = self.scales["omega"] * alpha[self.m:].real.reshape(n-2, n-2)
        omega[0, 1:-1] = -2*psi[1, 1:-1]/h**2
        omega[-1, 1:-1] = -2*psi[-2, 1:-1]/h**2 - 2/h
        omega[1:-1, 0] = -2*psi[1:-1, 1]/h**2
        omega[1:-1, -1] = -2*psi[1:-1, -2]/h**2
        # A plotting convention only: corners never enter the discrete PDE.
        for j, jj in ((0, 1), (-1, -2)):
            for i, ii in ((0, 1), (-1, -2)):
                omega[j, i] = (omega[jj, i]+omega[j, ii])/2
        u, v = np.zeros_like(psi), np.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[2:, 1:-1]-psi[:-2, 1:-1])/(2*h)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:]-psi[1:-1, :-2])/(2*h)
        u[-1, 1:-1] = 1  # Stationary-wall convention at the two singular lid corners.
        return dict(psi=psi, omega=omega, u=u, v=v)

    def residuals(self, alpha):
        f = self.generator(alpha)
        return {"poisson_residual": float(np.max(np.abs(f[:self.m]))*self.scales["psi"]/self.relaxation),
                "vorticity_residual": float(np.max(np.abs(f[self.m:]))*self.scales["omega"])}
