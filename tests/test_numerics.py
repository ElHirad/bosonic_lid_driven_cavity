import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from ldc_mean_field.bosons import LocalBosons
from ldc_mean_field.operators import CavityOperators, PolynomialGenerator
from ldc_mean_field.solver import Config
from validation.dns import Reference, laplacian, transport, walls
from validation.verify import verify


class NumericalTests(unittest.TestCase):
    def test_polynomial_matches_independent_stencils_including_walls(self):
        rng = np.random.default_rng(13)
        for n in (5, 8, 32):
            op = CavityOperators(n, 100., 0.7, 80., 0.1)
            alpha = rng.normal(size=op.size)*0.02
            psi = np.zeros((n, n))
            psi[1:-1, 1:-1] = 0.7*alpha[:op.m].reshape(n-2, n-2)
            interior = 80*alpha[op.m:].reshape(n-2, n-2)
            omega = walls(psi, interior, op.h)
            expected = np.r_[0.1/0.7*(laplacian(psi, op.h)+interior).ravel(),
                             transport(psi, omega, op.h, 100.).ravel()/80]
            np.testing.assert_allclose(op.generator(alpha), expected, atol=2e-13, rtol=2e-13)
            np.testing.assert_allclose(op.fields(alpha)["omega"], omega, atol=1e-13)

    def test_full_decoupling_jacobian_and_repeated_site_terms(self):
        generator = PolynomialGenerator(3, {(0, ()): 2., (0, (1, 1)): 3.,
                                             (1, (0, 2)): -0.4, (2, (1,)): 0.7})
        alpha = np.array([0.1+0.03j, -0.2+0.02j, 0.13-0.04j])
        f, b, c = generator.local_coefficients(alpha)
        eps = 1e-6
        jacobian = np.column_stack([(generator(alpha+eps*e)-generator(alpha-eps*e))/(2*eps)
                                    for e in np.eye(3)])
        np.testing.assert_allclose(b, jacobian.T @ alpha.conj(), atol=1e-10)
        np.testing.assert_allclose(c, -alpha*b, atol=1e-15)
        self.assertAlmostEqual(f[0], 2+3*alpha[1]**2)

    def test_local_operator_evolution_has_expected_coherent_tangent(self):
        bosons = LocalBosons(14)
        op = CavityOperators(5)
        alpha = np.linspace(-0.05, 0.05, op.size)
        states = bosons.coherent_states(alpha)
        derivative = bosons.derivative(states, op.generator)
        a = bosons.annihilation
        observed = (np.sum(derivative*(states @ a.T), axis=1)
                    + np.sum(states*(derivative @ a.T), axis=1))
        np.testing.assert_allclose(observed, op.generator(alpha), atol=3e-15)
        advanced = bosons.advance(states, 1e-4, op.generator)
        self.assertLess(bosons.quality(advanced)["norm_error"], 1e-14)

    def test_lid_forcing_sign_and_vacuum_streamfunction(self):
        op = CavityOperators(32)
        f = op.generator(np.zeros(op.size))
        omega_dot = f[op.m:].reshape(30, 30)*100
        np.testing.assert_array_equal(f[:op.m], 0)
        np.testing.assert_array_equal(omega_dot[:-1], 0)
        np.testing.assert_allclose(omega_dot[-1], -2/(100*op.h**3))

    def test_manufactured_poisson_solution_and_refinement(self):
        errors = []
        for n in (17, 33, 65):
            ref = Reference(n)
            x = np.linspace(0, 1, n)
            exact = np.sin(np.pi*x[:, None])*np.sin(np.pi*x[None, :])
            numerical = ref.poisson(2*np.pi**2*exact[1:-1, 1:-1])
            errors.append(np.max(np.abs(numerical-exact)))
            discrete = ref.poisson(-laplacian(exact, ref.h))
            np.testing.assert_allclose(discrete, exact, atol=2e-14)
        self.assertGreater(errors[0]/errors[1], 3.9)
        self.assertGreater(errors[1]/errors[2], 3.9)

    def test_invalid_configuration(self):
        for kwargs in ({"n": 3}, {"dt": 1}, {"reynolds": float("nan")}, {"cutoff": 2.5}):
            with self.assertRaises(ValueError):
                Config(**kwargs).validate()

    def test_production_dependency_boundary(self):
        root = Path(__file__).resolve().parents[1]/"ldc_mean_field"
        forbidden = ("validation", "mixing_layer", "scipy.fft", "scipy.sparse.linalg")
        for file in root.glob("*.py"):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    self.assertFalse(name.startswith(forbidden), (file, name))

    def test_production_runs_when_dns_and_poisson_routines_are_blocked(self):
        code = '''
import importlib.abc
import sys
class BlockReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("validation", "mixing_layer")):
            raise ImportError("reference calculation blocked: " + fullname)
sys.meta_path.insert(0, BlockReference())
import numpy as np
import scipy.fft
import scipy.sparse.linalg
def forbidden(*args, **kwargs):
    raise AssertionError("field inverse or transform called by production")
for name in ("spsolve", "splu", "factorized", "cg", "gmres", "bicgstab"):
    setattr(scipy.sparse.linalg, name, forbidden)
for name in ("dst", "dstn", "idst", "idstn", "fft", "fft2", "fftn", "dct", "dctn"):
    setattr(scipy.fft, name, forbidden)
np.linalg.solve = np.linalg.inv = forbidden
from ldc_mean_field.bosons import LocalBosons
from ldc_mean_field.operators import CavityOperators
op = CavityOperators(8)
b = LocalBosons(12)
states = b.coherent_states(np.zeros(op.size))
for _ in range(4):
    states = b.advance(states, 1e-4, op.generator)
assert op.fields(b.amplitudes(states))["omega"][1:-1, 1:-1].min() < 0
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_saved_result_verification_and_corruption_detection(self):
        result = Path(__file__).resolve().parents[1]/"results"/"mf32.npz"
        if not result.exists():
            self.skipTest("production result not present")
        self.assertTrue(verify(result)["verified"])
        with np.load(result, allow_pickle=False) as source:
            data = {key: source[key].copy() for key in source.files}
        data["u"][10, 10] += 0.01
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"corrupted.npz"
            np.savez_compressed(path, **data)
            with self.assertRaises(AssertionError):
                verify(path)


if __name__ == "__main__":
    unittest.main()
