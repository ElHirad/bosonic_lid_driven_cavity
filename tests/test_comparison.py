"""Check the reported L2 normalization against a hand-computable field."""
import unittest
import numpy as np
from validation.compare import difference


class ComparisonTests(unittest.TestCase):
    def test_absolute_spatial_relative_and_vector_norms_exclude_walls(self):
        reference = {key: np.full((5, 5), 2.) for key in ("psi", "omega", "u", "v")}
        candidate = {key: np.full((5, 5), 5.) for key in reference}
        for field in candidate.values():
            field[[0, -1], :] = 1e6
            field[:, [0, -1]] = 1e6
        errors = difference(candidate, reference)
        # Nine interior values, each with error 3; h=1/4.
        for key in reference:
            self.assertAlmostEqual(errors[key]["absolute_l2"], 9.)
            self.assertAlmostEqual(errors[key]["spatial_l2"], 2.25)
            self.assertAlmostEqual(errors[key]["relative_l2"], 1.5)
            self.assertAlmostEqual(errors[key]["maximum_absolute"], 3.)
        self.assertAlmostEqual(errors["velocity"]["absolute_l2"], 9*np.sqrt(2))
        self.assertAlmostEqual(errors["velocity"]["spatial_l2"], 2.25*np.sqrt(2))
        self.assertAlmostEqual(errors["velocity"]["relative_l2"], 1.5)


if __name__ == "__main__":
    unittest.main()
