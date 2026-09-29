"""Check the reported L2 normalization against a hand-computable field."""
import unittest
import numpy as np
from validation.compare import difference, ghia_values


class ComparisonTests(unittest.TestCase):
    def test_reynolds_number_selects_correct_published_samples(self):
        low, high = ghia_values(100), ghia_values(1000)
        self.assertEqual(high["u"].shape, (17, 2))
        self.assertEqual(high["v"].shape, (17, 2))
        self.assertAlmostEqual(high["u"][high["u"][:, 0] == 0.5, 1].item(), -0.06080)
        self.assertAlmostEqual(low["u"][low["u"][:, 0] == 0.5, 1].item(), -0.20581)
        with self.assertRaises(ValueError):
            ghia_values(500)

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
