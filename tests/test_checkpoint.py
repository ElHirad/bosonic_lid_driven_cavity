"""Resume must preserve the actual Fock-vector trajectory."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ldc_mean_field.run_case import run
from ldc_mean_field.solver import Config


class CheckpointTests(unittest.TestCase):
    def test_resumed_kets_match_uninterrupted_run_and_config_guard(self):
        config = Config(n=8, reynolds=1000., max_time=0.02, check_every=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run(config, root/"split", checkpoint_every=2, steps_per_run=3)
            with self.assertRaises(FileExistsError):
                run(config, root/"split")
            with self.assertRaises(ValueError):
                run(Config(n=8, reynolds=100.), root/"split", resume=True)
            resumed = run(config, root/"split", resume=True, checkpoint_every=2, steps_per_run=5)
            full = run(config, root/"whole", checkpoint_every=2, steps_per_run=8)
            self.assertFalse(resumed["converged"])
            self.assertEqual(resumed["steps"], full["steps"])
            with np.load(root/"split"/"checkpoint.npz") as a, np.load(root/"whole"/"checkpoint.npz") as b:
                np.testing.assert_array_equal(a["states"], b["states"])
                for field in ("psi", "omega", "u", "v"):
                    np.testing.assert_array_equal(a[field], b[field])


if __name__ == "__main__":
    unittest.main()
