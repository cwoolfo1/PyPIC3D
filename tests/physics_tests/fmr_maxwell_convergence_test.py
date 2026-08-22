import unittest

from tests.fmr_support import (
    REGIONS,
    ROOT_RESOLUTIONS,
    _observed_orders,
    _result_table,
    _run_problem,
)


class TestFMRMaxwellConvergence(unittest.TestCase):
    def _assert_problem(self, problem):
        results = [_run_problem(problem, resolution) for resolution in ROOT_RESOLUTIONS]
        orders = {
            diagnostic: _observed_orders(results, diagnostic)
            for diagnostic in ("E_l2", "B_l2", "EM_l2")
        }
        message = _result_table(problem, results, orders)

        for result in results:
            self.assertTrue(result["finite"], msg=message)
            self.assertLess(result["interface_residual"], 2.0e-12, msg=message)
            self.assertLess(result["energy_max_relative_error"], 5.0e-2, msg=message)

        for region in REGIONS:
            self.assertGreater(orders["EM_l2"][region][-1], 1.8, msg=message)
            self.assertLess(results[-1]["divB_max"][region], 2.0e-10, msg=message)

        energy_errors = [result["energy_max_relative_error"] for result in results]
        self.assertGreater(energy_errors[0], energy_errors[1], msg=message)
        self.assertGreater(energy_errors[1], energy_errors[2], msg=message)
        final_energy_errors = [
            abs(result["energy_final_relative_error"])
            for result in results
        ]
        self.assertGreater(final_energy_errors[0], final_energy_errors[1], msg=message)
        self.assertGreater(final_energy_errors[1], final_energy_errors[2], msg=message)
        return results, orders

    def test_tm111_pec_cavity_converges_with_one_fine_patch(self):
        self._assert_problem("TM111 PEC cavity")

    def test_periodic_plane_wave_converges_with_one_fine_patch(self):
        self._assert_problem("periodic")


if __name__ == "__main__":
    unittest.main()
