"""Regression test for the in-place yaw maneuver with leg extension."""
import argparse
from pathlib import Path
import tempfile
import unittest

from wheel_leg.paths import GENERATED_MODEL_DIR
from wheel_leg.simulate_spin_extension import SpinExtensionExperiment


class SpinExtensionTests(unittest.TestCase):
    def test_spin_extension_and_recovery(self):
        with tempfile.TemporaryDirectory(prefix="wheel-spin-extension-") as tmp:
            args = argparse.Namespace(
                model=GENERATED_MODEL_DIR / "ground_robot.xml",
                output=Path(tmp), duration=11.0, drop_height=.01,
                pitch_deg=3.0, roll_deg=0.0, leg_amplitude_mm=10.0,
                leg_period=6.0, motion_start=2.0, target_yaw_deg=90.0,
                spin_start=3.0, spin_duration=3.0, post_spin_settle=3.0,
                horizontal_position_weight_x=300.0,
                horizontal_position_weight_y=0.0,
                horizontal_velocity_weight_x=10.0,
                horizontal_velocity_weight_y=0.0,
                station_keeping_position_gain=2.0,
                station_keeping_velocity_gain=1.0,
                yaw_tolerance_deg=10.0, max_planar_drift_mm=80.0,
            )
            experiment = SpinExtensionExperiment(args)
            while experiment.d.time < args.duration and experiment.step():
                pass
            report = experiment.save(experiment.failure or "completed",
                                     verbose=False)
            self.assertTrue(report["spin_extension_passed"], report)
            self.assertAlmostEqual(report["spin_end_yaw_deg"], 90.0, delta=10.0)
            self.assertGreater(report["max_yaw_rate_deg_s"], 45.0)
            self.assertLess(report["max_planar_displacement_mm"], 80.0)
            self.assertTrue(report["extension_passed"])
            self.assertLess(report["max_leg_tracking_error_mm"], 10.0)
            self.assertEqual(report["max_nonwheel_ground_contacts"], 0)
            self.assertEqual(report["mujoco_warnings"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
