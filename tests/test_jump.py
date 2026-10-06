"""Regression test for the motor-driven hop and landing recovery."""
import argparse
from pathlib import Path
import tempfile
import unittest

from wheel_leg.paths import GENERATED_MODEL_DIR
from wheel_leg.simulate_jump import JumpExperiment


class JumpTests(unittest.TestCase):
    def test_jump_and_recover(self):
        with tempfile.TemporaryDirectory(prefix="wheel-jump-") as tmp:
            args = argparse.Namespace(
                model=GENERATED_MODEL_DIR / "ground_robot.xml",
                output=Path(tmp), duration=5.0, drop_height=.01,
                pitch_deg=3.0, roll_deg=0.0, balance_duration=1.0,
                squat_duration=1.0, thrust_duration=.25,
                squat_length_mm=145.0, thrust_torque=4.5,
                hip_thrust_torque=0.0, thrust_ramp=.02,
                contact_debounce=.03, landing_settle=1.0,
                min_jump_height_mm=80.0, max_flight_duration=1.5,
            )
            experiment = JumpExperiment(args)
            while experiment.d.time < args.duration and experiment.step():
                pass
            report = experiment.save(experiment.failure or "completed",
                                     verbose=False)
            self.assertTrue(report["jump_passed"], report)
            self.assertGreater(report["jump_height_m"], .08)
            self.assertLess(report["takeoff_time_s"], report["landing_time_s"])
            self.assertGreater(report["flight_duration_s"], .05)
            self.assertEqual(report["max_flight_nonwheel_ground_contacts"], 0)
            self.assertLess(report["max_knee_limit_excess_rad"], .01)
            self.assertEqual(report["mujoco_warnings"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
