"""从无动力躺地姿态启动并自扶正的动力学回归测试。"""
import argparse
from pathlib import Path
import tempfile
import unittest

from wheel_leg.paths import GENERATED_MODEL_DIR
from wheel_leg.simulate_balance import BalanceExperiment


class SelfRightingTests(unittest.TestCase):
    def test_unpowered_lie_then_balance(self):
        with tempfile.TemporaryDirectory(prefix="wheel-self-right-") as tmp:
            args = argparse.Namespace(
                model=GENERATED_MODEL_DIR / "ground_robot.xml", output=Path(tmp),
                duration=8., drop_height=.01, pitch_deg=3., roll_deg=0.,
                push_n=0., push_start=5., push_duration=.2,
                disable_feedback=False, motion="hold", leg_amplitude_mm=30.,
                leg_period=8., motion_start=2., startup="self-right",
                lying_pitch_deg=90., laying_idle=1., righting_timeout=4.,
                righting_torque=7.,
            )
            experiment = BalanceExperiment(args)
            self.assertEqual(experiment.self_righting.phase, "idle")
            for _ in range(200):
                self.assertTrue(experiment.step())
                self.assertTrue((abs(experiment.d.ctrl) < 1e-12).all())
            self.assertEqual(experiment.self_righting.phase, "idle")
            while experiment.d.time < args.duration and experiment.step():
                pass
            report = experiment.save(experiment.failure or "completed", verbose=False)
            self.assertTrue(report["self_righting_passed"], report)
            self.assertTrue(report["balanced"], report)
            self.assertLess(report["righting_transition_s"], 3.)
            self.assertGreater(report["max_startup_nonwheel_ground_contacts"], 0)
            self.assertEqual(report["max_nonwheel_ground_contacts"], 0)
            self.assertEqual(report["mujoco_warnings"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
