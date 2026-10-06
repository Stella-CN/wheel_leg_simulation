"""Synchronous length tracking with real contact and balance disturbances."""
import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wheel_leg.balance_control import ExtensionTrajectory
from wheel_leg.geometry import Geometry
from wheel_leg.paths import GENERATED_MODEL_DIR
from wheel_leg.simulate_balance import BalanceExperiment


class ExtensionTests(unittest.TestCase):
    def test_smooth_start_and_bounds(self):
        nominal = Geometry().nominal_length
        trajectory = ExtensionTrajectory(nominal)
        self.assertEqual(trajectory.sample(0), (nominal,0.))
        self.assertEqual(trajectory.sample(2), (nominal,0.))
        for time in np.linspace(0,20,2001):
            length,speed = trajectory.sample(time)
            self.assertGreaterEqual(length,nominal-.03-1e-12)
            self.assertLessEqual(length,nominal+.03+1e-12)
            if time > .001:
                h=1e-5
                finite_difference=(trajectory.sample(time+h)[0]-trajectory.sample(time-h)[0])/(2*h)
                self.assertAlmostEqual(speed,finite_difference,delta=1e-8)

    def test_invalid_trajectory(self):
        for kwargs in ({'period':0}, {'period':3}, {'start':1},
                       {'amplitude':-1}, {'amplitude':float('nan')}):
            with self.assertRaises(ValueError):
                ExtensionTrajectory(.212,**kwargs)

    def run_motion(self, period, push, roll):
        with tempfile.TemporaryDirectory(prefix='wheel-extension-') as tmp:
            args=argparse.Namespace(model=GENERATED_MODEL_DIR/'ground_robot.xml',output=Path(tmp),
                duration=16.,drop_height=.01,pitch_deg=3.,roll_deg=roll,
                push_n=push,push_start=7.,push_duration=.2,disable_feedback=False,
                motion='extend',leg_amplitude_mm=30.,leg_period=period,motion_start=2.)
            experiment=BalanceExperiment(args)
            while experiment.d.time<args.duration and experiment.step():
                pass
            report=experiment.save(experiment.failure or 'completed',verbose=False)
            self.assertTrue(report['motion_passed'], report)
            self.assertLess(report['max_leg_tracking_error_mm'],3.)
            self.assertLess(report['max_closure_mm'],1.)
            self.assertEqual(report['motion_both_wheels_contact_fraction'],1.)
            self.assertEqual(report['mujoco_warnings'],{})
            self.assertTrue(all(v==0 for v in report['saturation_fractions']))
            for row in experiment.rows:
                if row['time_s']>4:
                    self.assertLess(abs(row['left_leg_length_m']-row['right_leg_length_m']),.003)

    def test_full_cycle_and_push(self):
        self.run_motion(period=8.,push=5.,roll=0.)

    def test_faster_cycle_reverse_push_and_roll(self):
        self.run_motion(period=4.,push=-5.,roll=2.)


if __name__=='__main__':
    unittest.main()
