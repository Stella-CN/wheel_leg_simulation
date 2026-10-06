"""Ground support, physical bookkeeping and closed-loop/negative-control tests."""
import argparse
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wheel_leg.build_ground_model import make_ground_xml
from wheel_leg.paths import GENERATED_MODEL_DIR
from wheel_leg.simulate_balance import BalanceExperiment


class GroundBalanceTests(unittest.TestCase):
    def test_mass_and_floating_topology(self):
        xml = make_ground_xml()
        root = ET.fromstring(xml)
        self.assertEqual((GENERATED_MODEL_DIR/'ground_robot.xml').read_text(), xml)
        m = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual((m.nq,m.nv,m.nu,m.neq,m.nmesh), (17,16,6,2,2))
        self.assertEqual(m.joint('base').type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertTrue(np.all(m.eq_type == mujoco.mjtEq.mjEQ_CONNECT))
        self.assertTrue(np.allclose(m.opt.gravity, [0,0,-9.81]))
        self.assertAlmostEqual(float(m.body('chassis').mass[0]), 1.2827328)
        np.testing.assert_allclose(m.geom('chassis_box').rgba, [.28, .32, .38, .35])
        for side in ('left','right'):
            self.assertEqual(m.body(f'hip_mount_{side}').parentid, m.body('chassis').id)
            self.assertEqual(m.body(f'hip_mount_{side}').jntnum[0], 0)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            hip = m.geom(f'hip_motor_stator_{side}').id
            knee = m.geom(f'knee_motor_stator_{side}').id
            self.assertLessEqual(abs(d.geom_xpos[hip,1])+m.geom_size[hip,1], .073+1e-12)
            self.assertGreaterEqual(abs(d.geom_xpos[knee,1])-m.geom_size[knee,1], .077-1e-12)
            self.assertAlmostEqual(abs(d.site(f'W_{side}').xpos[1]),.137)
            for name,mass in (('hip',.3),('knee',.3),('hub',.36)):
                # Geom masses are compiler inputs, check source definitions.
                values = [float(root.find(f".//geom[@name='{name}_motor_{part}_{side}']").get('mass'))
                          for part in ('stator','rotor')]
                self.assertAlmostEqual(sum(values),mass)
        self.assertTrue(np.all(m.body_inertia[1:] > 0))

    def run_scenario(self, *, pitch=3, roll=0, drop=.01, push=3, feedback=True):
        with tempfile.TemporaryDirectory(prefix='wheel-balance-') as tmp:
            args = argparse.Namespace(model=GENERATED_MODEL_DIR/'ground_robot.xml', output=Path(tmp),
                duration=8., drop_height=drop, pitch_deg=pitch, roll_deg=roll,
                push_n=push, push_start=2., push_duration=.2, disable_feedback=not feedback)
            experiment=BalanceExperiment(args)
            while experiment.d.time<args.duration and experiment.step():
                pass
            report=experiment.save(experiment.failure or 'completed', verbose=False)
            self.assertEqual(report['mujoco_warnings'], {})
            if feedback:
                self.assertTrue(report['balanced'])
                self.assertEqual(report['max_nonwheel_ground_contacts'],0)
                self.assertLess(report['max_closure_mm'],1.)
                self.assertAlmostEqual(report['tail_mean_floor_normal_n'],
                                       report['total_mass_kg']*9.81, delta=.1)
                self.assertTrue(all(f==0 for f in report['saturation_fractions']))
            else:
                self.assertFalse(report['balanced'])
                self.assertEqual(report['status'],'fallen')

    def test_landing_and_push(self):
        self.run_scenario()

    def test_roll_pitch_and_reverse_push(self):
        self.run_scenario(pitch=-5,roll=2,push=-5)

    def test_higher_drop(self):
        self.run_scenario(pitch=5,drop=.03,push=5)

    def test_feedback_is_required(self):
        self.run_scenario(push=0,feedback=False)


if __name__=='__main__':
    unittest.main()
