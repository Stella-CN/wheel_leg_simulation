"""Check concentric motor dimensions, flush faces, mirroring and attachments."""
import unittest

import mujoco
import numpy as np

from wheel_leg.build_biped_model import make_biped_xml
from wheel_leg.geometry import Geometry


class MotorVisualTests(unittest.TestCase):
    def setUp(self):
        self.m = mujoco.MjModel.from_xml_string(make_biped_xml(Geometry()))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

    def test_half_radius_height_and_flush_faces(self):
        m,d = self.m,self.d
        for side, sign in (('left',-1),('right',1)):
            for outer,inner,radius,height in (
                ('hip_stator_shell','hip_rotor_cap',.0285,.046),
                ('knee_stator_shell','knee_rotor_cap',.0285,.046),
                ('hub_rotor_cap','hub_stator_shell',.034,.0445),
            ):
                core = m.geom(f'{inner}_{side}').id
                back = m.geom(f'{outer}_back_{side}').id
                ring = m.geom(f'{outer}_{side}').id
                mesh = m.geom_dataid[ring]
                start, count = m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                vertices = m.mesh_vert[start:start+count] @ d.geom_xmat[ring].reshape(3,3).T + d.geom_xpos[ring]
                front = np.max(sign*vertices[:,1])
                rear = sign*d.geom_xpos[back,1]-m.geom_size[back,1]
                self.assertAlmostEqual(front-rear,height,delta=1e-7)
                self.assertAlmostEqual(m.geom_size[core,0],radius/2)
                self.assertAlmostEqual(2*m.geom_size[core,1],height/2)
                self.assertAlmostEqual(sign*d.geom_xpos[core,1]+m.geom_size[core,1],front,delta=1e-7)
                np.testing.assert_allclose(d.geom_xpos[core,[0,2]], d.geom_xpos[back,[0,2]],atol=1e-9)

    def test_knee_rotor_drives_OA_stator_carries_OB(self):
        m,d = self.m,self.d
        for side in ('left','right'):
            oa = m.geom(f'active_arm_OA_{side}').id
            ob = m.geom(f'fixed_arm_OB_{side}').id
            rotor = m.geom(f'knee_rotor_cap_{side}').id
            stator = m.geom(f'knee_stator_shell_{side}').id
            self.assertEqual(m.geom_bodyid[oa],m.geom_bodyid[rotor])
            self.assertEqual(m.geom(f'knee_output_shaft_{side}').bodyid,m.geom_bodyid[rotor])
            self.assertEqual(m.geom(f'knee_stator_OB_bracket_{side}').bodyid,m.geom_bodyid[ob])
            self.assertEqual(m.geom(f'fixed_arm_OB_visual_{side}').bodyid,m.geom_bodyid[ob])
            self.assertEqual(m.body(m.geom_bodyid[ob]).parentid,m.geom_bodyid[stator])
            before = d.geom_xmat[[oa,ob,rotor,stator]].copy()
            d.qpos[m.joint(f'knee_drive_{side}').qposadr[0]] += .2
            mujoco.mj_forward(m,d)
            after = d.geom_xmat[[oa,ob,rotor,stator]]
            self.assertFalse(np.allclose(before[0],after[0]))
            self.assertFalse(np.allclose(before[2],after[2]))
            np.testing.assert_allclose(before[[1,3]],after[[1,3]],atol=1e-12)


if __name__ == '__main__':
    unittest.main()
