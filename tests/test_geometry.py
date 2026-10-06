"""纯 Python 数值/XML 校验；不替代 MuJoCo 编译和动力学测试。"""
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from wheel_leg.build_model import make_xml, mass_properties_for_geometry
from wheel_leg.geometry import Geometry
from wheel_leg.motor_specs import DM_H6215, DM_J4310_2EC_V11
from wheel_leg.paths import GENERATED_MODEL_DIR


class GeometryTests(unittest.TestCase):
    def test_grid(self):
        g = Geometry()
        for i in range(41):
            length = .17 + .08 * i / 40
            for j in range(41):
                beta = math.radians(-20 + j)
                points = g.points(g.inverse(length, beta))
                self.assertLess(math.dist(points["C_AC"], points["C_BC"]), 1e-12)
                self.assertAlmostEqual(points["W"][0], length * math.sin(beta), places=12)
                self.assertAlmostEqual(points["W"][2] - g.hip_height,
                                       -length * math.cos(beta), places=12)

    def test_initial_body_transforms(self):
        """独立解析 MJCF 初始 body euler/ref，检查 O-A-C-B-W 闭合。"""
        root = ET.fromstring(make_xml(Geometry()))
        sites = {}

        def visit(body, pos, angle):
            p = tuple(map(float, body.get("pos", "0 0 0").split()))
            c, s = math.cos(angle), math.sin(angle)
            rotp = (c*p[0] + s*p[2], p[1], -s*p[0] + c*p[2])
            pos = tuple(a + b for a, b in zip(pos, rotp))
            angle += float(body.get("euler", "0 0 0").split()[1])
            for site in body.findall("site"):
                p = tuple(map(float, site.get("pos", "0 0 0").split()))
                c, s = math.cos(angle), math.sin(angle)
                rp = (c*p[0] + s*p[2], p[1], -s*p[0] + c*p[2])
                sites[site.get("name")] = tuple(a + b for a, b in zip(pos, rp))
            for child in body.findall("body"):
                visit(child, pos, angle)

        visit(root.find("worldbody/body"), (0, 0, 0), 0)
        refs = {j.get("name"): float(j.get("ref", "0"))
                for j in root.findall(".//joint") if j.get("name")}
        expected = Geometry().inverse(Geometry().nominal_length, 0)
        for name in expected:
            self.assertAlmostEqual(refs[name], expected[name], delta=1e-10)

        points = Geometry().points(expected)
        for name in ("A", "B", "C_AC", "C_BC", "W"):
            self.assertAlmostEqual(sites[name][0], points[name][0], delta=1e-10)
            self.assertAlmostEqual(sites[name][2], points[name][2], delta=1e-10)
            self.assertAlmostEqual(sites[name][1], 0.062, delta=1e-10)
        self.assertAlmostEqual(sites["O"][0], 0, delta=1e-10)
        self.assertAlmostEqual(sites["O"][2], .36, delta=1e-10)

    def test_model_topology(self):
        root = ET.fromstring(make_xml(Geometry()))
        self.assertEqual(len(root.findall(".//body/joint")), 5)
        self.assertEqual(len(root.findall("actuator/motor")), 3)
        self.assertIsNone(root.find(".//freejoint"))
        equality = root.find("equality/connect")
        self.assertEqual(equality.get("site1"), "C_AC")
        self.assertEqual(equality.get("site2"), "C_BC")
        self.assertIsNotNone(root.find(
            ".//body[@name='leg_carrier']/body[@name='knee_motor']"))
        self.assertIsNotNone(root.find(
            ".//body[@name='knee_motor']/body[@name='linkage_base']"))
        self.assertIsNotNone(root.find(
            ".//body[@name='linkage_base']/body[@name='active_arm_OA']"))
        self.assertIsNotNone(root.find(
            ".//body[@name='linkage_base']/body[@name='output_link_BCW']"))
        self.assertIsNotNone(root.find(
            ".//body[@name='output_link_BCW']/body[@name='wheel']"))
        self.assertEqual((GENERATED_MODEL_DIR / "wheel_leg.xml").read_text(),
                         make_xml(Geometry()))

    def test_mass_breakdown(self):
        root = ET.fromstring(make_xml(Geometry()))

        def mass(name):
            return float(root.find(f".//geom[@name='{name}']").get("mass"))

        self.assertAlmostEqual(mass("hip_motor_stator"), .150)
        self.assertAlmostEqual(mass("hip_motor_rotor"), .150)
        self.assertAlmostEqual(mass("knee_motor_stator"), .150)
        self.assertAlmostEqual(mass("knee_motor_rotor"), .150)
        self.assertAlmostEqual(mass("hub_motor_stator"), .180)
        self.assertAlmostEqual(mass("hub_motor_rotor"), .180)

    def test_shortened_link_mass_scaling(self):
        g = Geometry()
        masses = mass_properties_for_geometry(g)
        self.assertAlmostEqual(masses.thigh, .120 * .130 / .150)
        self.assertAlmostEqual(masses.coupler, .050 * .130 / .150)
        self.assertAlmostEqual(masses.crank, .040)
        self.assertAlmostEqual(masses.shank, .160 * (.130 + .035) / (.150 + .035))

    def test_motor_reference_dimensions(self):
        root = ET.fromstring(make_xml(Geometry()))

        def size(name):
            values = root.find(f".//geom[@name='{name}']").get("size").split()
            return tuple(float(value) for value in values[:2])

        self.assertEqual(size("hip_motor_stator"),
                         (DM_J4310_2EC_V11.outer_diameter_m / 2,
                          DM_J4310_2EC_V11.body_length_m / 2))
        self.assertEqual(size("knee_motor_stator"),
                         (DM_J4310_2EC_V11.outer_diameter_m / 2,
                          DM_J4310_2EC_V11.body_length_m / 2))
        self.assertEqual(size("hub_motor_stator"),
                         (DM_H6215.outer_diameter_m / 2,
                          DM_H6215.body_length_m / 2))
        self.assertEqual(root.find(".//actuator/motor[@name='hip_motor']")
                         .get("ctrlrange"), "-7.0 7.0")
        self.assertEqual(root.find(".//actuator/motor[@name='knee_motor']")
                         .get("ctrlrange"), "-7.0 7.0")
        self.assertEqual(root.find(".//actuator/motor[@name='wheel_motor']")
                         .get("ctrlrange"),
                         f"{-DM_H6215.peak_torque_nm:g} {DM_H6215.peak_torque_nm:g}")
        self.assertEqual(len(root.findall("asset/mesh")), 2)
        for mesh in root.findall("asset/mesh"):
            self.assertIsNone(mesh.get('file'))
            self.assertIsNotNone(mesh.get('vertex'))
        self.assertIsNone(root.find(".//geom[@name='mount']"))

    def test_unreachable(self):
        with self.assertRaises(ValueError):
            Geometry().inverse(2 * Geometry().length, 0)
        with self.assertRaises(ValueError):
            Geometry().inverse(2 * Geometry().length + .001, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
