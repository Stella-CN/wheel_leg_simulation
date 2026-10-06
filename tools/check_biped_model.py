"""编译并检查左右双腿机身模型。"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import math

import mujoco
import numpy as np

from wheel_leg.build_model import mass_properties_for_geometry
from wheel_leg.paths import GENERATED_MODEL_DIR
from wheel_leg.simulate import load_model


SIDES = ("left", "right")


def main():
    model_path = GENERATED_MODEL_DIR / "biped_wheel_leg.xml"
    m, d, g = load_model(model_path)
    assert (m.nq, m.nv, m.nu, m.neq) == (10, 10, 6, 2)
    assert m.nmesh == 2

    chassis = m.body("chassis").id
    chassis_geom = m.geom("chassis_box").id
    assert np.allclose(m.geom_size[chassis_geom], (0.1, 0.075, 0.05))
    assert m.geom_bodyid[chassis_geom] == chassis
    assert m.body("hip_mount_left").parentid == chassis
    assert m.body("hip_mount_right").parentid == chassis
    chassis_half_width = float(m.geom_size[chassis_geom][1])
    expected_axis_offset = chassis_half_width

    for side, sign in (("left", -1.0), ("right", 1.0)):
        hip_mount = m.body(f"hip_mount_{side}").id
        knee_motor = m.body(f"knee_motor_{side}").id
        linkage_base = m.body(f"linkage_base_{side}").id
        active_arm = m.body(f"active_arm_OA_{side}").id
        output_link = m.body(f"output_link_BCW_{side}").id
        wheel = m.body(f"wheel_{side}").id
        assert m.body(f"leg_carrier_{side}").parentid == hip_mount
        assert knee_motor == m.body(f"knee_motor_{side}").id
        assert m.body(f"linkage_base_{side}").parentid == knee_motor
        assert m.body(f"active_arm_OA_{side}").parentid == linkage_base
        assert m.body(f"output_link_BCW_{side}").parentid == linkage_base
        assert m.body(f"wheel_{side}").parentid == output_link

        # Hip stator is fixed inside the chassis; knee stator sits outside.
        assert np.isclose(d.xpos[hip_mount][1], sign * expected_axis_offset)
        stator_geom = m.geom(f"hip_motor_stator_{side}").id
        assert m.geom_bodyid[stator_geom] == hip_mount
        assert m.body_jntnum[hip_mount] == 0
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"mount_{side}") == -1

        stator_center_y = float(d.geom_xpos[stator_geom][1])
        stator_half_width = float(m.geom_size[stator_geom][1])
        assert 0 < sign*stator_center_y-stator_half_width
        assert sign*stator_center_y+stator_half_width <= chassis_half_width+1e-12
        knee_geom = m.geom(f'knee_motor_stator_{side}').id
        assert sign*d.geom_xpos[knee_geom,1]-m.geom_size[knee_geom,1] >= chassis_half_width
        assert np.isclose(d.site(f'W_{side}').xpos[1], sign*.137)

    max_gap, max_fk = 0.0, 0.0
    for length in np.linspace(g.length * 1.13333333333,
                              g.length * 1.66666666667, 21):
        for beta in np.linspace(math.radians(-20), math.radians(20), 21):
            q = g.inverse(float(length), float(beta))
            for side, sign in (("left", -1.0), ("right", 1.0)):
                for name, value in q.items():
                    d.qpos[int(m.joint(f"{name}_{side}").qposadr[0])] = value
            mujoco.mj_forward(m, d)
            for side in SIDES:
                ac = m.site(f"C_AC_{side}").id
                bc = m.site(f"C_BC_{side}").id
                o = m.site(f"O_{side}").id
                w = m.site(f"W_{side}").id
                max_gap = max(max_gap, float(np.linalg.norm(
                    d.site_xpos[ac] - d.site_xpos[bc])))
                actual = d.site_xpos[w] - d.site_xpos[o]
                wanted = np.array([
                    length * math.sin(beta), actual[1],
                    -length * math.cos(beta)])
                max_fk = max(max_fk, float(np.linalg.norm(actual - wanted)))

    assert max_gap < 1e-9, f"双腿闭环误差过大：{max_gap} m"
    assert max_fk < 1e-9, f"双腿轮心正运动学误差过大：{max_fk} m"

    constraint_jacobians = []
    for side in SIDES:
        ac = m.site(f"C_AC_{side}").id
        bc = m.site(f"C_BC_{side}").id
        jac_ac = np.zeros((3, m.nv))
        jac_bc = np.zeros((3, m.nv))
        rotational = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jac_ac, rotational, ac)
        mujoco.mj_jacSite(m, d, jac_bc, rotational, bc)
        constraint_jacobians.append(jac_ac - jac_bc)
    rank = int(np.linalg.matrix_rank(np.vstack(constraint_jacobians), tol=1e-9))
    assert rank == 4, f"双腿平面闭环预期约束秩为4，实际{rank}"

    modeled_mass = float(m.body_subtreemass[chassis])
    masses = mass_properties_for_geometry(g)
    expected_single_mass = sum((masses.thigh, masses.shank, masses.crank,
                                masses.coupler, masses.flange,
                                masses.shank_offset_arm, masses.shank_link_arm,
                                masses.hip_motor_stator, masses.hip_motor_rotor,
                                masses.knee_motor_stator, masses.knee_motor_rotor,
                                masses.wheel_tire, masses.wheel_rim,
                                masses.hub_motor_stator, masses.hub_motor_rotor))
    assert abs(modeled_mass - 2 * expected_single_mass) < 1e-9
    print(f"MuJoCo {mujoco.__version__}; nq={m.nq}; nv={m.nv}; "
          f"nu={m.nu}; neq={m.neq}")
    print(f"chassis = 200 x 150 x 100 mm; hip-axis offset = "
          f"{expected_axis_offset * 1000:.1f} mm; modeled mass = "
          f"{modeled_mass:.3f} kg")
    print(f"441 dual-leg configurations; max closure = {max_gap:.3e} m; "
          f"max FK error = {max_fk:.3e} m")
    print(f"effective closure rank={rank}; independent DOF={m.nv-rank}")
    print("BIPED MODEL CHECK PASSED; this is NOT a structural/load-capacity test.")


if __name__ == "__main__":
    main()
