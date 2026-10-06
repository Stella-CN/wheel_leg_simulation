"""在当前 .venv 中实际调用 MuJoCo 编译器，检查拓扑、闭环和坐标。"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import math

import mujoco
import numpy as np

from wheel_leg.motor_specs import DM_H6215, DM_J4310_2EC_V11
from wheel_leg.build_model import mass_properties_for_geometry
from wheel_leg.paths import GENERATED_MODEL_DIR
from wheel_leg.simulate import load_model


def main():
    m, d, g = load_model(GENERATED_MODEL_DIR / "wheel_leg.xml")
    assert (m.nq, m.nv, m.nu, m.neq) == (5, 5, 3, 1), "模型自由度/执行器/约束数量错误"
    assert m.nmesh == 2, "预期两个内嵌程序化环壳，无外部 CAD 文件"

    site_names = ("O", "A", "B", "C_AC", "C_BC", "W")
    ids = {name: m.site(name).id for name in site_names}
    joint_names = ("hip", "knee_drive", "bearing_A", "bearing_B")
    adr = {name: int(m.joint(name).qposadr[0]) for name in joint_names}

    # body 树必须表达：髋输出带动膝定子，膝定子承载整个腿部连杆。
    body_ids = {name: m.body(name).id for name in (
        "hip_mount", "leg_carrier", "knee_motor", "linkage_base",
        "active_arm_OA", "coupler_AC", "output_link_BCW", "wheel")}
    assert m.body("leg_carrier").parentid == body_ids["hip_mount"]
    assert m.body("knee_motor").parentid == body_ids["leg_carrier"]
    assert m.body("linkage_base").parentid == body_ids["knee_motor"]
    assert m.body("active_arm_OA").parentid == body_ids["linkage_base"]
    assert m.body("coupler_AC").parentid == body_ids["active_arm_OA"]
    assert m.body("output_link_BCW").parentid == body_ids["linkage_base"]
    assert m.body("wheel").parentid == body_ids["output_link_BCW"]
    assert m.joint("hip").bodyid == body_ids["leg_carrier"]
    assert m.joint("knee_drive").bodyid == body_ids["active_arm_OA"]
    assert m.joint("bearing_A").bodyid == body_ids["coupler_AC"]
    assert m.joint("bearing_B").bodyid == body_ids["output_link_BCW"]

    # 髋/膝轴心同 X/Z 坐标，髋定子在内侧、膝定子在外侧；连杆平面
    # 位于膝电机外侧并且 BC/BW 输出杆只有一个刚体 body。
    hip_axis = d.xpos[body_ids["leg_carrier"]]
    knee_axis = d.xpos[body_ids["knee_motor"]]
    assert np.allclose(hip_axis[[0, 2]], knee_axis[[0, 2]], atol=1e-12)
    assert knee_axis[1] > hip_axis[1]
    assert d.site_xpos[ids["C_AC"]][1] > knee_axis[1]
    assert np.allclose(d.site_xpos[ids["C_AC"]][1], d.site_xpos[ids["C_BC"]][1])
    assert np.allclose(d.site_xpos[ids["C_AC"]][1], d.site_xpos[ids["W"]][1])
    assert m.joint("hip").axis[1] == m.joint("knee_drive").axis[1]
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mount") == -1
    assert m.geom("hip_motor_stator").bodyid == body_ids["hip_mount"]
    for stator, rotor, owner in (
        ("hip_stator_shell", "hip_rotor_cap", "leg_carrier"),
        ("knee_stator_shell", "knee_rotor_cap", "active_arm_OA"),
        ("hub_stator_shell", "hub_rotor_cap", "wheel"),
    ):
        assert m.geom(rotor).bodyid == body_ids[owner]
        assert not np.allclose(m.geom(stator).rgba, m.geom(rotor).rgba)

    assert np.allclose(
        m.actuator("hip_motor").ctrlrange,
        (-DM_J4310_2EC_V11.peak_torque_nm, DM_J4310_2EC_V11.peak_torque_nm))
    assert np.allclose(
        m.actuator("knee_motor").ctrlrange,
        (-DM_J4310_2EC_V11.peak_torque_nm, DM_J4310_2EC_V11.peak_torque_nm))
    assert np.allclose(
        m.actuator("wheel_motor").ctrlrange,
        (-DM_H6215.peak_torque_nm, DM_H6215.peak_torque_nm))
    modeled_mass = float(m.body_subtreemass[body_ids["hip_mount"]])
    masses = mass_properties_for_geometry(g)
    expected_mass = sum((masses.thigh, masses.shank, masses.crank,
                         masses.coupler, masses.flange, masses.shank_offset_arm,
                         masses.shank_link_arm, masses.hip_motor_stator,
                         masses.hip_motor_rotor, masses.knee_motor_stator,
                         masses.knee_motor_rotor, masses.wheel_tire,
                         masses.wheel_rim, masses.hub_motor_stator,
                         masses.hub_motor_rotor))
    assert abs(modeled_mass - expected_mass) < 1e-9, (
        f"默认机构质量分配错误：{modeled_mass} kg")
    motor_geometry = m.numeric("motor_geometry").data
    assert np.allclose(motor_geometry, (
        DM_J4310_2EC_V11.outer_diameter_m,
        DM_J4310_2EC_V11.body_length_m,
        DM_H6215.outer_diameter_m,
        DM_H6215.body_length_m))

    max_gap, max_fk = 0.0, 0.0
    for length in np.linspace(g.length * 1.13333333333,
                              g.length * 1.66666666667, 41):
        for beta in np.linspace(math.radians(-20), math.radians(20), 41):
            for name, value in g.inverse(float(length), float(beta)).items():
                d.qpos[adr[name]] = value
            mujoco.mj_forward(m, d)
            gap = np.linalg.norm(
                d.site_xpos[ids["C_AC"]] - d.site_xpos[ids["C_BC"]])
            actual = d.site_xpos[ids["W"]] - d.site_xpos[ids["O"]]
            wanted = np.array([
                length * math.sin(beta),
                actual[1],
                -length * math.cos(beta)])
            max_gap = max(max_gap, float(gap))
            max_fk = max(max_fk, float(np.linalg.norm(actual - wanted)))
    assert max_gap < 1e-9, f"C 点闭环几何错误：{max_gap} m"
    assert max_fk < 1e-9, f"关节角/轮心坐标映射错误：{max_fk} m"

    jp1, jp2, jr = (np.zeros((3, m.nv)) for _ in range(3))
    mujoco.mj_jacSite(m, d, jp1, jr, ids["C_AC"])
    mujoco.mj_jacSite(m, d, jp2, jr, ids["C_BC"])
    rank = int(np.linalg.matrix_rank(jp1 - jp2, tol=1e-9))
    assert rank == 2, f"平面闭环预期有效约束秩为2，实际{rank}"

    print(f"MuJoCo {mujoco.__version__}; 1681 configurations checked")
    print(f"max C closure = {max_gap:.3e} m; max FK error = {max_fk:.3e} m")
    print(f"nv={m.nv}; effective closure rank={rank}; independent DOF={m.nv-rank}")
    print(f"modeled mechanism mass = {modeled_mass:.3f} kg (procedural motor visuals)")
    print("topology: fixed OB / active OA / passive AC + BC/BW rigid output")
    print(f"hip/knee motor={DM_J4310_2EC_V11.name}; peak="
          f"{DM_J4310_2EC_V11.peak_torque_nm:g} N m")
    print(f"wheel motor={DM_H6215.name}; mass={DM_H6215.mass_kg:g} kg (user supplied); "
          f"rated={DM_H6215.rated_torque_nm:g} N m; peak={DM_H6215.peak_torque_nm:g} N m")
    print("GEOMETRY CHECK PASSED; this is NOT a dynamics/load-capacity test.")


if __name__ == "__main__":
    main()
