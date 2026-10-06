"""生成同轴髋/膝电机平行四边形轮腿的 MJCF。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from .geometry import Geometry
from .motor_specs import DM_H6215, DM_J4310_2EC_V11
from .motor_visuals import add_ring_mesh
from .paths import GENERATED_MODEL_DIR


COAXIAL_MOTOR_CENTER_OFFSET_M = 0.025
HIP_STATOR_INNER_FACE_OFFSET_M = (
    COAXIAL_MOTOR_CENTER_OFFSET_M + DM_J4310_2EC_V11.body_length_m / 2.0)


@dataclass(frozen=True)
class MassProperties:
    """机构的可替换质量参数，单位 kg。

    当前只对已知/明确指定的零件分配质量。两个 DM-J4310-2EC 各为
    300 g，DM-H6215 为 360 g；每台电机按定子/转子 50/50 拆分，
    使质量归属与 body 树一致。自绘外观部件本身不计质量。

    为兼容已有参数名，``thigh`` 表示固定臂 OB，``shank`` 表示
    刚性输出杆 BC/BW，``crank`` 表示主动臂 OA。
    """

    # These are reference estimates at OB=AC=BW=150 mm and OA=BC=35 mm.
    # make_xml scales link masses with their modeled lengths when no explicit
    # MassProperties object is supplied.
    thigh: float = 0.120             # 固定臂 OB
    shank: float = 0.160             # 刚性输出杆 BC/BW
    crank: float = 0.040             # 主动臂 OA
    coupler: float = 0.050           # 被动连杆 AC
    flange: float = 0.060            # 髋-膝法兰盘
    shank_offset_arm: float = 0.012  # 输出杆 B 处轴承座/加强座
    shank_link_arm: float = 0.015     # 输出杆到轮毂的轮轴支架
    hip_motor_stator: float = DM_J4310_2EC_V11.mass_kg / 2.0
    hip_motor_rotor: float = DM_J4310_2EC_V11.mass_kg / 2.0
    knee_motor_stator: float = DM_J4310_2EC_V11.mass_kg / 2.0
    knee_motor_rotor: float = DM_J4310_2EC_V11.mass_kg / 2.0
    wheel_tire: float = 0.260
    wheel_rim: float = 0.140
    hub_motor_stator: float = DM_H6215.mass_kg / 2.0
    hub_motor_rotor: float = DM_H6215.mass_kg / 2.0


DEFAULT_MASSES = MassProperties()


def mass_properties_for_geometry(g: Geometry) -> MassProperties:
    """Scale estimated link masses while keeping cross-sections unchanged."""
    reference_output_length = 0.150 + 0.035
    modeled_output_length = g.length + g.crank
    return replace(
        DEFAULT_MASSES,
        thigh=DEFAULT_MASSES.thigh * g.length / 0.150,
        shank=DEFAULT_MASSES.shank * modeled_output_length / reference_output_length,
        crank=DEFAULT_MASSES.crank * g.crank / 0.035,
        coupler=DEFAULT_MASSES.coupler * g.length / 0.150,
    )


def make_xml(g: Geometry, masses: MassProperties | None = None) -> str:
    if masses is None:
        masses = mass_properties_for_geometry(g)
    L, r, R, H = g.length, g.crank, g.wheel_radius, g.hip_height
    refs = g.inverse(g.nominal_length, 0.0)

    j4310_radius = DM_J4310_2EC_V11.outer_diameter_m / 2.0
    j4310_half_width = DM_J4310_2EC_V11.body_length_m / 2.0
    j4310_output_radius = DM_J4310_2EC_V11.output_flange_diameter_m / 2.0
    j4310_register_radius = DM_J4310_2EC_V11.output_register_diameter_m / 2.0
    h6215_radius = DM_H6215.outer_diameter_m / 2.0
    h6215_half_width = DM_H6215.body_length_m / 2.0
    h6215_output_radius = DM_H6215.output_flange_diameter_m / 2.0

    wheel_torque_limit = DM_H6215.peak_torque_nm
    if wheel_torque_limit is None or wheel_torque_limit <= 0:
        raise ValueError("DM-H6215 peak torque is required for the wheel actuator limit.")

    # +Y 为外侧：髋电机定子在内侧，膝电机定子在外侧；二者 X/Z
    # 坐标相同，只有轴向 Y 位置不同，因此是同一旋转轴心。
    hip_motor_y = -COAXIAL_MOTOR_CENTER_OFFSET_M
    hip_rotor_y = 0.002
    flange_y = 0.000
    flange_half_width = 0.004
    knee_motor_y = COAXIAL_MOTOR_CENTER_OFFSET_M
    linkage_y = 0.062
    motor_to_link = linkage_y - knee_motor_y
    wheel_motor_face_y = 0.024

    def q(name: str) -> str:
        return f"{refs[name]:.12g}"

    xml = f'''<mujoco model="coaxial_parallelogram_wheel_leg_bench">
  <compiler angle="radian" autolimits="true" inertiafromgeom="true"/>
  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast"
          solver="Newton" iterations="100" tolerance="1e-10"/>
  <size njmax="200" nconmax="100"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality numslices="96" numstacks="32"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7"/>
  </visual>
  <statistic center="-0.04 0 {H - L/2:.12g}" extent="0.55"/>
  <custom>
    <numeric name="geometry" data="{L} {r} {R} {H}"/>
    <numeric name="motor_geometry" data="{DM_J4310_2EC_V11.outer_diameter_m} {DM_J4310_2EC_V11.body_length_m} {DM_H6215.outer_diameter_m} {DM_H6215.body_length_m}"/>
  </custom>
  <default>
    <joint type="hinge" axis="0 -1 0" damping="0.008" armature="0.00005"/>
    <!-- 链杆碰撞关闭：这是运动/控制模型，不是三维干涉验证模型。 -->
    <geom contype="0" conaffinity="0" friction="0.8 0.005 0.0001" mass="0"/>
    <site size="0.004" rgba="1 0.9 0.2 1"/>
    <motor gear="1" ctrllimited="true"/>
  </default>
  <worldbody>
    <light pos="0 -1 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0.7 0.7 0.01"
          rgba="0.22 0.24 0.27 1" contype="1" conaffinity="1" mass="0"/>
    <site name="target_W" pos="0 {linkage_y} {H-g.nominal_length:.12g}"
          size="0.007" rgba="1 0.55 0.1 0.8"/>

    <!-- 固定髋电机定子；leg_carrier 是髋转子输出及其法兰连接的腿架。 -->
    <body name="hip_mount" pos="0 0 {H}">
      <geom name="hip_motor_stator" type="cylinder" pos="0 {hip_motor_y} 0"
            size="{j4310_radius:.12g} {j4310_half_width:.12g}"
            euler="{math.pi/2} 0 0"
            mass="{masses.hip_motor_stator}" rgba="0.10 0.10 0.12 1"/>
      <site name="O"/>

      <!-- MuJoCo 的树内 hinge 轴为 -Y，euler 取 -ref，使 qpos 直接等于几何角。 -->
      <body name="leg_carrier" euler="0 {-refs['hip']:.12g} 0">
        <joint name="hip" ref="{q('hip')}" range="-1.75 0.35" armature="0.001"/>
        <geom name="hip_motor_rotor" type="cylinder" pos="0 {hip_rotor_y} 0"
              size="{j4310_output_radius:.12g} 0.008" euler="{math.pi/2} 0 0"
              mass="{masses.hip_motor_rotor}" rgba="0.16 0.16 0.18 1"/>
        <geom name="hip_knee_flange" type="cylinder" pos="0 {flange_y} 0"
              size="{j4310_output_radius:.12g} {flange_half_width}"
              euler="{math.pi/2} 0 0"
              mass="{masses.flange}" rgba="0.48 0.50 0.54 1"/>
        <geom name="hip_knee_register" type="cylinder" pos="0 {hip_rotor_y} 0"
              size="{j4310_register_radius:.12g} 0.006"
              euler="{math.pi/2} 0 0" mass="0" rgba="0.62 0.63 0.66 1"/>

        <!-- 膝电机定子随髋转子和法兰整体摆动，但与腿架同轴。 -->
        <body name="knee_motor" pos="0 {knee_motor_y} 0">
          <geom name="knee_motor_stator" type="cylinder" pos="0 0 0"
                size="{j4310_radius:.12g} {j4310_half_width:.12g}"
                euler="{math.pi/2} 0 0"
                mass="{masses.knee_motor_stator}" rgba="0.10 0.10 0.12 1"/>

          <!-- 整个腿部连杆机构挂在膝电机；此平面位于膝电机外侧。 -->
          <body name="linkage_base" pos="0 {motor_to_link:.12g} 0">
            <!-- OB 固定在膝电机定子，O 与膝电机同轴。 -->
            <geom name="fixed_arm_OB" type="capsule" fromto="0 0 0 0 0 -{L}"
                  size="0.010" mass="{masses.thigh}" rgba="0.30 0.80 0.35 1"/>
            <site name="B" pos="0 0 -{L}"/>

            <!-- OA 与膝电机转子刚性相连，knee_drive 是唯一主动的腿长关节。 -->
            <body name="active_arm_OA" euler="0 {-refs['knee_drive']:.12g} 0">
              <joint name="knee_drive" ref="{q('knee_drive')}"
                     range="-2.20 -0.90" armature="0.001"
                     solreflimit="0.001 1"
                     solimplimit="0.99 0.999 0.001 0.5 2"/>
              <geom name="knee_motor_rotor" type="cylinder"
                    pos="0 {-motor_to_link:.12g} 0"
                    size="{j4310_output_radius:.12g} 0.008"
                    euler="{math.pi/2} 0 0" mass="{masses.knee_motor_rotor}"
                    rgba="0.16 0.16 0.18 1"/>
              <geom name="knee_output_shaft" type="cylinder"
                    pos="0 {-motor_to_link/2:.12g} 0"
                    size="0.007 {motor_to_link/2:.12g}"
                    euler="{math.pi/2} 0 0" mass="0"
                    rgba="0.22 0.22 0.25 1"/>
              <geom name="knee_output_flange" type="cylinder"
                    pos="0 {-motor_to_link:.12g} 0"
                    size="{j4310_output_radius:.12g} 0.003"
                    euler="{math.pi/2} 0 0" mass="0"
                    rgba="0.48 0.50 0.54 1"/>
              <geom name="active_arm_OA" type="capsule"
                    fromto="0 0 0 0 0 -{r}" size="0.008"
                    mass="{masses.crank}" rgba="0.55 0.35 0.90 1"/>
              <site name="A" pos="0 0 -{r}"/>

              <!-- AC 仅有被动 A 轴承；其末端 C 由 connect 形成第三轴承。 -->
              <body name="coupler_AC" pos="0 0 -{r}"
                    euler="0 {-refs['bearing_A']:.12g} 0">
                <joint name="bearing_A" ref="{q('bearing_A')}"
                       range="0.60 2.45"/>
                <geom name="coupler_AC" type="capsule"
                      fromto="0 0 0 0 0 -{L}" size="0.0045"
                      mass="{masses.coupler}" rgba="0.90 0.35 0.35 1"/>
                <site name="C_AC" pos="0 0 -{L}" size="0.0025"
                      rgba="1 1 1 1"/>
              </body>
            </body>

            <!-- BC 与 BW 是同一根刚性输出杆，B 处为被动轴承。 -->
            <body name="output_link_BCW" pos="0 0 -{L}"
                  euler="0 {-refs['bearing_B']:.12g} 0">
              <joint name="bearing_B" ref="{q('bearing_B')}"
                     range="0.60 2.45"/>
              <geom name="rigid_link_BCW" type="capsule"
                    fromto="0 0 {r} 0 0 -{L}" size="0.009"
                    mass="{masses.shank}" rgba="0.20 0.75 0.85 1"/>
              <geom name="output_bearing_boss" type="cylinder"
                    pos="0 0 0" size="0.014 0.005"
                    euler="{math.pi/2} 0 0" mass="{masses.shank_offset_arm}"
                    rgba="0.20 0.65 0.75 1"/>
              <geom name="wheel_carrier" type="capsule"
                    fromto="0 -0.012 {-L} 0 0.012 {-L}" size="0.005"
                    mass="{masses.shank_link_arm}" rgba="0.20 0.65 0.75 1"/>
              <site name="C_BC" pos="0 0 {r}" size="0.0025"
                    rgba="1 1 1 1"/>
              <site name="W" pos="0 0 -{L}"/>

              <!-- 轮毂电机定子固定在 BC/BW 输出杆，转子和轮子同转。 -->
              <geom name="hub_motor_stator" type="cylinder"
                    pos="0 0 {-L}" size="{h6215_radius:.12g} {h6215_half_width:.12g}"
                    euler="{math.pi/2} 0 0"
                    mass="{masses.hub_motor_stator}" rgba="0.14 0.14 0.16 1"/>
              <geom name="hub_motor_stator_face" type="cylinder"
                    pos="0 {-wheel_motor_face_y:.12g} {-L}"
                    size="{h6215_output_radius:.12g} 0.003"
                    euler="{math.pi/2} 0 0" mass="0" rgba="0.20 0.20 0.23 1"/>
              <body name="wheel" pos="0 0 {-L}">
                <joint name="wheel_spin" axis="0 1 0" limited="false"
                       damping="0.002" armature="0.0001"/>
                <geom name="tire" type="cylinder" size="{R} 0.016"
                      euler="{math.pi/2} 0 0" mass="{masses.wheel_tire}"
                      rgba="0.06 0.06 0.07 1" contype="1" conaffinity="1" condim="3"/>
                <geom name="wheel_rim" type="cylinder" size="0.043 0.012"
                      euler="{math.pi/2} 0 0" mass="{masses.wheel_rim}"
                      rgba="0.34 0.35 0.38 1"/>
                <geom name="hub_motor_rotor" type="cylinder"
                      size="{h6215_radius:.12g} {h6215_half_width:.12g}"
                      euler="{math.pi/2} 0 0" mass="{masses.hub_motor_rotor}"
                      rgba="0.22 0.22 0.25 1"/>
                <geom name="hub_motor_rotor_face" type="cylinder"
                      pos="0 {wheel_motor_face_y:.12g} 0"
                      size="{h6215_output_radius:.12g} 0.003"
                      euler="{math.pi/2} 0 0" mass="0" rgba="0.26 0.26 0.29 1"/>
                <geom name="wheel_spoke" type="capsule"
                      fromto="0 -0.017 {-0.8*R} 0 -0.017 {0.8*R}"
                      size="0.003" mass="0" rgba="0.85 0.85 0.85 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <!-- C 处被动轴承：AC 与 BC 两个树分支的位置重合。 -->
    <connect name="loop_C" site1="C_AC" site2="C_BC"
             solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </equality>
  <actuator>
    <!-- hip 驱动整条腿前后摆动；knee_drive 驱动 OA 控制腿长。 -->
    <motor name="hip_motor" joint="hip"
           ctrlrange="{-DM_J4310_2EC_V11.peak_torque_nm} {DM_J4310_2EC_V11.peak_torque_nm}"/>
    <motor name="knee_motor" joint="knee_drive"
           ctrlrange="{-DM_J4310_2EC_V11.peak_torque_nm} {DM_J4310_2EC_V11.peak_torque_nm}"/>
    <motor name="wheel_motor" joint="wheel_spin"
           ctrlrange="{-wheel_torque_limit:.12g} {wheel_torque_limit:.12g}"/>
  </actuator>
</mujoco>
'''
    root = ET.fromstring(xml)
    # Keep the existing equivalent inertia solids, but draw separate exposed
    # stator/rotor surfaces. All display parts have zero mass.
    for name in ("hip_motor_stator", "hip_motor_rotor", "knee_motor_stator",
                 "knee_motor_rotor", "hub_motor_stator", "hub_motor_rotor",
                 "hub_motor_stator_face", "hub_motor_rotor_face",
                 "knee_output_flange"):
        geom = root.find(f".//geom[@name='{name}']")
        geom.set("rgba", "0 0 0 0")
        geom.set("group", "3")

    def cylinder(body_name, name, y, z, radius, half_width, color):
        body = root.find(f".//body[@name='{body_name}']")
        ET.SubElement(body, "geom", {
            "name": name, "type": "cylinder", "pos": f"0 {y:.12g} {z:.12g}",
            "size": f"{radius:.12g} {half_width:.12g}",
            "euler": f"{math.pi/2} 0 0", "mass": "0", "rgba": color,
        })

    def marker(body_name, name, y, z, radius):
        body = root.find(f".//body[@name='{body_name}']")
        ET.SubElement(body, "geom", {
            "name": name, "type": "capsule",
            "fromto": f"{radius*.55} {y} {z} {radius*.9} {y} {z}",
            "size": "0.0012", "mass": "0", "rgba": "1 1 1 1",
        })

    asset = ET.SubElement(root, 'asset')
    add_ring_mesh(asset, 'procedural_j4310_ring', j4310_radius, j4310_half_width)
    add_ring_mesh(asset, 'procedural_h6215_ring', h6215_radius, h6215_half_width)

    def outer_cylinder(body_name, name, center, z, radius, half_height, color, mesh):
        # Rear half is solid; the front half surrounds the smaller core.
        cylinder(body_name, name+'_back', center-half_height/2, z,
                 radius, half_height/2, color)
        body = root.find(f".//body[@name='{body_name}']")
        ET.SubElement(body, 'geom', {
            'name': name, 'type': 'mesh', 'mesh': mesh,
            'pos': f'0 {center:.12g} {z:.12g}', 'mass': '0', 'rgba': color,
        })

    for prefix, stator_body, rotor_body, center, rotor_center, colors in (
        ("hip", "hip_mount", "leg_carrier", hip_motor_y,
         hip_motor_y + j4310_half_width/2,
         ("0.08 0.30 0.80 0.65", "1 0.38 0.06 1")),
        ("knee", "knee_motor", "active_arm_OA", 0,
         -motor_to_link + j4310_half_width/2,
         ("0.04 0.65 0.70 0.65", "1 0.80 0.08 1")),
    ):
        outer_cylinder(stator_body, prefix+"_stator_shell", center, 0,
                       j4310_radius, j4310_half_width, colors[0], 'procedural_j4310_ring')
        cylinder(rotor_body, prefix+"_rotor_cap", rotor_center, 0,
                 j4310_radius/2, j4310_half_width/2, colors[1])
        marker(rotor_body, prefix+"_rotor_mark", rotor_center+j4310_half_width/2+.0005,
               0, j4310_radius/2)

    outer_cylinder("wheel", "hub_rotor_cap", 0, 0,
                   h6215_radius, h6215_half_width, "0.90 0.12 0.22 1", 'procedural_h6215_ring')
    cylinder("output_link_BCW", "hub_stator_shell", h6215_half_width/2,
             -L, h6215_radius/2, h6215_half_width/2, "0.48 0.18 0.75 1")
    marker("wheel", "hub_rotor_mark", h6215_half_width+.0005, 0, h6215_radius)

    # Axial connections match the exposed inner rotor and outer stator.
    flange = root.find(".//geom[@name='hip_knee_flange']")
    # Preserve the pre-existing flange mass/inertia: this edit is visual only.
    flange.set('rgba', '0 0 0 0')
    flange.set('size', f'{j4310_output_radius} {flange_half_width}')
    cylinder('leg_carrier', 'hip_rotor_coupling', 0, 0, j4310_radius/2, .002,
             '1 0.38 0.06 1')
    root.find(".//geom[@name='hip_knee_register']").set('rgba', '0 0 0 0')
    shaft = root.find(".//geom[@name='knee_output_shaft']")
    gap = motor_to_link-j4310_half_width
    shaft.set('pos', f'0 {-gap/2} 0')
    shaft.set('size', f'.006 {gap/2}')
    shaft.set('rgba', '1 .80 .08 1')
    base = root.find(".//body[@name='linkage_base']")
    # The fixed link's visible root sits on the stator annulus, not the rotor
    # axis. Its old full-length solid remains only as an inertia approximation.
    fixed_arm = root.find(".//geom[@name='fixed_arm_OB']")
    fixed_arm.set('rgba', '0 0 0 0')
    fixed_arm.set('group', '3')
    ET.SubElement(base, 'geom', {
        'name': 'fixed_arm_OB_visual', 'type': 'capsule',
        'fromto': f'0 0 -0.022 0 0 {-L}',
        'size': '.010', 'mass': '0', 'rgba': '.30 .80 .35 1',
    })
    ET.SubElement(base, 'geom', {
        'name': 'knee_stator_OB_bracket', 'type': 'capsule',
        'fromto': f'0 {-gap} -0.022 0 0 -0.022',
        'size': '.003', 'mass': '0', 'rgba': '.04 .65 .70 1',
    })
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length-mm", type=float, default=130,
                   help="OB=AC=BW 长度，默认 130 mm（温和缩短方案）")
    p.add_argument("--crank-mm", type=float, default=35)
    p.add_argument("--wheel-radius-mm", type=float, default=50)
    p.add_argument("--hip-height-mm", type=float, default=360)
    p.add_argument("--output", type=Path, default=GENERATED_MODEL_DIR / "wheel_leg.xml")
    args = p.parse_args()
    g = Geometry(args.length_mm/1000, args.crank_mm/1000,
                 args.wheel_radius_mm/1000, args.hip_height_mm/1000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_xml(g), encoding="utf-8")
    print(f"Written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
