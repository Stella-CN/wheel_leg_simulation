"""从单腿 MJCF 复制左右两套腿，并连接到固定机身长方体。

机身尺寸由用户确认为 20 x 15 x 10 cm，即 200 x 150 x 100 mm。
左右髋轴位于机身 Y 两侧，负 Y 为左腿，正 Y 为右腿。
髋电机位于机身内部，转子通过侧壁处法兰带动外部膝电机。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from .build_model import make_xml
from .build_model import COAXIAL_MOTOR_CENTER_OFFSET_M, HIP_STATOR_INNER_FACE_OFFSET_M
from .motor_specs import DM_J4310_2EC_V11
from .geometry import Geometry
from .paths import GENERATED_MODEL_DIR


SIDES = (("left", -1.0), ("right", 1.0))


def _numbers(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def _format_numbers(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _mirror_local_y(body: ET.Element, side: float) -> None:
    """镜像局部 Y 位置；X/Z 平面几何和关节角保持不变。"""
    for element in body.iter():
        if side < 0 and element.tag == 'geom' and element.get('type') == 'mesh':
            # Procedural annuli extend toward +Y; left-hand motors face -Y.
            element.set('euler', '3.141592653589793 0 0')
        if "pos" in element.attrib:
            values = _numbers(element.get("pos", "0 0 0"))
            if len(values) == 3:
                values[1] *= side
                element.set("pos", _format_numbers(values))
        if "fromto" in element.attrib:
            values = _numbers(element.get("fromto", ""))
            if len(values) == 6:
                values[1] *= side
                values[4] *= side
                element.set("fromto", _format_numbers(values))


def _clone_leg(source_body: ET.Element, side_name: str, side: float,
               hip_axis_offset_m: float) -> ET.Element:
    suffix = f"_{side_name}"
    leg = deepcopy(source_body)
    _mirror_local_y(leg, side)
    for element in leg.iter():
        if "name" in element.attrib:
            element.set("name", element.get("name") + suffix)
    # The source hip_mount is a world child at z=H. It becomes a child of
    # the chassis body, whose frame is already at z=H.
    leg.set("name", f"hip_mount{suffix}")
    leg.set("pos", f"0 {side * hip_axis_offset_m:.12g} 0")
    return leg


def _clone_actuators(source_root: ET.Element, side_name: str) -> list[ET.Element]:
    suffix = f"_{side_name}"
    motors = []
    for motor in source_root.findall("actuator/motor"):
        clone = deepcopy(motor)
        clone.set("name", clone.get("name") + suffix)
        clone.set("joint", clone.get("joint") + suffix)
        motors.append(clone)
    return motors


def make_biped_xml(g: Geometry, body_length_m: float = 0.200,
                   body_width_m: float = 0.150,
                   body_height_m: float = 0.100,
                   hip_axis_offset_m: float | None = None) -> str:
    """Return a two-leg MJCF with two independent copies of the closed leg."""
    if not all(value > 0 for value in
               (body_length_m, body_width_m, body_height_m)):
        raise ValueError("机身长宽高必须为正数。")
    if hip_axis_offset_m is None:
        # The inter-motor coupling straddles the chassis side wall.
        hip_axis_offset_m = body_width_m / 2.0
    half_motor = DM_J4310_2EC_V11.body_length_m / 2
    motor_radius = DM_J4310_2EC_V11.outer_diameter_m / 2
    hip_outer = hip_axis_offset_m-COAXIAL_MOTOR_CENTER_OFFSET_M+half_motor
    knee_inner = hip_axis_offset_m+COAXIAL_MOTOR_CENTER_OFFSET_M-half_motor
    if (not HIP_STATOR_INNER_FACE_OFFSET_M < hip_axis_offset_m
            or hip_outer > body_width_m/2+1e-12
            or knee_inner < body_width_m/2-1e-12
            or 2*motor_radius > min(body_length_m,body_height_m)):
        raise ValueError("安装位置必须使髋电机完整内置、左右不相交，膝电机位于机身外侧")

    source_root = ET.fromstring(make_xml(g))
    source_world = source_root.find("worldbody")
    if source_world is None:
        raise ValueError("单腿模板缺少 worldbody/hip_mount。")
    source_leg = source_world.find("body[@name='hip_mount']")
    if source_leg is None:
        raise ValueError("单腿模板缺少 worldbody/hip_mount。")

    root = deepcopy(source_root)
    root.set("model", "coaxial_biped_parallelogram_wheel_leg")

    custom = root.find("custom")
    if custom is None:
        raise ValueError("单腿模板缺少 custom 参数。")
    ET.SubElement(custom, "numeric", {
        "name": "chassis_geometry",
        "data": f"{body_length_m:.12g} {body_width_m:.12g} "
                f"{body_height_m:.12g} {hip_axis_offset_m:.12g}",
    })

    old_world = root.find("worldbody")
    if old_world is None:
        raise ValueError("单腿模板缺少 worldbody。")
    world_index = list(root).index(old_world)
    world = ET.Element("worldbody")
    for child in source_world:
        if child.tag in ("light", "geom") and child.get("name") == "floor":
            world.append(deepcopy(child))
        elif child.tag == "light":
            world.append(deepcopy(child))

    # The body block is fixed to the world in this bench model. Its mass is
    # intentionally zero because no chassis mass was supplied by the user.
    chassis = ET.SubElement(world, "body", {
        "name": "chassis",
        "pos": f"0 0 {g.hip_height:.12g}",
    })
    ET.SubElement(chassis, "geom", {
        "name": "chassis_box",
        "type": "box",
        "size": f"{body_length_m/2:.12g} {body_width_m/2:.12g} "
                f"{body_height_m/2:.12g}",
        "mass": "0",
        "rgba": "0.28 0.32 0.38 0.35",
    })

    target_z = g.hip_height - g.nominal_length
    for side_name, side in SIDES:
        ET.SubElement(world, "site", {
            "name": f"target_W_{side_name}",
            "pos": f"0 {side * (hip_axis_offset_m + 0.062):.12g} {target_z:.12g}",
            "size": "0.007",
            "rgba": "1 0.55 0.1 0.8",
        })
        chassis.append(_clone_leg(source_leg, side_name, side,
                                   hip_axis_offset_m))

    root.remove(old_world)
    root.insert(world_index, world)

    old_equality = root.find("equality")
    if old_equality is None:
        raise ValueError("单腿模板缺少 equality。")
    equality_index = list(root).index(old_equality)
    equality = ET.Element("equality")
    source_connect = source_root.find("equality/connect")
    if source_connect is None:
        raise ValueError("单腿模板缺少 C 点 connect 约束。")
    for side_name, _ in SIDES:
        suffix = f"_{side_name}"
        connect = deepcopy(source_connect)
        connect.set("name", f"loop_C{suffix}")
        connect.set("site1", f"C_AC{suffix}")
        connect.set("site2", f"C_BC{suffix}")
        equality.append(connect)
    root.remove(old_equality)
    root.insert(equality_index, equality)

    old_actuator = root.find("actuator")
    if old_actuator is None:
        raise ValueError("单腿模板缺少 actuator。")
    actuator_index = list(root).index(old_actuator)
    actuator = ET.Element("actuator")
    for side_name, _ in SIDES:
        for motor in _clone_actuators(source_root, side_name):
            actuator.append(motor)
    root.remove(old_actuator)
    root.insert(actuator_index, actuator)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length-mm", type=float, default=130,
                        help="OB=AC=BW 长度，默认 130 mm（温和缩短方案）")
    parser.add_argument("--crank-mm", type=float, default=35)
    parser.add_argument("--wheel-radius-mm", type=float, default=50)
    parser.add_argument("--hip-height-mm", type=float, default=360)
    parser.add_argument("--body-length-mm", type=float, default=200,
                        help="机身 X 尺寸，默认 200 mm")
    parser.add_argument("--body-width-mm", type=float, default=150,
                        help="机身 Y 尺寸，默认 150 mm")
    parser.add_argument("--body-height-mm", type=float, default=100,
                        help="机身 Z 尺寸，默认 100 mm")
    parser.add_argument("--hip-axis-offset-mm", type=float, default=None,
                        help="髋部连接坐标系 Y 偏移；默认在机身侧壁，髋电机内置、膝电机外置")
    parser.add_argument("--output", type=Path,
                        default=GENERATED_MODEL_DIR / "biped_wheel_leg.xml")
    args = parser.parse_args()
    g = Geometry(args.length_mm / 1000, args.crank_mm / 1000,
                 args.wheel_radius_mm / 1000, args.hip_height_mm / 1000)
    hip_axis_offset = (None if args.hip_axis_offset_mm is None else
                       args.hip_axis_offset_mm / 1000)
    xml = make_biped_xml(
        g,
        args.body_length_mm / 1000,
        args.body_width_mm / 1000,
        args.body_height_mm / 1000,
        hip_axis_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml, encoding="utf-8")
    print(f"Written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
