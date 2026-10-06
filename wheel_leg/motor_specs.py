"""电机资料参数；数值来自项目外部参考目录或用户明确提供的数据。

MuJoCo 当前使用这些参数生成外形、质量分配和输出力矩限幅。没有在参考
资料中出现的字段保持为 None，不用猜测值替代。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorSpec:
    name: str
    reference_dir: str
    mass_kg: float | None
    outer_diameter_m: float | None
    body_length_m: float | None
    rated_voltage_v: float | None
    maximum_voltage_v: float | None
    rated_torque_nm: float | None
    peak_torque_nm: float | None
    rated_speed_rpm: float | None
    no_load_speed_rpm: float | None
    gear_ratio: float | None
    pole_pairs: int | None
    phase_inductance_h: float | None
    phase_resistance_ohm: float | None
    slots: int | None
    encoder_bits: int | None
    encoder_count: int | None
    control_interface: str
    tuning_interface: str | None
    output_flange_diameter_m: float | None = None
    output_register_diameter_m: float | None = None
    mounting_bolt_circle_m: float | None = None
    notes: str = ""


DM_J4310_2EC_V11 = MotorSpec(
    name="DM-J4310-2EC V1.1",
    reference_dir="/Users/lvjiaqing/MyProjects/MyRobot/references/DM-J4310-2EC",
    mass_kg=0.300,
    outer_diameter_m=0.057,
    body_length_m=0.046,
    rated_voltage_v=24.0,
    maximum_voltage_v=28.0,
    rated_torque_nm=3.0,
    peak_torque_nm=7.0,
    rated_speed_rpm=120.0,
    no_load_speed_rpm=200.0,
    gear_ratio=10.0,
    pole_pairs=14,
    phase_inductance_h=340e-6,
    phase_resistance_ohm=0.65,
    slots=24,
    encoder_bits=14,
    encoder_count=2,
    control_interface="CAN 1 Mbps",
    tuning_interface="UART 921600 bps",
    output_flange_diameter_m=0.050,
    output_register_diameter_m=0.02430,
    mounting_bolt_circle_m=0.038,
    notes="24 V version; the manual also lists a separate 48 V variant.",
)


DM_H6215 = MotorSpec(
    name="DM-H6215",
    reference_dir="/Users/lvjiaqing/MyProjects/MyRobot/references/DM-H6215",
    # The user supplied 360 g; the supplied manual/drawing does not state mass.
    mass_kg=0.360,
    outer_diameter_m=0.068,
    body_length_m=0.0445,
    rated_voltage_v=24.0,
    maximum_voltage_v=None,
    # Rated and peak torque supplied by the user; not present in the supplied manual.
    rated_torque_nm=1.0,
    peak_torque_nm=2.0,
    rated_speed_rpm=None,
    no_load_speed_rpm=None,
    gear_ratio=None,
    pole_pairs=None,
    phase_inductance_h=None,
    phase_resistance_ohm=None,
    slots=None,
    encoder_bits=None,
    encoder_count=None,
    control_interface="CAN 1 Mbps",
    tuning_interface=None,
    output_flange_diameter_m=0.033,
    output_register_diameter_m=0.022,
    mounting_bolt_circle_m=0.028,
    notes=("Mass, rated torque and peak torque are user-provided. The drawing "
           "also shows a rear Ø17.50 mm mounting circle and 3xM3 holes; no "
           "speed table was found."),
)


J4310 = DM_J4310_2EC_V11
H6215 = DM_H6215
