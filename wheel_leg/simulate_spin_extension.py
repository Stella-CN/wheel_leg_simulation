"""In-place yaw maneuver with synchronized leg extension and recovery.

GUI:
    .venv/bin/mjpython -m wheel_leg.simulate_spin_extension \
        --config configs/simulations/spin_extension.json
Headless:
    .venv/bin/python -m wheel_leg.simulate_spin_extension --headless \
        --config configs/simulations/spin_extension.json

The maneuver uses only the six motor actuators. A smooth yaw reference is
tracked by the wheel motors while the length-scheduled LQR keeps both legs
moving together and holds the chassis position as far as the estimated tire
friction permits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from .balance_control import BalanceController, ExtensionBalanceController
from .config import load_simulation_config
from .paths import CURRENT_RESULTS_DIR, GENERATED_MODEL_DIR, project_path


def attitude(d, body_id):
    rotation = d.xmat[body_id].reshape(3, 3)
    return (math.atan2(rotation[2, 1], rotation[2, 2]),
            math.asin(np.clip(-rotation[2, 0], -1, 1)))


def tilt_angle(d, body_id):
    rotation = d.xmat[body_id].reshape(3, 3)
    return math.acos(float(np.clip(rotation[2, 2], -1, 1)))


def yaw_angle(d, body_id):
    rotation = d.xmat[body_id].reshape(3, 3)
    return math.atan2(rotation[1, 0], rotation[0, 0])


def unwrap_delta(angle, previous):
    return (angle - previous + math.pi) % (2 * math.pi) - math.pi


class YawTrajectory:
    """Quintic yaw reference with zero angular velocity at both ends."""

    def __init__(self, start, duration, target):
        values = (start, duration, target)
        if (not np.isfinite(values).all() or start < 0 or duration <= 0
                or abs(target) < math.radians(1e-3)):
            raise ValueError("航向轨迹参数必须有限，时长为正且目标角不能为零")
        self.start = float(start)
        self.duration = float(duration)
        self.target = float(target)

    @staticmethod
    def _position(s):
        s = min(1.0, max(0.0, s))
        return 10 * s**3 - 15 * s**4 + 6 * s**5

    @staticmethod
    def _rate(s):
        s = min(1.0, max(0.0, s))
        return 30 * s**2 - 60 * s**3 + 30 * s**4

    def sample(self, now):
        s = (float(now) - self.start) / self.duration
        return (self.target * self._position(s),
                self.target / self.duration * self._rate(s)
                if 0 < s < 1 else 0.0)


class YawReferenceExtensionController(ExtensionBalanceController):
    """Leg-length scheduling plus a smooth chassis-yaw reference."""

    def __init__(self, m, *, leg_amplitude, leg_period, motion_start,
                 spin_start, spin_duration, target_yaw,
                 horizontal_position_weights,
                 horizontal_velocity_weights,
                 station_keeping_position_gain,
                 station_keeping_velocity_gain):
        super().__init__(
            m, leg_amplitude, leg_period, motion_start,
            horizontal_position_weights=horizontal_position_weights,
            horizontal_velocity_weights=horizontal_velocity_weights)
        self.yaw_trajectory = YawTrajectory(
            spin_start, spin_duration, target_yaw)
        self.station_keeping_position_gain = float(
            station_keeping_position_gain)
        self.station_keeping_velocity_gain = float(
            station_keeping_velocity_gain)
        self.body_id = m.body("chassis").id

    def control(self, d):
        # Update the leg-length reference first, then replace only the base
        # yaw and yaw-rate target. The remaining LQR states retain the loaded
        # equilibrium and the synchronized leg trajectory.
        self.update_reference(float(d.time))
        target_yaw, target_rate = self.yaw_trajectory.sample(d.time)
        self.qref[3:7] = [math.cos(target_yaw / 2), 0.0, 0.0,
                          math.sin(target_yaw / 2)]
        self.vref[5] = target_rate
        command = BalanceController.control(self, d)
        rotation = d.xmat[self.body_id].reshape(3, 3)
        forward_position = (rotation[0, 0] * d.qpos[0]
                            + rotation[1, 0] * d.qpos[1])
        forward_velocity = (rotation[0, 0] * d.qvel[0]
                            + rotation[1, 0] * d.qvel[1])
        station_keeping = (
            self.station_keeping_position_gain * forward_position
            + self.station_keeping_velocity_gain * forward_velocity)
        for side in ("left", "right"):
            command[self.m.actuator(f"wheel_motor_{side}").id] += station_keeping
        return command


class SpinExtensionExperiment:
    """Run the spin, leg-extension and post-maneuver balance test."""

    def __init__(self, args):
        self.args = args
        self.m = mujoco.MjModel.from_xml_path(str(args.model))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        if (m.nq, m.nv, m.nu, m.neq) != (17, 16, 6, 2):
            raise ValueError("Spin-extension requires the six-DOF floating ground model")

        self.body = m.body("chassis").id
        self.floor = m.geom("floor").id
        self.tire_ids = {
            side: m.geom(f"tire_{side}").id for side in ("left", "right")
        }
        self.tire_set = set(self.tire_ids.values())
        self.controller = YawReferenceExtensionController(
            m,
            leg_amplitude=float(args.leg_amplitude_mm) / 1000,
            leg_period=float(args.leg_period),
            motion_start=float(args.motion_start),
            spin_start=float(args.spin_start),
            spin_duration=float(args.spin_duration),
            target_yaw=math.radians(float(args.target_yaw_deg)),
            horizontal_position_weights=(
                float(args.horizontal_position_weight_x),
                float(args.horizontal_position_weight_y)),
            horizontal_velocity_weights=(
                float(args.horizontal_velocity_weight_x),
                float(args.horizontal_velocity_weight_y)),
            station_keeping_position_gain=float(
                args.station_keeping_position_gain),
            station_keeping_velocity_gain=float(
                args.station_keeping_velocity_gain),
        )
        self.spin_start = float(args.spin_start)
        self.spin_end = self.spin_start + float(args.spin_duration)
        self.nominal_length = self.controller.trajectory.nominal
        self.target_yaw_rad = math.radians(float(args.target_yaw_deg))

        d.qpos[:] = self.controller.qref
        d.qvel[:] = 0
        d.ctrl[:] = 0
        self._initialize_start_pose()
        mujoco.mj_forward(m, d)

        self.initial_height = float(d.xpos[self.body, 2])
        self.initial_pitch_deg = float(args.pitch_deg)
        self.initial_roll_deg = float(args.roll_deg)
        self.initial_yaw = yaw_angle(d, self.body)
        self._last_yaw = self.initial_yaw
        self._yaw_unwrapped = self.initial_yaw
        self._yaw_at_spin_end = None
        self.phase = "balance_extension"
        self.failure = None
        self.rows = []
        self.steps = 0
        self.max_closure = 0.0
        self.max_nonwheel_contacts = 0
        self.max_planar_displacement = 0.0
        self.max_yaw_error = 0.0
        self.max_yaw_rate = 0.0
        self.max_leg_error = 0.0
        self.peaks = np.zeros(m.nu)
        self.saturation_counts = np.zeros(m.nu, dtype=int)
        self.rated_counts = np.zeros(m.nu, dtype=int)
        self.rated = np.array([
            1 if "wheel" in m.actuator(i).name else 3 for i in range(m.nu)
        ])
        self.record()

    def _initialize_start_pose(self):
        """Place the tilted nominal pose with the tires just above the floor."""
        d, m, a = self.d, self.m, self.args
        roll_q, pitch_q, quat = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            roll_q, np.array([1.0, 0.0, 0.0]),
            math.radians(float(a.roll_deg)))
        mujoco.mju_axisAngle2Quat(
            pitch_q, np.array([0.0, 1.0, 0.0]),
            math.radians(float(a.pitch_deg)))
        mujoco.mju_mulQuat(quat, pitch_q, roll_q)
        d.qpos[3:7] = quat
        mujoco.mj_forward(m, d)
        lowest = min(
            d.geom_xpos[geom_id, 2] - (
                m.geom_size[geom_id, 0]
                * math.sqrt(max(0.0, 1.0 -
                                d.geom_xmat[geom_id].reshape(3, 3)[2, 2] ** 2))
                + m.geom_size[geom_id, 1]
                * abs(d.geom_xmat[geom_id].reshape(3, 3)[2, 2]))
            for geom_id in self.tire_set
        )
        d.qpos[2] += float(a.drop_height) - lowest

    def _floor_contacts(self):
        contacted = set()
        nonwheel = 0
        normal = 0.0
        for index in range(self.d.ncon):
            contact = self.d.contact[index]
            if contact.efc_address < 0 or self.floor not in (
                    contact.geom1, contact.geom2):
                continue
            tire_ids = self.tire_set.intersection((contact.geom1, contact.geom2))
            if tire_ids:
                contacted.update(tire_ids)
            else:
                nonwheel += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, index, force)
            normal += float(force[0])
        return contacted, nonwheel, normal

    def _phase(self):
        now = float(self.d.time)
        if now < self.spin_start:
            return "balance_extension"
        if now < self.spin_end:
            return "spin_and_extension"
        return "post_spin_balance"

    def step(self):
        m, d = self.m, self.d
        d.ctrl[:] = np.clip(
            self.controller.control(d), m.actuator_ctrlrange[:, 0],
            m.actuator_ctrlrange[:, 1])
        d.xfrc_applied[:] = 0
        self.peaks = np.maximum(self.peaks, np.abs(d.ctrl))
        self.saturation_counts += np.abs(d.ctrl) >= m.actuator_ctrlrange[:, 1] * .999
        self.rated_counts += np.abs(d.ctrl) > self.rated
        previous_time = d.time
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        self.steps += 1
        self.max_yaw_rate = max(self.max_yaw_rate, abs(float(d.qvel[5])))

        now_yaw = yaw_angle(d, self.body)
        self._yaw_unwrapped += unwrap_delta(now_yaw, self._last_yaw)
        self._last_yaw = now_yaw
        if self._yaw_at_spin_end is None and d.time >= self.spin_end:
            self._yaw_at_spin_end = self._yaw_unwrapped

        contacted, nonwheel, normal = self._floor_contacts()
        self.max_nonwheel_contacts = max(self.max_nonwheel_contacts, nonwheel)
        gaps = [np.linalg.norm(
            d.site(f"C_AC_{side}").xpos - d.site(f"C_BC_{side}").xpos)
                for side in ("left", "right")]
        self.max_closure = max(self.max_closure, *gaps)
        displacement = float(np.hypot(d.qpos[0], d.qpos[1]))
        self.max_planar_displacement = max(
            self.max_planar_displacement, displacement)
        target_yaw, _ = self.controller.yaw_trajectory.sample(d.time)
        self.max_yaw_error = max(
            self.max_yaw_error,
            abs(self._yaw_unwrapped - self.initial_yaw
                - target_yaw))
        target_length, _ = self.controller.trajectory.sample(d.time)
        actual_lengths = []
        for side in ("left", "right"):
            relative = d.site(f"W_{side}").xpos - d.site(f"O_{side}").xpos
            local = d.xmat[self.body].reshape(3, 3).T @ relative
            actual_lengths.append(float(np.linalg.norm(local[[0, 2]])))
        self.max_leg_error = max(
            self.max_leg_error,
            *(abs(length - target_length) for length in actual_lengths))
        self.phase = self._phase()
        roll, pitch = attitude(d, self.body)
        if (d.time <= previous_time or not np.isfinite(d.qpos).all()
                or not np.isfinite(d.qvel).all()):
            self.failure = "numerical_instability"
        elif max(gaps) > .005:
            self.failure = "closure_exceeds_5mm"
        elif nonwheel:
            self.failure = "nonwheel_ground_contact"
        elif np.any(d.warning.number):
            self.failure = "mujoco_warning"
        elif (d.xpos[self.body, 2] < .14
              or max(abs(roll), abs(pitch)) > math.radians(35)):
            self.failure = "fallen"
        if self.steps % 10 == 0 or self.failure:
            self.record(contacted=contacted, nonwheel=nonwheel, normal=normal)
        return self.failure is None

    def record(self, contacted=None, nonwheel=None, normal=None):
        m, d = self.m, self.d
        if contacted is None or nonwheel is None or normal is None:
            contacted, nonwheel, normal = self._floor_contacts()
        roll, pitch = attitude(d, self.body)
        target_yaw, target_rate = self.controller.yaw_trajectory.sample(d.time)
        target_length, target_speed = self.controller.trajectory.sample(d.time)
        row = {
            "time_s": float(d.time),
            "phase": self.phase,
            "x_m": float(d.qpos[0]),
            "y_m": float(d.qpos[1]),
            "planar_displacement_m": float(np.hypot(d.qpos[0], d.qpos[1])),
            "height_m": float(d.xpos[self.body, 2]),
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(self._yaw_unwrapped - self.initial_yaw),
            "target_yaw_deg": math.degrees(target_yaw),
            "yaw_error_deg": math.degrees(
                self._yaw_unwrapped - self.initial_yaw - target_yaw),
            "yaw_rate_deg_s": math.degrees(float(d.qvel[5])),
            "target_yaw_rate_deg_s": math.degrees(target_rate),
            "tilt_deg": math.degrees(tilt_angle(d, self.body)),
            "wheel_contact_count": len(contacted),
            "both_wheels_contact": len(contacted) == 2,
            "floor_normal_n": float(normal),
            "nonwheel_ground_contacts": int(nonwheel),
            "target_leg_length_m": target_length,
            "target_leg_speed_m_s": target_speed,
        }
        row.update({m.actuator(i).name + "_nm": float(d.ctrl[i])
                   for i in range(m.nu)})
        for side in ("left", "right"):
            relative = d.site(f"W_{side}").xpos - d.site(f"O_{side}").xpos
            local = d.xmat[self.body].reshape(3, 3).T @ relative
            row[f"{side}_leg_length_m"] = float(np.linalg.norm(local[[0, 2]]))
        self.rows.append(row)

    def save(self, status, *, verbose=True):
        d, m, a = self.d, self.m, self.args
        out = a.output
        out.mkdir(parents=True, exist_ok=True)
        self.record()
        tail_start = self.spin_end + float(a.post_spin_settle)
        tail = [row for row in self.rows if row["time_s"] >= tail_start]
        if not tail:
            tail = [row for row in self.rows
                    if row["time_s"] >= max(self.spin_end, d.time - 1.0)]
        balanced = bool(
            status == "completed" and d.time >= tail_start and tail
            and all(row["both_wheels_contact"] for row in tail)
            and all(row["nonwheel_ground_contacts"] == 0 for row in tail)
            and max(abs(row["pitch_deg"]) for row in tail) < 3
            and max(abs(row["roll_deg"]) for row in tail) < 3
            and max(abs(row["yaw_rate_deg_s"]) for row in tail) < 3
            and max(abs(row["x_m"]) for row in tail) < .1
            and max(abs(row["y_m"]) for row in tail) < .1
            and self.max_closure < .005)
        spin_end_yaw = (self._yaw_at_spin_end
                        if self._yaw_at_spin_end is not None
                        else self._yaw_unwrapped)
        actual_turn = spin_end_yaw - self.initial_yaw
        yaw_tolerance = math.radians(float(a.yaw_tolerance_deg))
        spin_passed = bool(
            status == "completed"
            and abs(actual_turn - self.target_yaw_rad) <= yaw_tolerance
            and self.max_planar_displacement <= float(a.max_planar_drift_mm) / 1000
            and self.max_nonwheel_contacts == 0
            and self.max_closure < .005)
        active = [row for row in self.rows
                  if row["time_s"] >= float(a.motion_start)]
        if active:
            actual_ranges = {
                side: [min(row[f"{side}_leg_length_m"] for row in active),
                       max(row[f"{side}_leg_length_m"] for row in active)]
                for side in ("left", "right")
            }
            target_range = max(row["target_leg_length_m"] for row in active) - min(
                row["target_leg_length_m"] for row in active)
        else:
            actual_ranges, target_range = {}, 0.0
        full_cycle = d.time >= float(a.motion_start) + float(a.leg_period)
        extension_passed = bool(
            full_cycle and active and target_range > 0
            and all(hi - lo > .6 * target_range
                    for lo, hi in actual_ranges.values())
            and self.max_leg_error < .010)
        report = {
            "status": status,
            "spin_extension_passed": bool(spin_passed and extension_passed
                                           and balanced),
            "spin_passed": spin_passed,
            "extension_passed": extension_passed,
            "balanced_after_turn": balanced,
            "model": str(a.model.resolve()),
            "scenario_config": (str(a.config_path)
                                 if getattr(a, "config_path", None) else None),
            "controller": "horizontal-position-weighted length-scheduled LQR + quintic yaw reference",
            "state_feedback": "ideal MuJoCo state; no sensor noise or delay",
            "total_mass_kg": float(m.body_subtreemass[self.body]),
            "duration_s": float(d.time),
            "initial_height_m": self.initial_height,
            "target_yaw_deg": math.degrees(self.target_yaw_rad),
            "spin_start_s": self.spin_start,
            "spin_duration_s": float(a.spin_duration),
            "spin_end_yaw_deg": math.degrees(actual_turn),
            "yaw_error_at_spin_end_deg": math.degrees(actual_turn - self.target_yaw_rad),
            "actual_turns": actual_turn / (2 * math.pi),
            "max_yaw_error_deg": math.degrees(self.max_yaw_error),
            "max_yaw_rate_deg_s": math.degrees(self.max_yaw_rate),
            "max_planar_displacement_mm": self.max_planar_displacement * 1000,
            "max_pitch_deg": max(abs(row["pitch_deg"]) for row in self.rows),
            "max_roll_deg": max(abs(row["roll_deg"]) for row in self.rows),
            "max_tilt_deg": max(row["tilt_deg"] for row in self.rows),
            "max_closure_mm": self.max_closure * 1000,
            "max_nonwheel_ground_contacts": self.max_nonwheel_contacts,
            "leg_amplitude_mm": float(a.leg_amplitude_mm),
            "leg_period_s": float(a.leg_period),
            "motion_start_s": float(a.motion_start),
            "target_leg_range_mm": target_range * 1000,
            "leg_length_range_m": actual_ranges,
            "max_leg_tracking_error_mm": self.max_leg_error * 1000,
            "complete_leg_cycle_tested": full_cycle,
            "horizontal_position_weights": [
                float(a.horizontal_position_weight_x),
                float(a.horizontal_position_weight_y)],
            "horizontal_velocity_weights": [
                float(a.horizontal_velocity_weight_x),
                float(a.horizontal_velocity_weight_y)],
            "station_keeping_position_gain": float(
                a.station_keeping_position_gain),
            "station_keeping_velocity_gain": float(
                a.station_keeping_velocity_gain),
            "peak_torques_nm": dict(zip(
                [m.actuator(i).name for i in range(m.nu)], self.peaks.tolist())),
            "saturation_fractions": (
                self.saturation_counts / max(self.steps, 1)).tolist(),
            "above_rated_torque_fractions": (
                self.rated_counts / max(self.steps, 1)).tolist(),
            "mujoco_warnings": {
                str(i): int(w.number) for i, w in enumerate(d.warning)
                if w.number},
        }
        with (out / "log.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        (out / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if verbose:
            print(json.dumps(report, indent=2))
        return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    preliminary_args, _ = preliminary.parse_known_args()
    config, config_path = load_simulation_config(preliminary_args.config)
    parser.add_argument("--config", type=Path, default=preliminary_args.config,
                        help="原地转向+屈伸专项 JSON 配置")
    parser.add_argument("--model", type=Path,
                        default=config.get("model", GENERATED_MODEL_DIR / "ground_robot.xml"))
    parser.add_argument("--output", type=Path,
                        default=config.get("output", CURRENT_RESULTS_DIR / "spin_extension"))
    parser.add_argument("--headless", action="store_true",
                        default=config.get("headless", False))
    parser.add_argument("--duration", type=float, default=config.get("duration", 11.0))
    parser.add_argument("--drop-height", type=float,
                        default=config.get("drop_height", .01))
    parser.add_argument("--pitch-deg", type=float, default=config.get("pitch_deg", 3.0))
    parser.add_argument("--roll-deg", type=float, default=config.get("roll_deg", 0.0))
    parser.add_argument("--leg-amplitude-mm", type=float,
                        default=config.get("leg_amplitude_mm", 10.0))
    parser.add_argument("--leg-period", type=float,
                        default=config.get("leg_period", 6.0))
    parser.add_argument("--motion-start", type=float,
                        default=config.get("motion_start", 2.0))
    parser.add_argument("--target-yaw-deg", type=float,
                        default=config.get("target_yaw_deg", 90.0))
    parser.add_argument("--spin-start", type=float,
                        default=config.get("spin_start", 3.0))
    parser.add_argument("--spin-duration", type=float,
                        default=config.get("spin_duration", 3.0))
    parser.add_argument("--post-spin-settle", type=float,
                        default=config.get("post_spin_settle", 3.0))
    parser.add_argument("--horizontal-position-weight-x", type=float,
                        default=config.get("horizontal_position_weight_x", 3000.0))
    parser.add_argument("--horizontal-position-weight-y", type=float,
                        default=config.get("horizontal_position_weight_y", 1000.0))
    parser.add_argument("--horizontal-velocity-weight-x", type=float,
                        default=config.get("horizontal_velocity_weight_x", 10.0))
    parser.add_argument("--horizontal-velocity-weight-y", type=float,
                        default=config.get("horizontal_velocity_weight_y", 10.0))
    parser.add_argument("--station-keeping-position-gain", type=float,
                        default=config.get("station_keeping_position_gain", 2.0))
    parser.add_argument("--station-keeping-velocity-gain", type=float,
                        default=config.get("station_keeping_velocity_gain", 1.0))
    parser.add_argument("--yaw-tolerance-deg", type=float,
                        default=config.get("yaw_tolerance_deg", 10.0))
    parser.add_argument("--max-planar-drift-mm", type=float,
                        default=config.get("max_planar_drift_mm", 80.0))
    args = parser.parse_args()
    args.config_path = config_path
    args.model = project_path(args.model)
    args.output = project_path(args.output)
    values = (
        args.duration, args.drop_height, args.pitch_deg, args.roll_deg,
        args.leg_amplitude_mm, args.leg_period, args.motion_start,
        args.target_yaw_deg, args.spin_start, args.spin_duration,
        args.post_spin_settle, args.horizontal_position_weight_x,
        args.horizontal_position_weight_y, args.horizontal_velocity_weight_x,
        args.horizontal_velocity_weight_y, args.yaw_tolerance_deg,
        args.max_planar_drift_mm, args.station_keeping_position_gain,
        args.station_keeping_velocity_gain)
    if not np.isfinite(values).all():
        parser.error("所有转向与屈伸参数必须是有限数")
    if (args.duration <= 0 or args.drop_height < 0 or args.leg_amplitude_mm <= 0
            or args.leg_period < 4 or args.motion_start < 2
            or abs(args.target_yaw_deg) < 1 or args.spin_start < args.motion_start
            or args.spin_duration < 3 or args.post_spin_settle <= 0
            or min(args.horizontal_position_weight_x,
                   args.horizontal_position_weight_y,
                   args.horizontal_velocity_weight_x,
                   args.horizontal_velocity_weight_y) < 0
            or min(args.station_keeping_position_gain,
                   args.station_keeping_velocity_gain) < 0
            or args.yaw_tolerance_deg <= 0 or args.max_planar_drift_mm <= 0):
        parser.error("转向、屈伸、权重或时序参数超出有效范围")
    if args.duration <= args.spin_start + args.spin_duration + args.post_spin_settle:
        parser.error("duration 必须覆盖转向结束后的平衡恢复阶段")
    return args


def main():
    args = parse_args()
    experiment = SpinExtensionExperiment(args)
    status = "completed"
    try:
        if args.headless:
            while experiment.d.time < args.duration:
                if not experiment.step():
                    status = experiment.failure
                    break
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(experiment.m, experiment.d) as viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = (0, 0, .22)
                    viewer.cam.distance = 1.1
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -20
                while viewer.is_running() and experiment.d.time < args.duration:
                    start = time.perf_counter()
                    with viewer.lock():
                        for _ in range(10):
                            if experiment.d.time >= args.duration or not experiment.step():
                                break
                    viewer.sync()
                    if experiment.failure:
                        status = experiment.failure
                        break
                    time.sleep(max(0.0, .01 - (time.perf_counter() - start)))
                if not experiment.failure and experiment.d.time < args.duration:
                    status = "viewer_closed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "error"
        raise
    finally:
        report = experiment.save(status)
    if args.headless and not report["spin_extension_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
