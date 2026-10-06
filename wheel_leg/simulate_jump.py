"""Balance, squat, vertical hop, landing and post-landing balance.

GUI:
    .venv/bin/mjpython -m wheel_leg.simulate_jump \
        --config configs/simulations/jump.json
Headless:
    .venv/bin/python -m wheel_leg.simulate_jump --headless \
        --config configs/simulations/jump.json

The jump is generated only by the six motor actuators. During flight the
motors are released; after both tires re-contact the floor, the standing LQR
controller takes over again.
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

from .balance_control import BalanceController, ReferenceTransitionController
from .config import load_simulation_config
from .geometry import Geometry
from .paths import CURRENT_RESULTS_DIR, GENERATED_MODEL_DIR, project_path


def attitude(d, body_id):
    rotation = d.xmat[body_id].reshape(3, 3)
    return (math.atan2(rotation[2, 1], rotation[2, 2]),
            math.asin(np.clip(-rotation[2, 0], -1, 1)))


def tilt_angle(d, body_id):
    rotation = d.xmat[body_id].reshape(3, 3)
    return math.acos(float(np.clip(rotation[2, 2], -1, 1)))


class JumpExperiment:
    """Run one complete hop and evaluate the landing balance."""

    def __init__(self, args):
        self.args = args
        self.m = mujoco.MjModel.from_xml_path(str(args.model))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        if (m.nq, m.nv, m.nu, m.neq) != (17, 16, 6, 2):
            raise ValueError("Jump requires the six-DOF floating ground model")

        self.body = m.body("chassis").id
        self.floor = m.geom("floor").id
        self.tire_ids = {
            side: m.geom(f"tire_{side}").id for side in ("left", "right")
        }
        self.tire_set = set(self.tire_ids.values())
        self.g = Geometry(*map(float, m.numeric("geometry").data))

        self.standing = BalanceController(m)
        squat_length = float(args.squat_length_mm) / 1000
        self.squat = BalanceController(
            m, squat_length, (self.standing.qref, self.standing.uref))
        self.squat_transition = ReferenceTransitionController(
            self.standing, self.squat, float(args.squat_duration))
        self.nominal_length = self.g.nominal_length

        self.balance_end = float(args.balance_duration)
        self.squat_end = self.balance_end + float(args.squat_duration)
        self.thrust_start = self.squat_end
        self.thrust_end = self.thrust_start + float(args.thrust_duration)

        d.qpos[:] = self.standing.qref
        d.qvel[:] = 0
        d.ctrl[:] = 0
        self._initialize_start_pose()
        mujoco.mj_forward(m, d)

        self.initial_height = float(d.xpos[self.body, 2])
        self.initial_pitch_deg = float(args.pitch_deg)
        self.initial_roll_deg = float(args.roll_deg)
        self.phase = "balance"
        self.takeoff_time = None
        self.landing_time = None
        self._no_wheel_since = None
        self._both_wheel_since = None
        self.failure = None
        self.rows = []
        self.steps = 0
        self.max_height = self.initial_height
        self.max_closure = 0.0
        self.max_nonwheel_contacts = 0
        self.max_flight_nonwheel_contacts = 0
        self.max_startup_nonwheel_contacts = 0
        self.max_knee_limit_excess = 0.0
        self.peaks = np.zeros(m.nu)
        self.saturation_counts = np.zeros(m.nu, dtype=int)
        self.rated_counts = np.zeros(m.nu, dtype=int)
        self.rated = np.array([
            1 if "wheel" in m.actuator(i).name else 3 for i in range(m.nu)
        ])
        self.record()

    def _initialize_start_pose(self):
        """Start slightly above the tires, with the requested small tilt."""
        d, m = self.d, self.m
        roll_q, pitch_q, quat = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            roll_q, np.array([1.0, 0.0, 0.0]),
            math.radians(float(self.args.roll_deg)))
        mujoco.mju_axisAngle2Quat(
            pitch_q, np.array([0.0, 1.0, 0.0]),
            math.radians(float(self.args.pitch_deg)))
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
        d.qpos[2] += float(self.args.drop_height) - lowest

    def _floor_contacts(self):
        sides = set()
        nonwheel = 0
        normal = 0.0
        for index in range(self.d.ncon):
            contact = self.d.contact[index]
            if contact.efc_address < 0 or self.floor not in (
                    contact.geom1, contact.geom2):
                continue
            tire_sides = [side for side, geom_id in self.tire_ids.items()
                          if geom_id in (contact.geom1, contact.geom2)]
            if tire_sides:
                sides.update(tire_sides)
            else:
                nonwheel += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, index, force)
            normal += float(force[0])
        return sides, nonwheel, normal

    def _control(self):
        d = self.d
        if self.landing_time is not None:
            self.phase = "landing_balance"
            return self.standing.control(d)
        if self.takeoff_time is not None:
            self.phase = "flight"
            return np.zeros(self.m.nu)
        if d.time < self.balance_end:
            self.phase = "balance"
            return self.standing.control(d)
        if d.time < self.squat_end:
            self.phase = "squat"
            return self.squat_transition.control(
                d, d.time - self.balance_end)
        if d.time < self.thrust_end:
            self.phase = "thrust"
            elapsed = d.time - self.thrust_start
            ramp = min(1.0, max(0.0, elapsed / float(self.args.thrust_ramp)))
            envelope = 10 * ramp**3 - 15 * ramp**4 + 6 * ramp**5
            command = np.zeros(self.m.nu)
            for side in ("left", "right"):
                command[self.m.actuator(f"hip_motor_{side}").id] = (
                    float(self.args.hip_thrust_torque) * envelope)
                command[self.m.actuator(f"knee_motor_{side}").id] = (
                    -float(self.args.thrust_torque) * envelope)
            return command
        self.phase = "launch_wait"
        return np.zeros(self.m.nu)

    def _update_events(self, sides):
        now = float(self.d.time)
        no_wheel = not sides
        both_wheels = len(sides) == 2
        debounce = float(self.args.contact_debounce)

        if now >= self.thrust_start and self.takeoff_time is None:
            if no_wheel:
                if self._no_wheel_since is None:
                    self._no_wheel_since = now
                if now - self._no_wheel_since >= debounce:
                    self.takeoff_time = self._no_wheel_since
                    self.phase = "flight"
            else:
                self._no_wheel_since = None

        if self.takeoff_time is not None and self.landing_time is None:
            if both_wheels:
                if self._both_wheel_since is None:
                    self._both_wheel_since = now
                if (now - self._both_wheel_since >= debounce
                        and now > self.takeoff_time + debounce):
                    self.landing_time = self._both_wheel_since
                    self.phase = "landing_balance"
            else:
                self._both_wheel_since = None

    def step(self):
        m, d = self.m, self.d
        command = np.clip(
            self._control(), m.actuator_ctrlrange[:, 0],
            m.actuator_ctrlrange[:, 1])
        d.ctrl[:] = command
        self.peaks = np.maximum(self.peaks, np.abs(command))
        self.saturation_counts += np.abs(command) >= m.actuator_ctrlrange[:, 1] * .999
        self.rated_counts += np.abs(command) > self.rated
        previous_time = d.time
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        self.steps += 1

        sides, nonwheel, _ = self._floor_contacts()
        self._update_events(sides)
        if self.takeoff_time is None:
            self.max_startup_nonwheel_contacts = max(
                self.max_startup_nonwheel_contacts, nonwheel)
        else:
            self.max_flight_nonwheel_contacts = max(
                self.max_flight_nonwheel_contacts, nonwheel)
        self.max_nonwheel_contacts = max(self.max_nonwheel_contacts, nonwheel)

        gaps = [np.linalg.norm(
            d.site(f"C_AC_{side}").xpos - d.site(f"C_BC_{side}").xpos)
                for side in ("left", "right")]
        self.max_closure = max(self.max_closure, *gaps)
        self.max_height = max(self.max_height, float(d.xpos[self.body, 2]))
        self.max_knee_limit_excess = max(
            self.max_knee_limit_excess, self._limit_excess())
        roll, pitch = attitude(d, self.body)

        if (d.time <= previous_time or not np.isfinite(d.qpos).all()
                or not np.isfinite(d.qvel).all()):
            self.failure = "numerical_instability"
        elif max(gaps) > .005:
            self.failure = "closure_exceeds_5mm"
        elif np.any(d.warning.number):
            self.failure = "mujoco_warning"
        elif (self.takeoff_time is not None and self.landing_time is None
              and d.time - self.takeoff_time > self.args.max_flight_duration):
            self.failure = "landing_timeout"
        elif (self.landing_time is not None
              and d.time > self.landing_time + .15
              and (max(abs(roll), abs(pitch)) > math.radians(35)
                   or d.xpos[self.body, 2] < .14)):
            self.failure = "fallen_after_landing"
        if self.steps % 10 == 0 or self.failure:
            self.record()
        return self.failure is None

    def record(self):
        m, d = self.m, self.d
        sides, nonwheel, normal = self._floor_contacts()
        roll, pitch = attitude(d, self.body)
        row = {
            "time_s": float(d.time),
            "phase": self.phase,
            "height_m": float(d.xpos[self.body, 2]),
            "vertical_velocity_m_s": float(d.qvel[2]),
            "x_m": float(d.qpos[0]),
            "vx_m_s": float(d.qvel[0]),
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "tilt_deg": math.degrees(tilt_angle(d, self.body)),
            "wheel_contact_count": len(sides),
            "both_wheels_contact": len(sides) == 2,
            "floor_normal_n": normal,
            "nonwheel_ground_contacts": nonwheel,
            "takeoff_detected": self.takeoff_time is not None,
            "landing_detected": self.landing_time is not None,
        }
        row.update({m.actuator(i).name + "_nm": float(d.ctrl[i])
                    for i in range(m.nu)})
        for side in ("left", "right"):
            relative = d.site(f"W_{side}").xpos - d.site(f"O_{side}").xpos
            local = d.xmat[self.body].reshape(3, 3).T @ relative
            row[f"{side}_leg_length_m"] = float(np.linalg.norm(local[[0, 2]]))
        self.rows.append(row)

    def _limit_excess(self):
        excess = 0.0
        for side in ("left", "right"):
            joint_id = self.m.joint(f"knee_drive_{side}").id
            q = self.d.qpos[self.m.joint(f"knee_drive_{side}").qposadr[0]]
            lower, upper = self.m.jnt_range[joint_id]
            excess = max(excess, float(lower - q), float(q - upper), 0.0)
        return excess

    def save(self, status, *, verbose=True):
        d, m, a = self.d, self.m, self.args
        out = a.output
        out.mkdir(parents=True, exist_ok=True)
        self.record()
        tail_start = (self.landing_time + float(a.landing_settle)
                      if self.landing_time is not None else float("inf"))
        tail = [row for row in self.rows if row["time_s"] >= tail_start]
        if not tail and self.landing_time is not None:
            tail = [row for row in self.rows
                    if row["time_s"] >= max(self.landing_time, d.time - 1.0)]
        final_balance = bool(
            self.landing_time is not None and d.time >= tail_start
            and tail
            and all(row["both_wheels_contact"] for row in tail)
            and all(row["nonwheel_ground_contacts"] == 0 for row in tail)
            and max(abs(row["pitch_deg"]) for row in tail) < 2
            and max(abs(row["roll_deg"]) for row in tail) < 2
            and max(abs(row["vertical_velocity_m_s"]) for row in tail) < .05
            and max(abs(row["vx_m_s"]) for row in tail) < .05
            and self.max_closure < .005)
        flight_duration = (None if self.takeoff_time is None
                           or self.landing_time is None else
                           self.landing_time - self.takeoff_time)
        jump_height = self.max_height - self.initial_height
        jump_passed = bool(
            status == "completed" and self.takeoff_time is not None
            and self.landing_time is not None
            and flight_duration >= .05
            and jump_height >= float(a.min_jump_height_mm) / 1000
            and final_balance)
        report = {
            "status": status,
            "jump_passed": jump_passed,
            "balanced_after_landing": final_balance,
            "model": str(a.model.resolve()),
            "scenario_config": (str(a.config_path)
                                 if getattr(a, "config_path", None) else None),
            "controller": "standing LQR + squat LQR + symmetric knee thrust pulse + post-landing LQR",
            "state_feedback": "ideal MuJoCo state; no sensor noise or delay",
            "total_mass_kg": float(m.body_subtreemass[self.body]),
            "duration_s": float(d.time),
            "initial_height_m": self.initial_height,
            "max_height_m": self.max_height,
            "jump_height_m": jump_height,
            "initial_pitch_deg": self.initial_pitch_deg,
            "initial_roll_deg": self.initial_roll_deg,
            "balance_duration_s": float(a.balance_duration),
            "squat_duration_s": float(a.squat_duration),
            "squat_length_mm": float(a.squat_length_mm),
            "thrust_duration_s": float(a.thrust_duration),
            "thrust_torque_nm": float(a.thrust_torque),
            "hip_thrust_torque_nm": float(a.hip_thrust_torque),
            "takeoff_time_s": self.takeoff_time,
            "landing_time_s": self.landing_time,
            "flight_duration_s": flight_duration,
            "landing_vertical_speed_m_s": (
                next((row["vertical_velocity_m_s"] for row in self.rows
                      if self.landing_time is not None
                      and row["time_s"] >= self.landing_time), None)),
            "max_pitch_deg": max(abs(row["pitch_deg"]) for row in self.rows),
            "max_roll_deg": max(abs(row["roll_deg"]) for row in self.rows),
            "max_tilt_deg": max(row["tilt_deg"] for row in self.rows),
            "max_closure_mm": self.max_closure * 1000,
            "max_nonwheel_ground_contacts": self.max_nonwheel_contacts,
            "max_flight_nonwheel_ground_contacts": self.max_flight_nonwheel_contacts,
            "max_startup_nonwheel_ground_contacts": self.max_startup_nonwheel_contacts,
            "max_knee_limit_excess_rad": self.max_knee_limit_excess,
            "max_knee_limit_excess_deg": math.degrees(self.max_knee_limit_excess),
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
                        help="专项跳跃 JSON 配置；命令行参数优先级更高")
    parser.add_argument("--model", type=Path,
                        default=config.get("model", GENERATED_MODEL_DIR / "ground_robot.xml"))
    parser.add_argument("--output", type=Path,
                        default=config.get("output", CURRENT_RESULTS_DIR / "jump"))
    parser.add_argument("--headless", action="store_true",
                        default=config.get("headless", False))
    parser.add_argument("--duration", type=float, default=config.get("duration", 6.0))
    parser.add_argument("--drop-height", type=float,
                        default=config.get("drop_height", .01))
    parser.add_argument("--pitch-deg", type=float, default=config.get("pitch_deg", 3.0))
    parser.add_argument("--roll-deg", type=float, default=config.get("roll_deg", 0.0))
    parser.add_argument("--balance-duration", type=float,
                        default=config.get("balance_duration", 1.0))
    parser.add_argument("--squat-duration", type=float,
                        default=config.get("squat_duration", 1.0))
    parser.add_argument("--thrust-duration", type=float,
                        default=config.get("thrust_duration", .25))
    parser.add_argument("--squat-length-mm", type=float,
                        default=config.get("squat_length_mm", 155.0))
    parser.add_argument("--thrust-torque", type=float,
                        default=config.get("thrust_torque", 3.0))
    parser.add_argument("--hip-thrust-torque", type=float,
                        default=config.get("hip_thrust_torque", 0.0))
    parser.add_argument("--thrust-ramp", type=float,
                        default=config.get("thrust_ramp", .02))
    parser.add_argument("--contact-debounce", type=float,
                        default=config.get("contact_debounce", .03))
    parser.add_argument("--landing-settle", type=float,
                        default=config.get("landing_settle", 1.0))
    parser.add_argument("--min-jump-height-mm", type=float,
                        default=config.get("min_jump_height_mm", 30.0))
    parser.add_argument("--max-flight-duration", type=float,
                        default=config.get("max_flight_duration", 1.5))
    args = parser.parse_args()
    args.config_path = config_path
    args.model = project_path(args.model)
    args.output = project_path(args.output)
    numeric = (args.duration, args.drop_height, args.pitch_deg, args.roll_deg,
               args.balance_duration, args.squat_duration, args.thrust_duration,
               args.squat_length_mm, args.thrust_torque, args.hip_thrust_torque,
               args.thrust_ramp, args.contact_debounce, args.landing_settle,
               args.min_jump_height_mm, args.max_flight_duration)
    if not np.isfinite(numeric).all():
        parser.error("所有跳跃参数必须是有限数")
    if (args.duration <= 0 or args.drop_height < 0
            or args.balance_duration < 0 or args.squat_duration <= 0
            or args.thrust_duration <= 0 or args.squat_length_mm <= 0
            or not 0 < args.thrust_torque <= 7
            or abs(args.hip_thrust_torque) > 7
            or not 0 < args.thrust_ramp <= args.thrust_duration
            or not 0 < args.contact_debounce <= .2
            or args.landing_settle <= 0 or args.min_jump_height_mm <= 0
            or args.max_flight_duration <= 0):
        parser.error("跳跃时序、腿长和力矩参数超出有效范围")
    if args.duration <= args.balance_duration + args.squat_duration + args.thrust_duration:
        parser.error("duration 必须覆盖蹬地和落地后的平衡阶段")
    return args


def main():
    args = parse_args()
    experiment = JumpExperiment(args)
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
    if args.headless and not report["jump_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
