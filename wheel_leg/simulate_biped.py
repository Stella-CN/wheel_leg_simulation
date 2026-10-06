"""左右双腿固定机身台架的运动学预览和力矩动力学。

GUI:
    .venv/bin/mjpython -m wheel_leg.simulate_biped --test combo
无窗口:
    .venv/bin/python -m wheel_leg.simulate_biped --headless --test extend
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from .motor_specs import DM_H6215, DM_J4310_2EC_V11
from .simulate import load_model, ramp, target
from .paths import GENERATED_MODEL_DIR, RESULTS_DIR


JOINTS = ("hip", "knee_drive", "bearing_A", "bearing_B", "wheel_spin")
SIDES = ("left", "right")


class BipedExperiment:
    def __init__(self, args):
        self.args = args
        self.m, self.d, self.g = load_model(args.model)
        self.m.opt.gravity[:] = (0, 0, -9.81 if args.gravity else 0)
        self.qadr = {
            side: {name: int(self.m.joint(f"{name}_{side}").qposadr[0])
                   for name in JOINTS}
            for side in SIDES
        }
        self.vadr = {
            side: {name: int(self.m.joint(f"{name}_{side}").dofadr[0])
                   for name in JOINTS}
            for side in SIDES
        }
        self.act = {
            side: {
                "hip": self.m.actuator(f"hip_motor_{side}").id,
                "knee": self.m.actuator(f"knee_motor_{side}").id,
                "wheel": self.m.actuator(f"wheel_motor_{side}").id,
            }
            for side in SIDES
        }
        self.sites = {
            side: {name: self.m.site(f"{name}_{side}").id
                   for name in ("O", "W", "C_AC", "C_BC", "target_W")}
            for side in SIDES
        }
        self.leg_body_id = {
            side: self.m.body(f"output_link_BCW_{side}").id for side in SIDES
        }
        self.rows: list[dict[str, float]] = []
        self.step_count = 0
        self.applied_load = {side: 0.0 for side in SIDES}
        mujoco.mj_forward(self.m, self.d)
        initial_gaps = {side: self.closure_error(side) for side in SIDES}
        if max(initial_gaps.values()) > 1e-8:
            raise ValueError(f"初始双腿闭环未对齐：{initial_gaps}")
        print(f"MuJoCo {mujoco.__version__}; nq={self.m.nq}, nv={self.m.nv}, "
              f"nu={self.m.nu}, neq={self.m.neq}")
        print("Mode:", "KINEMATIC PREVIEW (not dynamics)" if args.kinematic
              else "TORQUE DYNAMICS; fixed chassis; wheels off ground")

    def closure_error(self, side: str) -> float:
        ids = self.sites[side]
        return float(np.linalg.norm(self.d.site_xpos[ids["C_AC"]]
                                    - self.d.site_xpos[ids["C_BC"]]))

    def step(self):
        m, d, a = self.m, self.d, self.args
        dt = float(m.opt.timestep)
        if a.kinematic:
            new_time = float(d.time) + dt
            _, _, q, dq = target(new_time, self.g, a.test, a.period)
            for side in SIDES:
                for name in q:
                    d.qpos[self.qadr[side][name]] = q[name]
                    d.qvel[self.vadr[side][name]] = dq[name]
                speed = a.wheel_speed * ramp(new_time)[0]
                d.qpos[self.qadr[side]["wheel_spin"]] += speed * dt
                d.qvel[self.vadr[side]["wheel_spin"]] = speed
            d.time = new_time
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
        else:
            _, _, q, dq = target(float(d.time), self.g, a.test, a.period)
            d.qfrc_applied[:] = 0.0
            load = a.load_n * ramp(float(d.time), 3.0)[0]
            for side in SIDES:
                self.applied_load[side] = load
                if load:
                    mujoco.mj_applyFT(
                        m, d, np.array([0., 0., load]), np.zeros(3),
                        d.site_xpos[self.sites[side]["W"]].copy(),
                        self.leg_body_id[side], d.qfrc_applied)

            for side in SIDES:
                ids = self.qadr[side]
                vels = self.vadr[side]
                feedforward = {"hip": 0.0, "knee": 0.0, "wheel": 0.0}
                if a.bias_comp:
                    b = {name: float(d.qfrc_bias[vels[name]]) for name in JOINTS}
                    feedforward["hip"] = b["hip"]
                    feedforward["knee"] = (
                        b["knee_drive"] - b["bearing_A"] + b["bearing_B"])
                    feedforward["wheel"] = b["wheel_spin"]
                for command_name, joint_name in (
                        ("hip", "hip"), ("knee", "knee_drive")):
                    torque = (
                        a.kp * (q[joint_name] - d.qpos[ids[joint_name]])
                        + a.kd * (dq[joint_name] - d.qvel[vels[joint_name]])
                        + feedforward[command_name])
                    actuator_id = self.act[side][command_name]
                    d.ctrl[actuator_id] = np.clip(
                        torque, *m.actuator_ctrlrange[actuator_id])
                speed = a.wheel_speed * ramp(float(d.time))[0]
                actuator_id = self.act[side]["wheel"]
                wheel_torque = 0.03 * (
                    speed - d.qvel[vels["wheel_spin"]])
                d.ctrl[actuator_id] = np.clip(
                    wheel_torque + feedforward["wheel"],
                    *m.actuator_ctrlrange[actuator_id])
            before = float(d.time)
            mujoco.mj_step(m, d)
            if d.time <= before:
                raise RuntimeError("仿真时间重置：可能发生数值发散。")
            mujoco.mj_forward(m, d)

        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("状态中出现非有限数值。")
        if max(self.closure_error(side) for side in SIDES) > 0.005:
            raise RuntimeError("双腿闭环误差超过 5 mm，已停止。")
        self.step_count += 1
        length, beta, _, _ = target(float(d.time), self.g, a.test, a.period)
        for side in SIDES:
            ids = self.sites[side]
            m.site_pos[ids["target_W"]] = (
                length * math.sin(beta),
                float(d.site_xpos[ids["W"]][1]),
                self.g.hip_height - length * math.cos(beta))
        if self.step_count % 10 == 0:
            self.record(length, beta)

    def record(self, target_length: float, target_beta: float):
        row: dict[str, float] = {"time_s": float(self.d.time)}
        for side in SIDES:
            d, ids, qadr, vadr = (self.d, self.sites[side],
                                   self.qadr[side], self.vadr[side])
            relative = d.site_xpos[ids["W"]] - d.site_xpos[ids["O"]]
            theta_ob = float(d.qpos[qadr["hip"]])
            theta_bw = theta_ob + float(d.qpos[qadr["bearing_B"]])
            theta_oa = theta_ob + float(d.qpos[qadr["knee_drive"]])
            expected = np.array([
                target_length * math.sin(target_beta), relative[1],
                -target_length * math.cos(target_beta)])
            prefix = f"{side}_"
            row[prefix + "length_m"] = float(np.linalg.norm(relative))
            row[prefix + "beta_deg"] = math.degrees(
                math.atan2(float(relative[0]), -float(relative[2])))
            row[prefix + "closure_m"] = self.closure_error(side)
            row[prefix + "wheel_error_m"] = float(
                np.linalg.norm(relative - expected))
            row[prefix + "parallel_error_deg"] = math.degrees(math.atan2(
                math.sin(theta_oa + math.pi - theta_bw),
                math.cos(theta_oa + math.pi - theta_bw)))
            row[prefix + "wheel_speed_rad_s"] = float(
                d.qvel[vadr["wheel_spin"]])
            row[prefix + "load_n"] = self.applied_load[side]
            for command_name, actuator_id in self.act[side].items():
                row[prefix + command_name + "_torque_nm"] = float(
                    d.ctrl[actuator_id])
        self.rows.append(row)

    def save(self, status="completed"):
        if not self.rows:
            return
        out = self.args.output
        out.mkdir(parents=True, exist_ok=True)
        with (out / "log.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        settled = [row for row in self.rows if row["time_s"] >= 2.0] or self.rows
        closure_values = [row[f"{side}_closure_m"] for row in self.rows
                          for side in SIDES]
        error_values = [row[f"{side}_wheel_error_m"] for row in settled
                        for side in SIDES]
        summary = dict(
            status=status,
            mujoco_version=mujoco.__version__,
            mode="kinematic_preview" if self.args.kinematic else "torque_dynamics",
            test=self.args.test,
            legs=2,
            chassis_geometry_m=[0.2, 0.15, 0.1],
            gravity=self.args.gravity,
            bias_compensation=self.args.bias_comp,
            base_fixed=True,
            ground_support_test=False,
            masses_are_placeholders=True,
            hip_motor_model=DM_J4310_2EC_V11.name,
            knee_motor_model=DM_J4310_2EC_V11.name,
            wheel_motor_model=DM_H6215.name,
            hip_knee_peak_torque_nm=DM_J4310_2EC_V11.peak_torque_nm,
            wheel_motor_rated_torque_nm=DM_H6215.rated_torque_nm,
            wheel_motor_peak_torque_nm=DM_H6215.peak_torque_nm,
            duration_s=self.rows[-1]["time_s"],
            max_closure_mm=1000 * max(closure_values),
            rms_wheel_error_mm=1000 * math.sqrt(
                sum(value * value for value in error_values) /
                len(error_values)),
        )
        for side in SIDES:
            for command_name, actuator_id in self.act[side].items():
                values = [row[f"{side}_{command_name}_torque_nm"]
                          for row in self.rows]
                summary[f"{side}_{command_name}_peak_abs_nm"] = max(
                    abs(value) for value in values)
                limit = max(abs(self.m.actuator_ctrlrange[actuator_id]))
                summary[f"{side}_{command_name}_saturation_fraction"] = sum(
                    abs(value) >= 0.999 * limit for value in values) / len(values)
        summary["mujoco_warnings"] = {
            str(index): int(self.d.warning[index].number)
            for index in range(len(self.d.warning))
            if self.d.warning[index].number
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Log: {(out / 'log.csv').resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=GENERATED_MODEL_DIR / "biped_wheel_leg.xml")
    parser.add_argument("--test", choices=("hold", "extend", "swing", "combo"),
                        default="extend")
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--period", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--kinematic", action="store_true")
    parser.add_argument("--gravity", action="store_true")
    parser.add_argument("--bias-comp", action="store_true")
    parser.add_argument("--load-n", type=float, default=0.0)
    parser.add_argument("--wheel-speed", type=float, default=0.0)
    parser.add_argument("--kp", type=float, default=25.0)
    parser.add_argument("--kd", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "biped_extend")
    args = parser.parse_args()
    values = (args.duration, args.period, args.kp, args.kd, args.load_n,
              args.wheel_speed)
    if not all(math.isfinite(value) for value in values):
        parser.error("参数必须是有限数。")
    if min(args.duration, args.period, args.kp) <= 0 or args.kd < 0:
        parser.error("duration、period、kp 必须 >0，kd 必须 >=0。")
    if args.kinematic and (args.gravity or args.load_n or args.bias_comp):
        parser.error("运动学预览不能用于重力/外载/补偿测试。")
    return args


def main():
    args = parse_args()
    experiment = BipedExperiment(args)
    status = "completed"
    try:
        if args.headless:
            while experiment.d.time < args.duration:
                experiment.step()
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(experiment.m, experiment.d) as viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = (-0.04, 0, experiment.g.hip_height - 0.08)
                    viewer.cam.distance = 0.85
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -8
                    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
                while viewer.is_running() and experiment.d.time < args.duration:
                    begin = time.perf_counter()
                    with viewer.lock():
                        for _ in range(10):
                            if experiment.d.time >= args.duration:
                                break
                            experiment.step()
                    viewer.sync()
                    time.sleep(max(
                        0.0,
                        10 * experiment.m.opt.timestep -
                        (time.perf_counter() - begin)))
                if experiment.d.time < args.duration:
                    status = "viewer_closed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        experiment.save(status)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if sys.platform == "darwin":
            print("macOS 图形模式须使用 mjpython simulate_biped.py ...；"
                  "无窗口可用 python。", file=sys.stderr)
        sys.exit(1)
