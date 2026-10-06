"""固定髋部轮腿：运动学预览或力矩驱动的闭环动力学。
macOS GUI: .venv/bin/mjpython -m wheel_leg.simulate --test extend
无窗口: .venv/bin/python -m wheel_leg.simulate --headless --test extend --duration 16
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import mujoco
from .geometry import Geometry
from .motor_specs import DM_H6215, DM_J4310_2EC_V11
from .paths import GENERATED_MODEL_DIR, RESULTS_DIR


def ramp(t: float, duration: float = 2.0) -> tuple[float, float]:
    """五次平滑启动，返回包络及其时间导数。"""
    s = min(max(t / duration, 0.0), 1.0)
    f = 10*s**3 - 15*s**4 + 6*s**5
    df = (30*s**2 - 60*s**3 + 30*s**4) / duration if 0 < s < 1 else 0.0
    return f, df


def target(t: float, g: Geometry, test: str, period: float):
    envelope, denvelope = ramp(t)
    omega = 2 * math.pi / period
    wave = envelope * math.sin(omega * t)
    dwave = denvelope * math.sin(omega*t) + envelope*omega*math.cos(omega*t)
    # 随杆长同比缩放，默认腿长约 149~219 mm；摆角 ±15°。
    amp_l = g.length * (0.035 / 0.130) if test in ("extend", "combo") else 0.0
    amp_b = math.radians(15) if test in ("swing", "combo") else 0.0
    length = g.nominal_length + amp_l * wave
    beta = amp_b * wave
    dlength, dbeta = amp_l * dwave, amp_b * dwave
    dalpha = -dlength / math.sqrt(4 * g.length**2 - length**2)
    q = g.inverse(length, beta)
    # 膝电机安装在腿架上，因此 knee_drive 是相对腿架的主动角速度；
    # bearing_A/B 是两个树内被动轴承的速度。
    dq = dict(hip=dbeta-dalpha, knee_drive=2*dalpha,
              bearing_A=-2*dalpha, bearing_B=2*dalpha)
    return length, beta, q, dq


def load_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"模型不存在：{path}")
    m = mujoco.MjModel.from_xml_path(str(path.resolve()))
    d = mujoco.MjData(m)
    numeric_id = m.numeric("geometry").id
    adr = int(m.numeric_adr[numeric_id])
    size = int(m.numeric_size[numeric_id])
    if size != 4:
        raise ValueError("geometry 应包含 L、r、R、H 四个数。")
    g = Geometry(*map(float, m.numeric_data[adr:adr+4]))
    mujoco.mj_forward(m, d)
    return m, d, g


class Experiment:
    def __init__(self, args):
        self.args = args
        self.m, self.d, self.g = load_model(args.model)
        self.m.opt.gravity[:] = (0, 0, -9.81 if args.gravity else 0)
        self.qadr = {name: int(self.m.joint(name).qposadr[0])
                     for name in ("hip", "knee_drive", "bearing_A",
                                  "bearing_B", "wheel_spin")}
        self.vadr = {name: int(self.m.joint(name).dofadr[0]) for name in self.qadr}
        self.act = {name: self.m.actuator(name+"_motor").id
                    for name in ("hip", "knee", "wheel")}
        self.sites = {name: self.m.site(name).id for name in
                      ("O", "W", "C_AC", "C_BC", "target_W")}
        self.leg_body_id = self.m.body("output_link_BCW").id
        self.rows = []
        self.step_count = 0
        self.applied_load = 0.0
        self.warning_counts = np.zeros(len(self.d.warning), dtype=int)
        mujoco.mj_forward(self.m, self.d)
        initial_gap = self.closure_error()
        if initial_gap > 1e-8:
            raise ValueError(f"初始闭环未对齐：{initial_gap:.6g} m；先检查几何和 ref。")
        print(f"MuJoCo {mujoco.__version__}; nq={self.m.nq}, nv={self.m.nv}, "
              f"nu={self.m.nu}, neq={self.m.neq}")
        print("Mode:", "KINEMATIC PREVIEW (not dynamics)" if args.kinematic
              else "TORQUE DYNAMICS; fixed hip; wheel off ground")

    def closure_error(self) -> float:
        return float(np.linalg.norm(self.d.site_xpos[self.sites["C_AC"]]
                                  - self.d.site_xpos[self.sites["C_BC"]]))

    def step(self):
        m, d, a = self.m, self.d, self.args
        dt = float(m.opt.timestep)
        if a.kinematic:
            new_time = float(d.time) + dt
            _, _, q, dq = target(new_time, self.g, a.test, a.period)
            for name in q:
                d.qpos[self.qadr[name]] = q[name]
                d.qvel[self.vadr[name]] = dq[name]
            speed = a.wheel_speed * ramp(new_time)[0]
            d.qpos[self.qadr["wheel_spin"]] += speed * dt
            d.qvel[self.vadr["wheel_spin"]] = speed
            d.time = new_time
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
        else:
            _, _, q, dq = target(float(d.time), self.g, a.test, a.period)
            d.qfrc_applied[:] = 0.0
            self.applied_load = a.load_n * ramp(float(d.time), 3.0)[0]
            if self.applied_load:
                # 等效外载：在轮轴 W 给小腿施加世界 +Z 方向的力。
                mujoco.mj_applyFT(m, d, np.array([0., 0., self.applied_load]),
                                 np.zeros(3), d.site_xpos[self.sites["W"]].copy(),
                                 self.leg_body_id, d.qfrc_applied)
            feedforward = {"hip": 0.0, "knee": 0.0, "wheel": 0.0}
            if a.bias_comp:
                # 膝驱动角是相对腿架的坐标；A/B 被动轴承通过闭环投影。
                b = {name: float(d.qfrc_bias[adr]) for name, adr in self.vadr.items()}
                feedforward["hip"] = b["hip"]
                # da/d(knee_drive)=-1，db/d(knee_drive)=+1。
                feedforward["knee"] = (b["knee_drive"] - b["bearing_A"]
                                         + b["bearing_B"])
                feedforward["wheel"] = b["wheel_spin"]
            for name in ("hip", "knee"):
                joint_name = "hip" if name == "hip" else "knee_drive"
                torque = (a.kp * (q[joint_name] - d.qpos[self.qadr[joint_name]])
                          + a.kd * (dq[joint_name] - d.qvel[self.vadr[joint_name]])
                          + feedforward[name])
                idx = self.act[name]
                d.ctrl[idx] = np.clip(torque, *m.actuator_ctrlrange[idx])
            speed = a.wheel_speed * ramp(float(d.time))[0]
            idx = self.act["wheel"]
            wheel_torque = 0.03 * (speed - d.qvel[self.vadr["wheel_spin"]])
            d.ctrl[idx] = np.clip(wheel_torque + feedforward["wheel"],
                                  *m.actuator_ctrlrange[idx])
            before = float(d.time)
            mujoco.mj_step(m, d)
            if d.time <= before:
                raise RuntimeError("仿真时间重置：可能发生数值发散，检查 MuJoCo 警告。")
            # mj_step 积分后的派生位置需刷新；本例模型小，便于准确记录轨迹。
            mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("状态中出现非有限数值。")
        if self.closure_error() > 0.005:
            raise RuntimeError("闭环误差超过 5 mm，已停止；不要把该状态视为正确机构运动。")
        self.step_count += 1
        # 目标标记位于膝电机外侧连杆平面，避免与机构平面错位。
        length, beta, _, _ = target(float(d.time), self.g, a.test, a.period)
        m.site_pos[self.sites["target_W"]] = (
            length*math.sin(beta), float(d.site_xpos[self.sites["W"]][1]),
            self.g.hip_height-length*math.cos(beta))
        if self.step_count % 10 == 0:
            self.record(length, beta)

    def record(self, target_length: float, target_beta: float):
        d = self.d
        relative = d.site_xpos[self.sites["W"]] - d.site_xpos[self.sites["O"]]
        length = float(np.linalg.norm(relative))
        beta = math.atan2(float(relative[0]), -float(relative[2]))
        theta1 = float(d.qpos[self.qadr["hip"]])
        theta2 = theta1 + float(d.qpos[self.qadr["bearing_B"]])
        crank_theta2 = (theta1 + float(d.qpos[self.qadr["knee_drive"]])
                        + math.pi)
        expected = np.array([target_length*math.sin(target_beta), 0.,
                            -target_length*math.cos(target_beta)])
        expected[1] = relative[1]
        row = dict(time_s=float(d.time), length_target_m=target_length,
                   length_actual_m=length, beta_target_deg=math.degrees(target_beta),
                   beta_actual_deg=math.degrees(beta), wheel_x_m=float(relative[0]),
                   wheel_z_m=float(relative[2]), closure_m=self.closure_error(),
                   wheel_position_error_m=float(np.linalg.norm(relative-expected)),
                   theta1_deg=math.degrees(theta1), theta2_deg=math.degrees(theta2),
                   parallel_error_deg=math.degrees(math.atan2(
                       math.sin(crank_theta2-theta2), math.cos(crank_theta2-theta2))),
                   wheel_speed_relative_rad_s=float(d.qvel[self.vadr["wheel_spin"]]),
                   load_n=self.applied_load)
        for name in self.act:
            row[name+"_torque_nm"] = float(d.ctrl[self.act[name]])
        self.rows.append(row)

    def save(self, status="completed"):
        if not self.rows:
            return
        out = self.args.output
        out.mkdir(parents=True, exist_ok=True)
        with (out / "log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        settled = [r for r in self.rows if r["time_s"] >= 2.0] or self.rows
        summary = dict(status=status, mujoco_version=mujoco.__version__,
                       mode="kinematic_preview" if self.args.kinematic else "torque_dynamics",
                       test=self.args.test, gravity=self.args.gravity,
                       bias_compensation=self.args.bias_comp,
                       base_fixed=True, ground_support_test=False,
                       masses_are_placeholders=True,
                       hip_motor_model=DM_J4310_2EC_V11.name,
                       knee_motor_model=DM_J4310_2EC_V11.name,
                       wheel_motor_model=DM_H6215.name,
                       hip_knee_peak_torque_nm=DM_J4310_2EC_V11.peak_torque_nm,
                       wheel_motor_rated_torque_nm=DM_H6215.rated_torque_nm,
                       wheel_motor_peak_torque_nm=DM_H6215.peak_torque_nm,
                       wheel_motor_torque_limit_source="user supplied peak torque",
                       duration_s=self.rows[-1]["time_s"],
                       max_closure_mm=1000*max(r["closure_m"] for r in self.rows),
                       rms_wheel_error_mm=1000*math.sqrt(sum(
                           r["wheel_position_error_m"]**2 for r in settled)/len(settled)))
        for name, idx in self.act.items():
            summary[name+"_peak_abs_nm"] = max(abs(r[name+"_torque_nm"]) for r in self.rows)
            limit = max(abs(self.m.actuator_ctrlrange[idx]))
            summary[name+"_saturation_fraction"] = sum(
                abs(r[name+"_torque_nm"]) >= 0.999*limit for r in self.rows)/len(self.rows)
        summary["mujoco_warnings"] = {str(i): int(self.d.warning[i].number)
                                       for i in range(len(self.d.warning))
                                       if self.d.warning[i].number}
        (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Log: {(out/'log.csv').resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=GENERATED_MODEL_DIR / "wheel_leg.xml")
    p.add_argument("--test", choices=("hold", "extend", "swing", "combo"), default="extend")
    p.add_argument("--duration", type=float, default=24.0)
    p.add_argument("--period", type=float, default=8.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--kinematic", action="store_true", help="直接设置关节位置，仅检查几何")
    p.add_argument("--gravity", action="store_true")
    p.add_argument("--bias-comp", action="store_true", help="可选：投影后的重力/科氏偏置补偿")
    p.add_argument("--load-n", type=float, default=0.0, help="轮轴处世界向上外力，单位 N")
    p.add_argument("--wheel-speed", type=float, default=0.0, help="轮子相对小腿角速度，rad/s")
    p.add_argument("--kp", type=float, default=25.0)
    p.add_argument("--kd", type=float, default=0.7)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "single_leg")
    args = p.parse_args()
    values = (args.duration, args.period, args.kp, args.kd, args.load_n, args.wheel_speed)
    if not all(math.isfinite(v) for v in values):
        p.error("参数必须是有限数。")
    if min(args.duration, args.period, args.kp) <= 0 or args.kd < 0:
        p.error("duration、period、kp 必须 >0，kd 必须 >=0。")
    if args.kinematic and (args.gravity or args.load_n or args.bias_comp):
        p.error("运动学预览不能用于重力/外载/补偿测试；请删除 --kinematic。")
    return args


def main():
    args = parse_args()
    exp = Experiment(args)
    status = "completed"
    try:
        if args.headless:
            while exp.d.time < args.duration:
                exp.step()
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(exp.m, exp.d) as viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = (-0.04, 0, exp.g.hip_height-0.08)
                    viewer.cam.distance = 0.70
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -8
                    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
                # 每 10 个 1ms 物理步同步一次显示；不要求每步都刷新窗口。
                while viewer.is_running() and exp.d.time < args.duration:
                    begin = time.perf_counter()
                    with viewer.lock():
                        for _ in range(10):
                            if exp.d.time >= args.duration:
                                break
                            exp.step()
                    viewer.sync()
                    time.sleep(max(0.0, 10*exp.m.opt.timestep-(time.perf_counter()-begin)))
                if exp.d.time < args.duration:
                    status = "viewer_closed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        exp.save(status)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if sys.platform == "darwin":
            print("macOS 图形模式须使用 mjpython simulate.py ...；无窗口可用 python。",
                  file=sys.stderr)
        sys.exit(1)
