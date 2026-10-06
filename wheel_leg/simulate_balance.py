"""Free-floating landing, self-righting startup and balance.

macOS GUI: .venv/bin/mjpython -m wheel_leg.simulate_balance
Headless:  .venv/bin/python -m wheel_leg.simulate_balance --headless
Self-right: .venv/bin/mjpython -m wheel_leg.simulate_balance --startup self-right
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

from .balance_control import (BalanceController, ExtensionBalanceController,
                              SelfRightingController)
from .config import load_simulation_config
from .geometry import Geometry
from .paths import (CURRENT_RESULTS_DIR, GENERATED_MODEL_DIR, PROJECT_ROOT,
                    project_path)


def attitude(d, body_id):
    rotation = d.xmat[body_id].reshape(3,3)
    return (math.atan2(rotation[2,1], rotation[2,2]),
            math.asin(np.clip(-rotation[2,0],-1,1)))


def tilt_angle(d, body_id):
    """Angle between the chassis +Z axis and world +Z, without Euler singularity."""
    rotation = d.xmat[body_id].reshape(3,3)
    return math.acos(float(np.clip(rotation[2,2], -1, 1)))


class BalanceExperiment:
    def __init__(self, args):
        self.args = args
        self.m = mujoco.MjModel.from_xml_path(str(args.model))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        if (m.nq,m.nv,m.nu,m.neq) != (17,16,6,2):
            raise ValueError("Balance requires the six-DOF floating ground model")
        self.motion = getattr(args, 'motion', 'hold')
        self.startup = getattr(args, 'startup', 'balance')
        if self.startup not in ('balance', 'self-right'):
            raise ValueError("startup must be 'balance' or 'self-right'")
        if self.startup == 'self-right' and args.disable_feedback:
            raise ValueError("自扶正后必须启用平衡反馈")
        self.body = m.body("chassis").id
        self.floor = m.geom("floor").id
        self.tires = {m.geom(f"tire_{side}").id for side in ("left","right")}
        if self.motion == 'extend':
            if args.disable_feedback:
                raise ValueError("屈伸模式需要启用平衡反馈")
            self.balance_controller = ExtensionBalanceController(
                m, args.leg_amplitude_mm/1000, args.leg_period, args.motion_start)
        else:
            self.balance_controller = BalanceController(m)
        if self.startup == 'self-right':
            self.self_righting = SelfRightingController(
                m, self.balance_controller,
                idle=getattr(args, 'laying_idle', 1.0),
                timeout=getattr(args, 'righting_timeout', 4.0),
                knee_torque=getattr(args, 'righting_torque', 7.0),
                body_id=self.body)
            self.controller = self.self_righting
        else:
            self.self_righting = None
            self.controller = self.balance_controller
        self.nominal_length = Geometry(*map(float, m.numeric('geometry').data)).nominal_length
        d.qpos[:] = self.balance_controller.qref
        d.qvel[:] = 0
        d.ctrl[:] = 0
        if self.startup == 'self-right':
            self._initialize_laying_pose(getattr(args, 'lying_pitch_deg', 90.0))
            self.initial_pitch_deg = float(getattr(args, 'lying_pitch_deg', 90.0))
        else:
            self.initial_pitch_deg = float(args.pitch_deg)
            roll_q, pitch_q, quat = np.zeros(4), np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(roll_q,np.array([1.,0,0]),math.radians(args.roll_deg))
            mujoco.mju_axisAngle2Quat(pitch_q,np.array([0.,1,0]),math.radians(args.pitch_deg))
            mujoco.mju_mulQuat(quat,pitch_q,roll_q)
            d.qpos[3:7] = quat
            mujoco.mj_forward(m,d)
            lowest = min(d.geom_xpos[i,2] - (
                m.geom_size[i,0]*np.sqrt(max(0,1-d.geom_xmat[i].reshape(3,3)[2,2]**2))
                +m.geom_size[i,1]*abs(d.geom_xmat[i].reshape(3,3)[2,2])) for i in self.tires)
            d.qpos[2] += args.drop_height-lowest
        mujoco.mj_forward(m,d)
        self.rows = []
        self.steps = 0
        self.failure = None
        self.max_closure = 0.
        self.max_nonwheel_contacts = 0
        self.max_startup_nonwheel_contacts = 0
        self.peaks = np.zeros(m.nu)
        self.saturation_counts = np.zeros(m.nu,dtype=int)
        self.rated_counts = np.zeros(m.nu,dtype=int)
        self.rated = np.array([1 if 'wheel' in m.actuator(i).name else 3 for i in range(m.nu)])
        self.first_contact = None
        self.record()

    def _initialize_laying_pose(self, pitch_deg):
        """Place the standing joint configuration on its front face, unpowered."""
        d, m = self.d, self.m
        d.qpos[0:2] = 0
        d.qvel[:] = 0
        d.ctrl[:] = 0
        pitch_q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            pitch_q, np.array([0., 1., 0.]), math.radians(pitch_deg))
        d.qpos[3:7] = pitch_q
        mujoco.mj_forward(m, d)
        active_geoms = [i for i in range(m.ngeom) if i != self.floor]
        lowest = min(float(d.geom_xpos[i, 2] - m.geom_rbound[i])
                     for i in active_geoms)
        d.qpos[2] -= lowest
        mujoco.mj_forward(m, d)

    def _balance_ready(self):
        return (self.self_righting is None
                or self.self_righting.balance_ready(float(self.d.time)))

    def step(self):
        m,d,a = self.m,self.d,self.args
        if self.self_righting is not None:
            d.ctrl[:] = self.controller.control(d)
        else:
            d.ctrl[:] = (self.controller.uref if a.disable_feedback
                         else self.controller.control(d))
        # This is the explicitly requested test disturbance, not a balancing force.
        d.xfrc_applied[:] = 0
        if a.push_start <= d.time < a.push_start+a.push_duration:
            d.xfrc_applied[self.body,0] = a.push_n
        limits = m.actuator_ctrlrange[:,1]
        self.peaks = np.maximum(self.peaks,np.abs(d.ctrl))
        self.saturation_counts += np.abs(d.ctrl) >= limits*.999
        self.rated_counts += np.abs(d.ctrl) > self.rated
        previous_time = d.time
        mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        self.steps += 1
        gaps = [np.linalg.norm(d.site(f"C_AC_{s}").xpos-d.site(f"C_BC_{s}").xpos)
                for s in ("left","right")]
        self.max_closure = max(self.max_closure, *gaps)
        contacts = [c for c in d.contact if self.floor in (c.geom1,c.geom2)]
        nonwheel = sum(not (c.geom1 in self.tires or c.geom2 in self.tires) for c in contacts)
        if self._balance_ready():
            self.max_nonwheel_contacts = max(self.max_nonwheel_contacts,nonwheel)
        else:
            self.max_startup_nonwheel_contacts = max(
                self.max_startup_nonwheel_contacts, nonwheel)
        if self.first_contact is None and contacts:
            self.first_contact = float(d.time)
        roll,pitch=attitude(d,self.body)
        tilt = tilt_angle(d, self.body)
        if d.time <= previous_time or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self.failure = "numerical_instability"
        elif (self._balance_ready()
              and (max(abs(roll),abs(pitch)) > math.radians(35)
                   or d.xpos[self.body,2] < .14)):
            self.failure = "fallen"
        elif max(gaps) > .005:
            self.failure = "closure_exceeds_5mm"
        elif self._balance_ready() and nonwheel:
            self.failure = "nonwheel_ground_contact"
        elif np.any(d.warning.number):
            self.failure = "mujoco_warning"
        if self.steps % 10 == 0 or self.failure:
            self.record()
        return self.failure is None

    def record(self):
        m,d=self.m,self.d
        roll,pitch=attitude(d,self.body)
        tilt = tilt_angle(d, self.body)
        normal = 0.
        wheel_contacts=0
        contacted=set()
        for i,c in enumerate(d.contact):
            if self.floor in (c.geom1,c.geom2):
                force=np.zeros(6)
                mujoco.mj_contactForce(m,d,i,force)
                normal += force[0]
                for tire in self.tires.intersection((c.geom1,c.geom2)):
                    contacted.add(tire)
                    wheel_contacts += 1
        row={"time_s":float(d.time), "x_m":float(d.qpos[0]), "y_m":float(d.qpos[1]),
             "height_m":float(d.qpos[2]), "roll_deg":math.degrees(roll),
             "pitch_deg":math.degrees(pitch), "vx_m_s":float(d.qvel[0]),
             "tilt_deg":math.degrees(tilt),
             "floor_normal_n":float(normal), "wheel_contacts":wheel_contacts,
             "both_wheels_contact":len(contacted)==2}
        if self.self_righting is None:
            row["startup_phase"] = "balance"
        elif self.self_righting.balance_ready(float(d.time)):
            row["startup_phase"] = "balance"
        else:
            row["startup_phase"] = self.self_righting.phase
        row.update({m.actuator(i).name+"_nm":float(d.ctrl[i]) for i in range(m.nu)})
        length = (self.balance_controller.trajectory.sample(float(d.time))[0]
                  if self.motion == 'extend' else self.nominal_length)
        row['target_leg_length_m'] = length
        row['target_height_m'] = float(self.balance_controller.positions(length)[2]
                                       if self.motion == 'extend'
                                       else self.balance_controller.qref[2])
        for side in ('left','right'):
            relative = d.site(f'W_{side}').xpos-d.site(f'O_{side}').xpos
            local = d.xmat[self.body].reshape(3,3).T @ relative
            row[f'{side}_leg_length_m'] = float(np.linalg.norm(local[[0,2]]))
        self.rows.append(row)

    def save(self, status, *, verbose=True):
        m,d,a=self.m,self.d,self.args
        out=a.output
        out.mkdir(parents=True,exist_ok=True)
        self.record()
        tail=[r for r in self.rows if r['time_s'] >= max(0,d.time-2)]
        finished=status=="completed" and d.time >= a.duration-m.opt.timestep
        # Require two quiet seconds after the push, not merely surviving a run.
        recovered=(finished and d.time >= max(3,a.push_start+a.push_duration+2 if a.push_n else 3)
                   and max(abs(r['pitch_deg']) for r in tail)<2
                   and max(abs(r['roll_deg']) for r in tail)<2
                   and max(abs(r['vx_m_s']) for r in tail)<.05
                   and all(r['both_wheels_contact'] for r in tail)
                   and self.max_nonwheel_contacts==0 and self.max_closure<.005)
        self_righting_passed = (self.self_righting is None
                                or (self.self_righting.success
                                    and self.self_righting.transition_time is not None))
        report={"status":status, "balanced":bool(recovered),
            "model":str(a.model.resolve()),
            "scenario_config": (str(a.config_path) if getattr(a, 'config_path', None)
                                 else None),
            "mass_parameters":"engineering_estimates_not_measured",
            "controller":"full-state discrete LQR + static motor feedforward",
            "state_feedback":"ideal MuJoCo state; no sensor noise or delay",
            "base_free_dof":6, "gravity_m_s2":m.opt.gravity.tolist(),
            "total_mass_kg":float(m.body_subtreemass[self.body]),
            "duration_s":float(d.time), "drop_height_m":a.drop_height,
            "initial_pitch_deg":self.initial_pitch_deg,"initial_roll_deg":a.roll_deg,
            "push_n":a.push_n,"push_start_s":a.push_start,"push_duration_s":a.push_duration,
            "feedback_enabled":not a.disable_feedback,"first_contact_s":self.first_contact,
            "max_pitch_deg":max(abs(r['pitch_deg']) for r in self.rows),
            "max_roll_deg":max(abs(r['roll_deg']) for r in self.rows),
            "max_tilt_deg":max(r['tilt_deg'] for r in self.rows),
            "final_pitch_deg":self.rows[-1]['pitch_deg'],"final_roll_deg":self.rows[-1]['roll_deg'],
            "final_tilt_deg":self.rows[-1]['tilt_deg'],
            "final_x_m":self.rows[-1]['x_m'],"final_vx_m_s":self.rows[-1]['vx_m_s'],
            "final_height_m":float(d.qpos[2]),
            "tail_mean_floor_normal_n":float(np.mean([r['floor_normal_n'] for r in tail])),
            "tail_both_wheels_contact_fraction":float(np.mean([r['both_wheels_contact'] for r in tail])),
            "max_closure_mm":self.max_closure*1000,
            "max_nonwheel_ground_contacts":self.max_nonwheel_contacts,
            "max_startup_nonwheel_ground_contacts":self.max_startup_nonwheel_contacts,
            "equilibrium_acceleration_residual":self.balance_controller.equilibrium_residual,
            "peak_torques_nm":dict(zip([m.actuator(i).name for i in range(m.nu)],self.peaks.tolist())),
            "saturation_fractions":(self.saturation_counts/max(self.steps,1)).tolist(),
            "above_rated_torque_fractions":(self.rated_counts/max(self.steps,1)).tolist(),
            "mujoco_warnings":{str(i):int(w.number) for i,w in enumerate(d.warning) if w.number}}
        report['motion'] = self.motion
        report['startup'] = self.startup
        report['self_righting_passed'] = bool(self_righting_passed)
        if self.self_righting is not None:
            report.update({
                'lying_pitch_deg': float(getattr(a, 'lying_pitch_deg', 90.0)),
                'laying_idle_s': self.self_righting.idle,
                'righting_timeout_s': self.self_righting.timeout,
                'righting_transition_s': self.self_righting.transition_time,
                'righting_phase': self.self_righting.phase,
                'righting_success': bool(self.self_righting.success),
            })
        if self.motion == 'extend':
            trajectory = self.balance_controller.trajectory
            active = [r for r in self.rows if r['time_s'] >= trajectory.start]
            errors = [r[f'{s}_leg_length_m']-r['target_leg_length_m']
                      for r in active for s in ('left','right')]
            max_error = max(map(abs,errors),default=float('inf'))
            ranges = {s: [min(r[f'{s}_leg_length_m'] for r in active),
                          max(r[f'{s}_leg_length_m'] for r in active)]
                      for s in ('left','right')} if active else {}
            full_cycle = d.time >= trajectory.start+2+trajectory.period
            motion_passed = (recovered and full_cycle and bool(active) and max_error < .003
                             and all(r['both_wheels_contact'] for r in active)
                             and all(abs(r['pitch_deg']) < 5 and abs(r['roll_deg']) < 5 for r in active)
                             and all(hi-lo > 1.5*trajectory.amplitude for lo,hi in ranges.values()))
            report.update({
                'motion_passed': bool(motion_passed),
                'leg_amplitude_mm': trajectory.amplitude*1000,
                'leg_period_s': trajectory.period,
                'motion_start_s': trajectory.start,
                'complete_cycle_tested': bool(full_cycle),
                'leg_length_range_m': ranges,
                'max_leg_tracking_error_mm': max_error*1000 if errors else None,
                'rms_leg_tracking_error_mm': float(np.sqrt(np.mean(np.square(errors)))*1000) if errors else None,
                'motion_max_pitch_deg': max((abs(r['pitch_deg']) for r in active),default=None),
                'motion_max_roll_deg': max((abs(r['roll_deg']) for r in active),default=None),
                'motion_both_wheels_contact_fraction': float(np.mean([r['both_wheels_contact'] for r in active])) if active else None,
            })
            report['controller'] = 'length-scheduled LQR + loaded equilibrium and reference velocity'
        if self.self_righting is not None:
            report['controller'] = ('self-righting knee torque sequence + '
                                    + report['controller'])
        with (out/'log.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=self.rows[0].keys())
            writer.writeheader();writer.writerows(self.rows)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+"\n")
        if verbose:
            print(json.dumps(report,indent=2))
        return report


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path)
    preliminary_args, _ = preliminary.parse_known_args()
    config, config_path = load_simulation_config(preliminary_args.config)
    p.add_argument('--config', type=Path, default=preliminary_args.config,
                   help='专项仿真 JSON 配置；命令行参数优先级更高')
    p.add_argument('--model',type=Path,default=config.get('model',
                   GENERATED_MODEL_DIR/'ground_robot.xml'))
    p.add_argument('--output',type=Path,default=config.get('output',
                   CURRENT_RESULTS_DIR/'balance'))
    p.add_argument('--headless',action='store_true', default=config.get('headless', False))
    p.add_argument('--duration',type=float,default=config.get('duration', 20))
    p.add_argument('--drop-height',type=float,default=config.get('drop_height', .01))
    p.add_argument('--pitch-deg',type=float,default=config.get('pitch_deg', 3))
    p.add_argument('--roll-deg',type=float,default=config.get('roll_deg', 0))
    p.add_argument('--push-n',type=float,default=config.get('push_n', 3))
    p.add_argument('--push-start',type=float,default=config.get('push_start', 5))
    p.add_argument('--push-duration',type=float,default=config.get('push_duration', .2))
    p.add_argument('--disable-feedback',action='store_true',
                   default=config.get('disable_feedback', False))
    p.add_argument('--startup',choices=('balance','self-right'),
                   default=config.get('startup', 'balance'),
                   help='balance 直接落地平衡；self-right 从无动力俯卧姿态自扶正')
    p.add_argument('--lying-pitch-deg',type=float,default=config.get('lying_pitch_deg', 90),
                   help='self-right 初始躺姿俯仰角，默认 90°')
    p.add_argument('--laying-idle',type=float,default=config.get('laying_idle', 1.0),
                   help='self-right 启动前保持无动力躺地的秒数')
    p.add_argument('--righting-timeout',type=float,default=config.get('righting_timeout', 4.0),
                   help='自扶正阶段最长秒数')
    p.add_argument('--righting-torque',type=float,default=config.get('righting_torque', 7.0),
                   help='双膝自扶正力矩，范围 0–7 N m')
    p.add_argument('--motion',choices=('hold','extend'),default=config.get('motion', 'hold'))
    p.add_argument('--leg-amplitude-mm',type=float,default=config.get('leg_amplitude_mm', 30),
                   help='同步屈伸的腿长半幅，默认 ±30 mm')
    p.add_argument('--leg-period',type=float,default=config.get('leg_period', 8),
                   help='屈伸周期秒数，至少 4 s')
    p.add_argument('--motion-start',type=float,default=config.get('motion_start', 2),
                   help='落地后开始平滑引入屈伸的时刻，至少 2 s')
    a=p.parse_args()
    a.config_path = config_path
    a.model = project_path(a.model)
    a.output = project_path(a.output)
    vals=(a.duration,a.drop_height,a.pitch_deg,a.roll_deg,a.push_n,a.push_start,
          a.push_duration,a.lying_pitch_deg,a.laying_idle,a.righting_timeout,
          a.righting_torque)
    if not all(math.isfinite(v) for v in vals) or a.duration<=0 or min(a.drop_height,a.push_start,a.push_duration)<0:
        p.error('Numeric arguments must be finite; duration positive; height/times nonnegative')
    if a.laying_idle < 0 or a.righting_timeout <= 0 or not 0 < a.righting_torque <= 7:
        p.error('self-right timing must be nonnegative and torque must be in (0, 7] N m')
    if a.startup == 'self-right' and abs(a.lying_pitch_deg) < 70:
        p.error('self-right lying pitch must be at least 70 degrees from upright')
    if a.startup == 'self-right' and a.disable_feedback:
        p.error('self-right requires feedback after the righting phase')
    return a


def main():
    a=parse_args()
    experiment=BalanceExperiment(a)
    status='completed'
    try:
        if a.headless:
            while experiment.d.time<a.duration:
                if not experiment.step():
                    status=experiment.failure;break
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(experiment.m,experiment.d) as viewer:
                viewer.cam.lookat[:]=(0,0,.2)
                viewer.cam.distance=1.1
                viewer.cam.azimuth=135
                viewer.cam.elevation=-20
                while viewer.is_running() and experiment.d.time<a.duration:
                    start=time.perf_counter()
                    with viewer.lock():
                        for _ in range(10):
                            if experiment.d.time>=a.duration or not experiment.step():break
                    viewer.sync()
                    if experiment.failure:
                        status=experiment.failure;break
                    time.sleep(max(0,.01-(time.perf_counter()-start)))
                if not experiment.failure and experiment.d.time<a.duration:status='viewer_closed'
    except KeyboardInterrupt:
        status='interrupted'
    except Exception:
        status='error'
        raise
    finally:
        report=experiment.save(status)
    if a.headless and (not report['balanced']
                       or not report.get('motion_passed',True)
                       or not report.get('self_righting_passed',True)):
        raise SystemExit(1)


if __name__=='__main__':
    main()
