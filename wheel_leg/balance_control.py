"""Numerical standing equilibrium and discrete full-state LQR, motor torques only."""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline

from .geometry import Geometry


# With L=130 mm, the knee-drive joint limits give an approximate geometric
# range of 113–231 mm at beta=0. Keep margin from those hard limits for loaded
# equilibrium continuation and the passive bearing ranges.
MIN_SCHEDULED_LEG_LENGTH_M = 0.120
MAX_SCHEDULED_LEG_LENGTH_M = 0.225


def standing_equilibrium(m, leg_length=None, seed=None):
    """Solve contact penetration, leg lean, passive-joint deflection and torques.

    Soft contact/loop constraints require small static deflections under load.
    Body pitch and nominal leg length are prescribed; all accelerations must
    vanish. This reference solve does not constrain the simulated free base.
    """
    g = Geometry(*map(float, m.numeric("geometry").data))
    length = g.nominal_length if leg_length is None else leg_length
    d = mujoco.MjData(m)
    sides = ("left", "right")
    def state(x):
        mujoco.mj_resetData(m, d)
        penetration, beta = x[:2]
        d.qpos[2] = length*np.cos(beta)+g.wheel_radius+penetration
        q = g.inverse(length, beta)
        q['bearing_A'] += x[2]
        q['bearing_B'] += x[3]
        for side in sides:
            for joint, value in q.items():
                d.qpos[m.joint(f"{joint}_{side}").qposadr[0]] = value
            for name, value in zip(("hip", "knee", "wheel"), x[4:]):
                d.ctrl[m.actuator(f"{name}_motor_{side}").id] = value
        mujoco.mj_forward(m, d)
        wheel_z = np.mean([d.site_xpos[m.site(f'W_{side}').id, 2] for side in sides])
        d.qpos[2] += g.wheel_radius + penetration - wheel_z
        mujoco.mj_forward(m, d)
        return d.qacc.copy()

    def jacobian(x):
        step = 1e-6
        identity = np.eye(len(x))*step
        return np.column_stack([(state(x+dx)-state(x-dx))/(2*step) for dx in identity])

    initial = [-.001, -.02, 0, 0, -.1, -1, 0]
    if seed is not None:
        qprev, uprev = seed
        d.qpos[:] = qprev
        mujoco.mj_forward(m, d)
        hip = qprev[m.joint('hip_left').qposadr[0]]
        knee = qprev[m.joint('knee_drive_left').qposadr[0]]
        a = qprev[m.joint('bearing_A_left').qposadr[0]]
        b = qprev[m.joint('bearing_B_left').qposadr[0]]
        initial = [d.site('W_left').xpos[2]-g.wheel_radius,
                   hip+(knee+np.pi)/2, a+knee, b-knee-np.pi,
                   *[uprev[m.actuator(f'{name}_motor_left').id] for name in ('hip','knee','wheel')]]
    result = least_squares(state, initial,
        bounds=([-.005,-.25,-.02,-.02,-7,-7,-2], [0,.25,.02,.02,7,7,2]),
        xtol=1e-13, ftol=1e-13, gtol=1e-11, max_nfev=500,
        jac=jacobian, x_scale="jac")
    residual = state(result.x)
    if np.linalg.norm(residual) > .01:
        raise RuntimeError(f"Standing equilibrium failed: {result.x}, residual={residual}")
    return d.qpos.copy(), d.ctrl.copy(), float(np.linalg.norm(residual))


class BalanceController:
    def __init__(self, m, leg_length=None, seed=None,
                 horizontal_position_weights=(30.0, 0.0),
                 horizontal_velocity_weights=(10.0, 0.0)):
        self.m = m
        self.qref, self.uref, self.equilibrium_residual = standing_equilibrium(m, leg_length, seed)
        self.vref = np.zeros(m.nv)
        position_weights = np.asarray(horizontal_position_weights, dtype=float)
        velocity_weights = np.asarray(horizontal_velocity_weights, dtype=float)
        if (position_weights.shape != (2,) or velocity_weights.shape != (2,)
                or not np.isfinite(position_weights).all()
                or not np.isfinite(velocity_weights).all()
                or np.any(position_weights < 0)
                or np.any(velocity_weights < 0)):
            raise ValueError("水平位置/速度权重必须是两个非负有限数")
        self.horizontal_position_weights = tuple(position_weights.tolist())
        self.horizontal_velocity_weights = tuple(velocity_weights.tolist())
        d = mujoco.MjData(m)
        d.qpos[:] = self.qref
        d.ctrl[:] = self.uref
        mujoco.mj_forward(m,d)
        n = 2*m.nv
        A, B = np.zeros((n,n)), np.zeros((n,m.nu))
        mujoco.mjd_transitionFD(m,d,1e-6,True,A,B,None,None)
        weights = np.ones(n)
        weights[:6] = [position_weights[0], position_weights[1], 600, 300, 600, 40]
        weights[m.nv:m.nv+6] = [velocity_weights[0], velocity_weights[1], 10, 5, 10, 3]
        for side in ("left","right"):
            for joint in ("hip","knee_drive","bearing_A","bearing_B"):
                i=m.joint(f"{joint}_{side}").dofadr[0]
                weights[i]=50
                weights[m.nv+i]=.3
            i=m.joint(f"wheel_spin_{side}").dofadr[0]
            weights[i]=0
            weights[m.nv+i]=.01
        Q=np.diag(weights)
        R=np.eye(m.nu)*.5
        P=Q.copy()
        converged = False
        for iteration in range(30000):
            K=np.linalg.solve(R+B.T@P@B, B.T@P@A)
            updated=Q+A.T@P@(A-B@K)
            updated=(updated+updated.T)/2
            if np.max(np.abs(updated-P)) < 1e-7:
                converged = True
                break
            P=updated
        if converged:
            self.K=np.linalg.solve(R+B.T@P@B, B.T@P@A)
        else:
            try:
                P=solve_discrete_are(A, B, Q, R)
                self.K=np.linalg.solve(R+B.T@P@B, B.T@P@A)
            except Exception as error:
                raise RuntimeError("LQR Riccati iteration and fallback failed") from error
        self.iterations=iteration+1
        self.eigenvalue_magnitudes=np.sort(np.abs(np.linalg.eigvals(A-B@self.K)))
        # Unpenalized lateral translation and absolute wheel phases are neutral.
        if np.max(self.eigenvalue_magnitudes) > 1+1e-6:
            raise RuntimeError("Linearized closed-loop controller is unstable")
        self.error=np.zeros(m.nv)

    def control(self,d):
        mujoco.mj_differentiatePos(self.m,self.error,1,self.qref,d.qpos)
        u=self.uref-self.K@np.concatenate((self.error,d.qvel-self.vref))
        return np.clip(u,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])


class ReferenceTransitionController:
    """Quintically interpolate two loaded balance references.

    A direct jump from the standing equilibrium to a short-leg equilibrium
    can saturate the knee drive even before the intended launch pulse. This
    small transition controller keeps the reference position, velocity,
    feed-forward torque and LQR gain continuous during a squat.
    """

    def __init__(self, start: BalanceController, end: BalanceController,
                 duration: float):
        if start.m is not end.m or not np.isfinite(duration) or duration <= 0:
            raise ValueError("平衡参考过渡参数无效")
        self.m = start.m
        self.start = start
        self.end = end
        self.duration = float(duration)
        self.qref = start.qref.copy()
        self.vref = np.zeros(self.m.nv)
        self.uref = start.uref.copy()
        self.K = start.K.copy()
        self.error = np.zeros(self.m.nv)
        self._q_delta = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(
            self.m, self._q_delta, 1, start.qref, end.qref)

    def control(self, d, elapsed: float):
        s = min(1.0, max(0.0, elapsed / self.duration))
        ds = ((30*s**2 - 60*s**3 + 30*s**4) / self.duration
              if 0.0 < s < 1.0 else 0.0)
        envelope = 10*s**3 - 15*s**4 + 6*s**5
        self.qref[:] = (1-envelope)*self.start.qref + envelope*self.end.qref
        self.vref[:] = ds*self._q_delta
        self.uref[:] = ((1-envelope)*self.start.uref
                        + envelope*self.end.uref)
        self.K[:] = ((1-envelope)*self.start.K + envelope*self.end.K)
        mujoco.mj_differentiatePos(self.m, self.error, 1, self.qref, d.qpos)
        command = self.uref - self.K @ np.concatenate((
            self.error, d.qvel - self.vref))
        return np.clip(command, self.m.actuator_ctrlrange[:, 0],
                       self.m.actuator_ctrlrange[:, 1])


class SelfRightingController:
    """Start from the prescribed face-down pose, then hand off to balance.

    The initial phase deliberately sends no motor torque. Once the requested
    idle time has elapsed, both knee motors apply a short, ramped positive
    torque. The controller waits for the chassis pitch and pitch rate to settle
    near upright before handing control to the supplied balance controller.
    """

    def __init__(self, m, balance_controller, *, idle=1.0, timeout=4.0,
                 knee_torque=7.0, body_id=None, settle_angle_deg=15.0,
                 settle_rate=3.0, min_righting_time=0.2, ramp_time=0.03,
                 balance_grace=0.5):
        values = (idle, timeout, knee_torque, settle_angle_deg, settle_rate,
                  min_righting_time, ramp_time, balance_grace)
        if (not np.isfinite(values).all() or idle < 0 or timeout <= 0
                or knee_torque <= 0 or knee_torque > 7
                or settle_angle_deg <= 0 or settle_rate <= 0
                or min_righting_time < 0 or ramp_time <= 0
                or balance_grace < 0):
            raise ValueError("自扶正参数必须有限且在电机峰值力矩范围内")
        self.m = m
        self.balance_controller = balance_controller
        self.idle = float(idle)
        self.timeout = float(timeout)
        self.knee_torque = float(knee_torque)
        self.body_id = m.body("chassis").id if body_id is None else body_id
        self.settle_angle = math.radians(settle_angle_deg)
        self.settle_rate = float(settle_rate)
        self.min_righting_time = float(min_righting_time)
        self.ramp_time = float(ramp_time)
        self.balance_grace = float(balance_grace)
        self.phase = "idle"
        self.success = False
        self.transition_time = None

    def _pitch(self, d):
        rotation = d.xmat[self.body_id].reshape(3, 3)
        return math.asin(float(np.clip(-rotation[2, 0], -1, 1)))

    def _zero(self):
        return np.zeros(self.m.nu)

    @property
    def completed(self):
        return self.phase == "balance"

    def balance_ready(self, time):
        return (self.completed and self.transition_time is not None
                and time >= self.transition_time + self.balance_grace)

    def control(self, d):
        if self.completed:
            return self.balance_controller.control(d)
        if d.time < self.idle:
            self.phase = "idle"
            return self._zero()

        self.phase = "righting"
        active_time = float(d.time - self.idle)
        pitch = self._pitch(d)
        pitch_rate = float(d.qvel[4])
        settled = (active_time >= self.min_righting_time
                   and abs(pitch) <= self.settle_angle
                   and abs(pitch_rate) <= self.settle_rate)
        timed_out = active_time >= self.timeout
        if settled or timed_out:
            self.phase = "balance"
            self.success = bool(settled)
            self.transition_time = float(d.time)
            trajectory = getattr(self.balance_controller, "trajectory", None)
            if trajectory is not None:
                # Do not begin cyclic extension during the post-righting
                # balance grace and settling intervals.
                trajectory.start = max(trajectory.start,
                                       self.transition_time + self.balance_grace
                                       + 0.5)
            return self.balance_controller.control(d)

        progress = min(1.0, max(0.0, active_time / self.ramp_time))
        envelope = 10*progress**3 - 15*progress**4 + 6*progress**5
        control = self._zero()
        for side in ("left", "right"):
            control[self.m.actuator(f"knee_motor_{side}").id] = (
                self.knee_torque * envelope)
        return control


class ExtensionTrajectory:
    """Smoothly introduce a periodic length command after landing."""

    def __init__(self, nominal, amplitude=.03, period=8., start=2.):
        values = (nominal, amplitude, period, start)
        if not np.isfinite(values).all() or amplitude <= 0 or period < 4 or start < 2:
            raise ValueError("屈伸幅度须 >0，周期须 >=4 s，落地等待须 >=2 s，参数须有限")
        self.nominal, self.amplitude = nominal, amplitude
        self.period, self.start = period, start

    def sample(self, time):
        elapsed = max(0., time-self.start)
        ramp_time = 2.
        s = min(elapsed/ramp_time, 1.)
        envelope = 10*s**3-15*s**4+6*s**5
        derivative = (30*s*s-60*s**3+30*s**4)/ramp_time if s < 1 else 0.
        omega = 2*np.pi/self.period
        wave = np.sin(omega*elapsed)
        length = self.nominal+self.amplitude*envelope*wave
        speed = self.amplitude*(derivative*wave+envelope*omega*np.cos(omega*elapsed))
        return float(length), float(speed)


class ExtensionBalanceController(BalanceController):
    """Gain scheduling along loaded equilibria with moving-state feedback.

    Cubic reference interpolation gives continuous reference velocities; gains
    use linear interpolation. All online outputs remain motor torques.
    """

    def __init__(self, m, amplitude=.03, period=8., start=2.,
                 horizontal_position_weights=(30.0, 0.0),
                 horizontal_velocity_weights=(10.0, 0.0)):
        self.m = m
        g = Geometry(*map(float, m.numeric("geometry").data))
        self.trajectory = ExtensionTrajectory(g.nominal_length, amplitude, period, start)
        self.horizontal_position_weights = tuple(horizontal_position_weights)
        self.horizontal_velocity_weights = tuple(horizontal_velocity_weights)
        self.lengths = np.linspace(g.nominal_length-amplitude, g.nominal_length+amplitude, 9)
        if (self.lengths[0] < MIN_SCHEDULED_LEG_LENGTH_M
                or self.lengths[-1] > MAX_SCHEDULED_LEG_LENGTH_M):
            raise ValueError("当前屈伸控制范围须位于 120–225 mm 内")
        controllers = [None]*len(self.lengths)
        controllers[4] = BalanceController(
            m,
            horizontal_position_weights=horizontal_position_weights,
            horizontal_velocity_weights=horizontal_velocity_weights)
        for indices in ((3, 2, 1, 0), (5, 6, 7, 8)):
            previous = controllers[4]
            for index in indices:
                controllers[index] = BalanceController(
                    m, self.lengths[index], (previous.qref, previous.uref),
                    horizontal_position_weights=horizontal_position_weights,
                    horizontal_velocity_weights=horizontal_velocity_weights)
                previous = controllers[index]
        refs = np.array([c.qref for c in controllers])
        self.gains = np.array([c.K for c in controllers])
        self.equilibrium_residual = max(c.equilibrium_residual for c in controllers)
        d = mujoco.MjData(m)
        for q in refs:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            for side in ("left", "right"):
                hip = m.joint(f"hip_{side}").qposadr[0]
                bearing = m.joint(f"bearing_B_{side}").qposadr[0]
                wheel = m.joint(f"wheel_spin_{side}").qposadr[0]
                q[wheel] = d.site(f"W_{side}").xpos[0]/g.wheel_radius+q[hip]+q[bearing]
        wheel_addresses = [m.joint(f"wheel_spin_{s}").qposadr[0]
                           for s in ("left", "right")]
        refs[:, wheel_addresses] -= refs[4, wheel_addresses]
        self.positions = CubicSpline(self.lengths, refs, axis=0)
        self.torques = CubicSpline(self.lengths, np.array([c.uref for c in controllers]), axis=0)
        self.error = np.zeros(m.nv)
        self.qref = controllers[4].qref.copy()
        self.vref = np.zeros(m.nv)
        self.uref = controllers[4].uref.copy()
        self.K = controllers[4].K.copy()
        self.update_reference(0.)

    def update_reference(self, time):
        length, speed = self.trajectory.sample(time)
        self.qref[:] = self.positions(length)
        derivative = self.positions(length, 1)*speed
        self.vref[:3] = derivative[:3]
        self.vref[3:6] = 0.
        self.vref[6:] = derivative[7:]
        self.uref[:] = self.torques(length)
        index = int(np.clip(np.searchsorted(self.lengths, length)-1, 0, len(self.lengths)-2))
        fraction = (length-self.lengths[index])/(self.lengths[index+1]-self.lengths[index])
        self.K[:] = (1-fraction)*self.gains[index]+fraction*self.gains[index+1]

    def control(self, d):
        self.update_reference(float(d.time))
        return super().control(d)
