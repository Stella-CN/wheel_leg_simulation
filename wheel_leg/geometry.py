"""同轴髋/膝电机平行四边形轮腿的纯 Python 几何。

坐标约定：X 向前、Y 向外侧、Z 向上；所有长度使用 SI 单位。

平行四边形按 O-A-C-B-O 建模：OA 是膝电机转子驱动的主动臂，OB
固定在膝电机定子，AC 是被动连杆，BC 与 BW 是同一根刚性输出杆。
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Geometry:
    length: float = 0.130       # OB = AC = BW = 0.130 m（温和缩短方案）
    crank: float = 0.035        # OA = BC = 0.035 m
    wheel_radius: float = 0.050
    hip_height: float = 0.360

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("全部几何参数必须是有限正数。")
        if self.crank >= self.length:
            raise ValueError("本示例要求主动臂 OA 短于固定臂 OB。")
        if self.hip_height <= 2 * self.length + self.wheel_radius:
            raise ValueError("本台架要求髋高度 > 2*杆长 + 轮半径，保证不接地。")

    @property
    def nominal_length(self) -> float:
        return math.sqrt(2.0) * self.length

    def inverse(self, leg_length: float, beta: float) -> dict[str, float]:
        """返回 MJCF 关节角，闭合 O-A-C-B-O。

        ``hip`` 是整体腿架相对固定髋部的角度；``knee_drive`` 是膝
        电机转子相对膝电机定子/腿架的主动角。``bearing_A`` 和
        ``bearing_B`` 分别是 A、B 两个被动轴承的树内关节角；C 处
        通过 MuJoCo connect 约束形成第三个被动轴承。
        """
        if not (math.isfinite(leg_length) and math.isfinite(beta)):
            raise ValueError("目标必须有限。")
        if not (0.0 < leg_length < 2.0 * self.length):
            raise ValueError("目标腿长不可达或处于伸直奇异位形。")

        alpha = math.acos(leg_length / (2.0 * self.length))
        # OB 的绝对角为 beta-alpha，BW 的绝对角为 beta+alpha；
        # OA 与 BW 反向共线，故 OA 绝对角为 beta+alpha-pi。
        hip = beta - alpha
        output_abs = beta + alpha
        active_abs = output_abs - math.pi
        return {
            "hip": hip,
            "knee_drive": active_abs - hip,
            "bearing_A": hip - active_abs,
            "bearing_B": output_abs - hip,
        }

    def points(self, q: dict[str, float]) -> dict[str, tuple[float, float, float]]:
        """按关节角计算 O、A、B、C 两侧和轮心 W 的位置。"""

        def add(a, b):
            return tuple(x + y for x, y in zip(a, b))

        def down(distance, angle):
            return (distance * math.sin(angle), 0.0,
                    -distance * math.cos(angle))

        O = (0.0, 0.0, self.hip_height)
        theta_ob = q["hip"]
        theta_oa = theta_ob + q.get("knee_drive", 0.0)
        theta_bw = theta_ob + q.get("bearing_B", 0.0)
        theta_ac = theta_oa + q.get("bearing_A", 0.0)

        A = add(O, down(self.crank, theta_oa))
        B = add(O, down(self.length, theta_ob))
        C_AC = add(A, down(self.length, theta_ac))
        C_BC = add(B, down(-self.crank, theta_bw))
        W = add(B, down(self.length, theta_bw))
        return dict(O=O, A=A, B=B, C_AC=C_AC, C_BC=C_BC, W=W)
