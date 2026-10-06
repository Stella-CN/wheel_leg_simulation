"""读取仿真日志，各指标单独成图；不需要启动 MuJoCo。"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, nargs="?", default=Path("results/log.csv"))
    args = p.parse_args()
    if not args.csv.is_file():
        p.error(f"文件不存在：{args.csv}")
    data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
    out = args.csv.parent
    # 兼容拓扑改造前的历史日志；新模型统一记录 knee_torque_nm。
    knee_torque_key = ("knee_torque_nm" if "knee_torque_nm" in data.dtype.names
                       else "crank_torque_nm")
    specs = [
        ("length", "Leg length (mm)", [("length_target_m", "Target", 1000),
                                          ("length_actual_m", "Actual", 1000)]),
        ("angle", "Virtual leg angle (deg)", [("beta_target_deg", "Target", 1),
                                                ("beta_actual_deg", "Actual", 1)]),
        ("torque", "Commanded output torque (N m)", [("hip_torque_nm", "Hip", 1),
                                                       (knee_torque_key, "Knee", 1),
                                                       ("wheel_torque_nm", "Wheel", 1)]),
        ("closure", "C closure error (mm)", [("closure_m", "Closure", 1000)]),
    ]
    for name, ylabel, series in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label, scale in series:
            ax.plot(data["time_s"], data[key]*scale, label=label)
        ax.set(xlabel="Simulation time (s)", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/(name+".png"), dpi=170)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(data["wheel_x_m"]*1000, data["wheel_z_m"]*1000, label="Actual")
    beta = np.deg2rad(data["beta_target_deg"])
    length = data["length_target_m"]*1000
    ax.plot(length*np.sin(beta), -length*np.cos(beta), "--", label="Target")
    ax.set(xlabel="X relative to hip (mm)", ylabel="Z relative to hip (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"wheel_trajectory.png", dpi=170)
    plt.close(fig)
    print(f"Saved 5 plots to: {out.resolve()}")

if __name__ == "__main__":
    main()
