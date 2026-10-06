"""Plot yaw tracking, planar drift, leg extension and motor torques."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "log.csv").open() as stream:
        rows = list(csv.DictReader(stream))

    def values(name):
        return np.array([float(row[name]) for row in rows])

    time = values("time_s")
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True,
                             layout="constrained")
    axes[0].plot(time, values("yaw_deg"), label="measured yaw")
    axes[0].plot(time, values("target_yaw_deg"), "--", label="target yaw")
    axes[0].set_ylabel("Yaw (deg)")
    rate_axis = axes[0].twinx()
    rate_axis.plot(time, values("yaw_rate_deg_s"), color="tab:red",
                   alpha=.75, label="yaw rate")
    rate_axis.set_ylabel("Yaw rate (deg/s)", color="tab:red")
    rate_axis.tick_params(axis="y", labelcolor="tab:red")
    lines, labels = axes[0].get_legend_handles_labels()
    rate_lines, rate_labels = rate_axis.get_legend_handles_labels()
    axes[0].legend(lines + rate_lines, labels + rate_labels, loc="upper left")

    axes[1].plot(time, values("x_m") * 1000, label="x")
    axes[1].plot(time, values("y_m") * 1000, label="y")
    axes[1].plot(time, values("planar_displacement_m") * 1000,
                 "--", label="planar distance")
    axes[1].set_ylabel("Position (mm)")
    axes[1].legend()

    axes[2].plot(time, values("target_leg_length_m") * 1000,
                 "k--", label="target")
    for side, style in (("left", "-"), ("right", ":")):
        axes[2].plot(time, values(f"{side}_leg_length_m") * 1000,
                     style, label=side)
    axes[2].set_ylabel("Leg length (mm)")
    axes[2].legend()

    for side, style in (("left", "-"), ("right", "--")):
        for motor in ("hip", "knee", "wheel"):
            axes[3].plot(time, values(f"{motor}_motor_{side}_nm"), style,
                         label=f"{side} {motor}", linewidth=1)
    axes[3].set_ylabel("Torque (N m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=3)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle("In-place yaw maneuver with synchronized leg extension")
    path = args.directory / "spin_extension.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path.resolve())


if __name__ == "__main__":
    main()
