"""Plot landing, balance error, ground support and all six motor torques."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    with (args.directory/'log.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    def values(name):
        return np.array([float(row[name]) for row in rows])
    t=values('time_s')
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
    for name in ('pitch','roll'):
        axes[0].plot(t,values(name+'_deg'),label=name)
    axes[0].set_ylabel('Body attitude (deg)')
    axes[0].legend()
    axes[1].plot(t,values('height_m')*1000,label='chassis height')
    if 'target_height_m' in rows[0]:
        axes[1].plot(t,values('target_height_m')*1000,'--',label='target height')
        axes[1].legend()
    axes[1].set_ylabel('Height (mm)')
    axes[2].plot(t,values('floor_normal_n'))
    axes[2].set_ylabel('Ground normal (N)')
    for side,style in (('left','-'),('right','--')):
        for motor in ('hip','knee','wheel'):
            axes[3].plot(t,values(f'{motor}_motor_{side}_nm'),style,
                         label=f'{side} {motor}',linewidth=1)
    axes[3].set_ylabel('Motor torque (N m)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(ncol=3)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle('Free-floating wheel-leg robot: landing and active balance')
    path=args.directory/'balance.png'
    fig.savefig(path,dpi=160)
    plt.close(fig)
    print(path.resolve())
    if 'target_leg_length_m' in rows[0]:
        fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
        target = values('target_leg_length_m')*1000
        axes[0].plot(t,target,'k--',label='target')
        for side,style in (('left','-'),('right',':')):
            actual = values(f'{side}_leg_length_m')*1000
            axes[0].plot(t,actual,style,label=side)
            axes[1].plot(t,actual-target,style,label=side)
        axes[0].set_ylabel('Leg length (mm)')
        axes[1].set_ylabel('Tracking error (mm)')
        axes[1].set_xlabel('Time (s)')
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend()
        fig.suptitle('Synchronized leg extension while balancing')
        path=args.directory/'leg_extension.png'
        fig.savefig(path,dpi=160)
        plt.close(fig)
        print(path.resolve())


if __name__=='__main__':
    main()
