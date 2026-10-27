"""Export paired real-simulator height/knee curves (not imagined trajectories)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('trajectory',type=Path);a=p.parse_args()
d=np.load(a.trajectory);t=np.arange(len(d['height']))*.02
fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True)
labels=['No force','Force + Teacher','Force, blinded Teacher','Latent only (no physics force)']
for m,title in enumerate(['Standing start: downward force','Squatting start: upward force']):
    axes[0,m].set_title(title)
    for g,label in enumerate(labels):
        idx=m*16+g*4
        axes[0,m].plot(t,d['height'][:,idx:idx+4].mean(1),label=label)
        axes[1,m].plot(t,d['knee'][:,idx:idx+4].mean(1),label=label)
    axes[2,m].plot(t,d['force_z'][:,m*16+4],color='black')
    for ax in axes[:,m]:ax.axvspan(1,4.5,color='gray',alpha=.1);ax.grid(alpha=.2)
    axes[2,m].set_xlabel('Time (s)')
axes[0,0].set_ylabel('Pelvis height (m)');axes[1,0].set_ylabel('Mean knee angle (rad)');axes[2,0].set_ylabel('Scheduled force Fz (N)')
axes[0,0].legend(fontsize=8)
fig.suptitle('One checkpoint, paired true-simulator tests; shaded = force pulse')
fig.tight_layout();fig.savefig(a.trajectory.with_suffix('.png'),dpi=160)
print(a.trajectory.with_suffix('.png'))
