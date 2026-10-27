import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
h=np.load('reports/exp03/heldout.npz');fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
t=np.arange(500)*.02;ids=np.arange(64)
for row,name in enumerate(['teacher','student']):
 for mode in [0,1]:
  ax=axes[row,mode]
  for forced,label,color in [(False,'No force','gray'),(True,'15–30 N','tab:blue')]:
   mask=(ids%2==mode)&((ids%4>=2)==forced);z=h[name+'_height'][:,mask]
   ax.plot(t,z.mean(1),label=label,color=color);ax.fill_between(t,z.min(1),z.max(1),alpha=.18,color=color)
  ax.set_title(f'{name}: '+('standing start' if mode==0 else 'squatting start'));ax.set_ylabel('Root height (m)');ax.set_xlabel('Time (s)');ax.legend();ax.grid(alpha=.2)
fig.suptitle('Held-out physical rollouts (shading: min/max across schedules)');fig.tight_layout();fig.savefig('reports/exp03/heldout.png',dpi=150)
