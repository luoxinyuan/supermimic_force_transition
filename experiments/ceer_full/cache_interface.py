"""Precompute command wrist geometry only, using MuJoCo FK and CEER hand offsets."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from supermimic.kinematics import FastTreeKinematics
from supermimic.ceer.interface import HAND_OFFSETS

ap=argparse.ArgumentParser();ap.add_argument('--motions',required=True);a=ap.parse_args()
torch.set_num_threads(4)
m=Entity(get_g1_robot_cfg()).spec.compile();kin=FastTreeKinematics(m,'cuda',root_body=1)
ids=kin.body_index(['left_wrist_yaw_link','right_wrist_yaw_link'])
off=torch.tensor(HAND_OFFSETS,device='cuda')
d=np.load(a.motions);qp=d['qpos'];n=len(qp);dest=Path(a.motions+'.ceer_wrist.npy')
out=np.lib.format.open_memmap(str(dest)+'.tmp',mode='w+',dtype=np.float32,shape=(n,12))
with torch.no_grad():
 for start in range(0,n,32768):
  q=torch.as_tensor(qp[start:start+32768,7:],device='cuda');b=len(q)
  p,r=kin.forward(torch.zeros(b,3,device='cuda'),torch.eye(3,device='cuda').expand(b,3,3),q)
  pos=p[:,ids]+(r[:,ids]@off[...,None]).squeeze(-1)
  rot=r[:,ids].cpu().numpy()
  rv=Rotation.from_matrix(rot.reshape(-1,3,3)).as_rotvec().reshape(b,6)
  out[start:start+b]=np.concatenate((pos.cpu().numpy().reshape(b,6),rv),-1)
  if start%(32768*100)==0:print(start,n,flush=True)
out.flush();del out;os.replace(str(dest)+'.tmp',dest)
print('saved',dest,n,flush=True)
