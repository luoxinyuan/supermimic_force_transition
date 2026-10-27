"""Same flat-ground criteria as filter_bones.py, with batched FK instead of per-clip GPU launches."""
import argparse,importlib.util,json
import mujoco,numpy as np,torch
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from supermimic.kinematics import FastTreeKinematics
from supermimic.rotations import quat_to_mat
spec=importlib.util.spec_from_file_location('filter_original','code/scripts/filter_bones.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--report',required=True);a=ap.parse_args()
d=np.load(a.input);qpos=d['qpos'];lengths=d['lengths'];starts=d['starts'];names=d['names'];fps=float(d['fps']);keep1=original.name_filter(names)
m=Entity(get_g1_robot_cfg()).spec.compile();kin=FastTreeKinematics(m,'cuda',root_body=1);caps=[]
for g in range(m.ngeom):
 name=m.geom(g).name
 if name.endswith('_collision') and 'foot' in name and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_CAPSULE:
  side=0 if name.startswith('left') else 1;b=kin.body_names.index(m.body(m.geom_bodyid[g]).name)
  R=np.zeros(9);mujoco.mju_quat2Mat(R,m.geom_quat[g]);axis=R.reshape(3,3)[:,2]*m.geom_size[g,1]
  caps.append((side,b,torch.tensor(np.stack((m.geom_pos[g]+axis,m.geom_pos[g]-axis)),dtype=torch.float32,device='cuda'),m.geom_size[g,0]))
assert len(caps)==14
# 4 CPU scalars/frame: lowest body z, lowest left/right foot capsule z, pelvis upright.
metrics=np.empty((len(qpos),4),dtype=np.float32)
def geometry(qp):
 R=quat_to_mat(qp[:,3:7]);P,Rb=kin.forward(qp[:,:3],R,qp[:,7:]);feet=torch.full((len(qp),2),1e9,device='cuda')
 for side,b,e,radius in caps:
  world=P[:,b,None,:]+(Rb[:,b,None]@e[None,:,:,None]).squeeze(-1)
  feet[:,side]=torch.minimum(feet[:,side],world[...,2].min(-1).values-radius)
 return torch.cat((P[...,2].min(-1).values[:,None],feet,R[:,2,2,None]),-1)
with torch.no_grad():
 for start in range(0,len(qpos),32768):
  qp=torch.as_tensor(qpos[start:start+32768],device='cuda');metrics[start:start+len(qp)]=geometry(qp).cpu().numpy()
  if start%(32768*100)==0:print('FK',start,len(qpos),flush=True)
 # Check per-clip vs batch numerical agreement on varied clips.
 maxerr=0.
 for c in np.random.default_rng(31).choice(len(names),32,replace=False):
  st=int(starts[c]);length=int(lengths[c]);one=geometry(torch.as_tensor(qpos[st:st+length],device='cuda')).cpu().numpy()
  maxerr=max(maxerr,float(np.max(np.abs(one-metrics[st:st+length]))))
 assert maxerr<2e-5,maxerr
keep=[];reasons={'elevated':0,'penetration':0,'floating_feet':0};removed=[]
for k,c in enumerate(keep1):
 st=int(starts[c]);length=int(lengths[c]);metric=metrics[st:st+length];qp=qpos[st:st+length]
 zmin=metric[:,0];feet=metric[:,1:3];upright=metric[:,3]>.9;standing=qp[:,2]>.65
 vel=np.linalg.norm(np.gradient(qp[:,:3],1/fps,axis=0),axis=-1);qd=np.abs(np.gradient(qp[:,7:],1/fps,axis=0)).mean(-1)
 stationary=(vel<.2)&(qd<.3);bad=None
 if original.runs_at_least(zmin>.2,int(round(.6*fps))):bad='elevated'
 elif zmin.min()<-.12 or (zmin<-.08).mean()>.8:bad='penetration'
 elif original.runs_at_least((feet>.02).all(-1)&upright&standing&stationary,int(round(.5*fps))):bad='floating_feet'
 if bad:reasons[bad]+=1;removed.append({'name':str(names[c]),'reason':bad})
 else:keep.append(c)
 if k%20000==0:print('filter',k,len(keep1),reasons,flush=True)
sel=np.concatenate([np.arange(starts[c],starts[c]+lengths[c]) for c in keep]);newlen=lengths[keep]
np.savez(a.output,qpos=qpos[sel],qvel=d['qvel'][sel],starts=np.concatenate([[0],np.cumsum(newlen)[:-1]]),lengths=newlen,names=names[keep],fps=d['fps'])
result={'raw':len(names),'after_name_filter':len(keep1),'after_feasibility_filter':len(keep),'feasibility_reasons':reasons,'hours':float(newlen.sum()/fps/3600),'frames':int(newlen.sum()),'batched_vs_per_clip_fk_max_error':maxerr,'paper':{'raw':142220,'after_name_filter':129667,'after_feasibility_filter':129514,'hours':264.56},'name_terms':original.NAME_TERMS,'removed_by_feasibility':removed}
open(a.report,'w').write(json.dumps(result,indent=2)+'\n');open(a.report.replace('.json','_clips.txt'),'w').write('\n'.join(str(names[c]) for c in keep)+'\n');print({k:v for k,v in result.items() if k not in ['removed_by_feasibility','name_terms']},flush=True)
