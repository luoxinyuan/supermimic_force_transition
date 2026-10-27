"""Teacher-labelled causal recurrent distillation and closed-loop deployment test."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg
from supermimic.task import TaskCfg
from supermimic.force_transition.env import TransitionEnv
from supermimic.force_transition.policy import ForceTeacher
from supermimic.force_transition.student import ForceStudent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rounds',type=int,default=4);ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--action-weight',type=float,default=1.);ap.add_argument('--resume');ap.add_argument('--out',default='runs/exp03_student');a=ap.parse_args()
    torch.manual_seed(303);np.random.seed(303);random.seed(303)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load('runs/exp02_wm_refine/last.pt',map_location='cuda',weights_only=False)
    teacher=ForceTeacher().cuda();teacher.load_state_dict(ck['policy']);teacher.eval()
    student=ForceStudent(teacher).cuda()
    if a.resume: student.load_state_dict(torch.load(a.resume,map_location='cuda',weights_only=False)['student'])
    opt=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=a.lr)
    lib=MotionLibrary(ck['cfg']['motions'],'cuda')
    env=TransitionEnv(lib,EnvCfg(num_envs=64,compile=False,domain_rand=False,obs_noise=False,reset_noise=False),TaskCfg(),'cuda',delay=ck['cfg']['force_delay'])
    env.reset();ids=torch.arange(64,device='cuda');mode=ids%2
    def reset(seed,paired=False):
        torch.manual_seed(seed);env.reset_envs(ids,frames=lib.starts[mode]);env._finish_resets(ids,force=True)
        env.schedule[:,0]=50 if paired else torch.randint(30,110,(64,),device='cuda')
        env.schedule[:,1]=175 if paired else torch.randint(125,200,(64,),device='cuda')
        mag=torch.full((64,),25.,device='cuda') if paired else torch.empty(64,device='cuda').uniform_(15,30)
        env.schedule[:,2]=mag*torch.where(mode==0,-1.,1.);env.schedule[ids%4<2,2]=0
        env.schedule[:,3]=10000
    datasets=[];results=[]
    for rnd in range(a.rounds):
        reset(303+rnd);hidden=None;obslist=[];zlist=[];alive=torch.ones(64,dtype=torch.bool,device='cuda');masks=[]
        with torch.no_grad():
            for t in range(500):
                obs,g=env.policy_inputs();z=teacher.latent(obs,g);ta=student.decode(obs,z)
                sa,hidden=student(obs,hidden)
                # DAgger labels on learner trajectories, with per-episode teacher mixing.
                use_student=(ids%8>=max(0,8-rnd*2))[:,None]
                action=torch.where(use_student,sa,ta) if not a.resume else ta+(sa-ta)*min(1.,.2+.1*rnd)
                obslist.append(obs.clone());zlist.append(z);masks.append(alive.clone())
                _,info=env.step(action,auto_reset=False);alive &= ~info['terminated']
        datasets.append((torch.stack(obslist,1),torch.stack(zlist,1),torch.stack(masks,1)))
        print(json.dumps({'round':rnd,'collected_valid_fraction':torch.stack(masks).float().mean().item()}),flush=True)
        student.train()
        for epoch in range(a.epochs):
            losses=[]
            for observations,labels,valid in datasets:
                for batch in torch.randperm(64,device='cuda').split(16):
                    h=None
                    for start in range(0,500,50):
                        x=observations[batch,start:start+50];target=labels[batch,start:start+50];mask=valid[batch,start:start+50]
                        z,h=student.sequence(x,h)
                        pred=student.decode(x,z)
                        with torch.no_grad(): truth=student.decode(x,target)
                        loss=((z-target).square().mean(-1)+a.action_weight*(pred-truth).square().mean(-1))
                        loss=(loss*mask).sum()/mask.sum().clamp_min(1)
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.);opt.step();h=h.detach();losses.append(loss.item())
            print(json.dumps({'round':rnd,'epoch':epoch,'loss':float(np.mean(losses))}),flush=True)
        student.eval();metrics={}
        for name,policy in [('teacher',teacher),('student',student)]:
            reset(9137,True);h=None;heights=[];knees=[];errors=[];fallen=torch.zeros(64,dtype=torch.bool,device='cuda')
            with torch.no_grad():
                for t in range(500):
                    # Student branch never calls policy_inputs or reads force/target/geometry.
                    obs=env.obs_hist.flatten(1)
                    if name=='student': action,h=policy(obs,h)
                    else: _,g=env.policy_inputs();action=policy(obs,g)
                    s,info=env.step(action,auto_reset=False);fallen|=info['terminated']
                    heights.append(s['pos'][:,2].cpu().numpy());knees.append(s['q'][:,[3,9]].mean(-1).cpu().numpy())
            heights=np.stack(heights);knees=np.stack(knees)
            np.savez(f'reports/exp03/{name}_round{rnd}.npz',height=heights,knee=knees)
            item={}
            for m,label in [(0,'stand_to_squat'),(1,'squat_to_stand')]:
                base=(np.arange(64)%2==m)&(np.arange(64)%4<2);force=(np.arange(64)%2==m)&(np.arange(64)%4>=2)
                dz=float(heights[-100:,force].mean()-heights[-100:,base].mean());falls=int(fallen[torch.as_tensor(base|force,device='cuda')].sum())
                item[label]={'no_force_height':float(heights[-100:,base].mean()),'force_height':float(heights[-100:,force].mean()),'height_change':dz,'falls':falls,'passed':bool(dz*(-1 if m==0 else 1)>.045 and falls==0 and abs(float(heights[-100:,base].mean())-(.782827 if m==0 else .706031))<.025)}
            metrics[name]=item
        metrics['round']=rnd;metrics['passed']=all(v['passed'] for v in metrics['student'].values())
        results.append(metrics);Path('reports/exp03/results.json').write_text(json.dumps(results,indent=2))
        torch.save({'student':student.state_dict(),'round':rnd,'metrics':metrics,'input':'930 proprioception values and recurrent hidden state; no force or root pose'},out/f'round{rnd}.pt')
        print(json.dumps(metrics),flush=True)
    torch.save({'student':student.state_dict(),'metrics':metrics},out/'last.pt')
if __name__=='__main__':main()
