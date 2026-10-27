"""SuperMimic training with CEER-style public command and numerical health checks."""
import json,math,os,faulthandler,signal
from pathlib import Path
import torch
from supermimic.train import Trainer,parse_args

class CEERTrainer(Trainer):
    def __init__(self,cfg):
        if cfg.command_interface!='ceer':raise ValueError('This launcher requires CEER interface')
        super().__init__(cfg)
        print(json.dumps({'rank':self.rank,'world':self.world,'device':str(self.device),'clips':self.lib.num_clips,'frames':self.lib.num_frames,'motion_hours':self.lib.hours,'goal_dim':self.goal_dim,'allocated_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
    def _extra_state(self):
        return {'interface':{'name':'ceer_style_root_hands_v1','command_dim':18,'history':10,'frame_dim':93,'action_dim':29,'external_force_training':False,'reference_joint_input':False}}
    def update_wm(self):
        result=super().update_wm();self.check(result);return result
    def update_policy(self):
        result=super().update_policy();self.check(result)
        if self.updates==1:
            print(json.dumps({'rank':self.rank,'first_updates_finite':True,'policy_gradient_norm':result.get('pi_gnorm'),'allocated_GB':torch.cuda.memory_allocated()/1e9,'peak_GB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
        return result
    def check(self,values):
        for key,value in values.items():
            if not math.isfinite(float(value)):raise FloatingPointError(f'rank {self.rank}: nonfinite {key}={value}')

if __name__=='__main__':
    faulthandler.register(signal.SIGUSR1)
    cfg=parse_args();trainer=CEERTrainer(cfg)
    if trainer.rank==0:
        Path(cfg.out,'interface.json').write_text(json.dumps(trainer._extra_state(),indent=2))
    trainer.run()
    if trainer.world>1:
        import torch.distributed as dist
        dist.destroy_process_group()
