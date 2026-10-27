"""Deployable student export and causal/input-isolation checks."""
import argparse,json
from pathlib import Path
import torch
from supermimic.force_transition.policy import ForceTeacher
from supermimic.force_transition.student import ForceStudent

ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='runs/exp03_student/last.pt');args=ap.parse_args()
torch.manual_seed(7)
t=ForceTeacher();s=ForceStudent(t)
ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
s.load_state_dict(ck['student']);s.eval()
x=torch.randn(2,8,930)*.01
with torch.no_grad():
    full,_=s.sequence(x)
    prefix,_=s.sequence(x[:,:4])
    torch.testing.assert_close(full[:,:4],prefix)
    h=None;zs=[]
    for i in range(8):
        z,h=s.sequence(x[:,i:i+1],h);zs.append(z)
    torch.testing.assert_close(full,torch.cat(zs,1))
    # Export a deployment graph with exactly obs and hidden arguments.
    example=(x[:,0],torch.zeros(1,2,128))
    graph=torch.jit.trace(s,example)
    graph.save(str(Path(args.checkpoint).parent/'deploy.pt'))
    expected=s(*example);actual=graph(*example)
    for aa,bb in zip(expected,actual):torch.testing.assert_close(aa,bb)
    for batch in (1,4):
        inputs=(torch.zeros(batch,930),torch.zeros(1,batch,128))
        for aa,bb in zip(s(*inputs),graph(*inputs)):torch.testing.assert_close(aa,bb)
    assert not any(any(w in k for w in ['reference_encoder','goal_norm','force_memory','schedule']) for k in s.state_dict())
result={'causal_prefix':True,'step_sequence_equivalence':True,'deployment_export_equivalence':True,'export_batch_sizes_tested':[1,2,4],'inputs':['obs_history[batch,930]','hidden[1,batch,128]'],'privileged_parameters_absent':True,'reset':'zero hidden at episode/robot reset; preserve across force removal','latent':'total teacher control latent, not calibrated force vector'}
Path('reports/exp03/input_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
