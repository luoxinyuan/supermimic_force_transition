"""Export the trained low-level actor with only proprioception and CEER commands."""
import argparse,json
from pathlib import Path
import torch
from supermimic.evaluate import load_supermimic_policy

ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);a=ap.parse_args()
p,ck=load_supermimic_policy(a.checkpoint,'cpu')
assert ck['cfg']['command_interface']=='ceer'
p.eval();example=(torch.zeros(1,930),torch.zeros(1,18))
with torch.no_grad():
 exported=torch.jit.trace(p,example)
 for batch in [1,4]:
  x=torch.randn(batch,930)*.1;g=torch.randn(batch,18)*.1
  torch.testing.assert_close(p(x,g),exported(x,g))
out=Path(a.checkpoint).parent
exported.save(str(out/'ceer_low_level.pt'))
meta={'checkpoint':a.checkpoint,'iteration':ck['it'],'inputs':{'proprioception':[None,930],'command':[None,18]},'command_order':['root_height_m','root_vx_body_m_s','root_vy_body_m_s','heading_x_current_yaw_frame','heading_y_current_yaw_frame','reserved_force_condition_constant_5','left_hand_x_root_m','left_hand_y_root_m','left_hand_z_root_m','right_hand_x_root_m','right_hand_y_root_m','right_hand_z_root_m','left_hand_rotvec_x_rad','left_hand_rotvec_y_rad','left_hand_rotvec_z_rad','right_hand_rotvec_x_rad','right_hand_rotvec_y_rad','right_hand_rotvec_z_rad'],'proprioception_order':'10 oldest-to-newest frames; each q-q0+encoder_bias(29), qd(29), body_gyro(3), projected_gravity(3), previous_action(29)','output':'29 normalized joint position-target actions in MuJoCo depth-first joint order','frequency_hz':50,'motor_mapping':'q_target = q0 + action_scale * action - encoder_bias; use training G1 actuator parameters','checkpoint_compatible_with_original_ceer':False,'external_forces_in_training':False,'export_numerical_equivalence':True}
(out/'ceer_low_level.json').write_text(json.dumps(meta,indent=2)+'\n');print(out/'ceer_low_level.pt')
