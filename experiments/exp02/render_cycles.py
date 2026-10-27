"""Render saved true-simulator states. No learned-model rollout or generated imagery."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
import mediapy as media
import imageio_ffmpeg
media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
from PIL import Image,ImageDraw,ImageFont
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import TrackingSimEnv,EnvCfg
from supermimic.task import TaskCfg

p=argparse.ArgumentParser();p.add_argument('--input',default='reports/exp02/cycles_2000.npz');p.add_argument('--output',default='reports/exp02/teacher_cycles.mp4');a=p.parse_args()
traj=np.load(a.input);states=traj['state'];forces=traj['force_z']
env=TrackingSimEnv(MotionLibrary('data/processed/standing_squatting.npz','cuda'),EnvCfg(num_envs=1,compile=False,domain_rand=False),TaskCfg(),'cuda')
m=env.mj_model;d=mujoco.MjData(m)
m.vis.global_.offwidth=480;m.vis.global_.offheight=360
renderer=mujoco.Renderer(m,height=360,width=480)
camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.65];camera.distance=2.1;camera.azimuth=120;camera.elevation=-12
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(fontpath,16) if Path(fontpath).exists() else ImageFont.load_default()
indices=[40,8,56,24];titles=['Standing start | no force','Standing start | force-guided','Squatting start | no force','Squatting start | force-guided']
with media.VideoWriter(a.output,shape=(800,960),fps=25) as writer:
    for t in range(0,len(states),2):
        canvas=Image.new('RGB',(960,800),(20,25,32));draw=ImageDraw.Draw(canvas)
        for j,(idx,title) in enumerate(zip(indices,titles)):
            d.qpos[:]=states[t,idx,:36];d.qvel[:]=0;mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=camera);frame=Image.fromarray(renderer.render())
            x=(j%2)*480;y=(j//2)*400;canvas.paste(frame,(x,y+40))
            draw.text((x+10,y+3),title,font=font,fill='white')
            draw.text((x+10,y+21),f't={t*.02:4.1f}s   Fz={forces[t,idx]:+5.1f} N   pelvis={states[t,idx,2]:.3f} m',font=font,fill=(115,210,220))
        writer.add_image(np.asarray(canvas))
        if t==150:canvas.save(str(Path(a.output).with_suffix('.png')))
renderer.close();print(a.output)
