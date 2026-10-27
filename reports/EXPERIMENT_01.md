# 实验 1：SuperMimic / LAFAN1-40

日期：2026-10-26（America/New_York）。项目：`/home/xl521/supermimic_force_transition`。

## 已完成

- 克隆上游 `dhmbb2/supermimic_vibe`，固定 commit `072f0910dadc9d95c720e625b546120e9ce917c9`。
- 独立 `.venv`，Python 3.11.17，原版 requirements 全部安装成功；145 个包通过依赖一致性检查。
- GPU 1：RTX A6000 48 GB；Torch 2.14.1 / CUDA 13.0，mjlab 1.2.0，MuJoCo / MuJoCo Warp 3.5.0，Warp 1.12.0。
- 下载并校验全部 40 个 G1 CSV。HF revision `ce1572906efe6157840e8474d5a0d7aa87481e74`，逐文件 git blob 和 SHA256 校验留档。
- 原版脚本转换：441,120 帧、50 Hz、2.450667 小时。输入路径明确为 `data/raw/lafan1_g1/g1`。
- 数据形状、有限值、片段边界、四元数单位长度通过；实际模型 29 个关节顺序与数据说明逐项一致。
- 仿真状态读写检查通过；FK 与 MuJoCo 最大绝对误差：位置 3.58e-7、旋转矩阵 5.66e-7、线速度 7.45e-7、角速度 2.38e-6。
- 全 40 段初始状态同时进入实际仿真，20 步有限值检查通过；原版 `test_env.py 64` 也正常退出。
- 完整 H=32 / k=8、小批量训练 10 次迭代、4 次联合更新成功；WM、策略、critic 三个优化器均更新，checkpoint 参数及日志无 NaN/Inf，策略梯度非零。
- 真实仿真全 40 段评估正常结束。短测试 normalized return 约 0.0018，40/40 终止；这是未训练充分的流程测试，**不是跟踪效果复现成功**。
- 实际 checkpoint 恢复到 iteration 15 / updates 6；补充 replay 后三个网络继续更新，40 段评估再次正常结束。

## 最小上游修复

`code/supermimic/train.py:Trainer.run`：checkpoint 不包含 replay，但上游恢复时跳过 warmup，会在首次更新采样空 buffer。改为根据 replay 大小补齐 warmup，同时移除运行中的旧 DONE 标志。默认从头训练仍预热 64 步，不改奖励、网络、优化器、H、k 或物理参数。恢复不保证逐位复现，因为上游没有保存 replay、仿真状态和 RNG。

## 正式基线

通过 smoke 后已启动 `runs/exp01_lafan40_seed0`。GPU 1，seed 0，4096 环境，WM batch 2048 × 10，BPTT batch 4096 × 32，replay 1000 步，UTD 0.4，预算 3 小时 trainer time，每 4000 次迭代保存并评估全部 40 段。初始化和评估另耗墙钟时间；更新中的冷编译计入 trainer time。

交付时后台 PID 为 `3676870`，仍处于正式 batch 的初次计算图编译阶段，尚无正式评估指标；`reports/exp01_launch.json` 是启动快照而非实时状态。正式效果以该目录 `eval.jsonl` 为准。该实验未添加外力、teacher 或 student 模块。不要将上游 README 的指标作为本次实验结果。

## 文件与复现

- `notes/PROJECT_SPEC.md`：用户原始项目说明。
- `notes/ARCHITECTURE.md`：实际代码架构与未来扩展位置。
- `experiments/exp01/`：下载、环境、数据/仿真校验、训练、恢复检查及完整环境锁。
- `reports/exp01_*.log/json`：本机真实验证记录。
- `runs/exp01_smoke/`、`runs/exp01_resume_smoke/`：小规模训练模型和指标。
- `runs/exp01_lafan40_seed0/`：正式训练输出。

```bash
cd /home/xl521/supermimic_force_transition
tail -f runs/exp01_lafan40_seed0/console.log
# 正式实验退出后继续至累计 12 小时；脚本自动恢复 last.pt：
bash experiments/exp01/train.sh --wall-hours 12
```

完整步骤见 `experiments/exp01/README.md`。数据、模型、缓存、虚拟环境不会加入 git。

## 2026-10-27 实际完成结果

训练已结束。iteration 36000: normalized return **0.90973866**，**0/40** 失败；iteration 32000 同样 0/40。最终 last.pt (36378) 为 0.89115459、1/40 失败。实验 2 采用 `ckpt_0036000.pt` 初始化，不把最后一个 checkpoint 当作最优。
