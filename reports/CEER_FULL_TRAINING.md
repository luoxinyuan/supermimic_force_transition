# CEER-style 接口的全量 SuperMimic 训练

项目：`/home/xl521/supermimic_force_transition`。

## 接入范围

新增 `--command-interface ceer`，默认仍为 `full`，原 SuperMimic 跟踪和既有 force-transition 实验保持原接口。

CEER-style 模式接收 18 维当前帧指令：root 高度、机体坐标下的 XY 目标速度、当前 yaw 坐标下的目标朝向 XY、固定为 5 的保留 conditioning 槽位，以及左右 hand_mimic 的 root-relative 位置和轴角姿态。手部固定偏移来自 feet_reach 的原始 G1 USD，约为 wrist_yaw_link 沿 X 正向 0.116 m。

策略同时读取 10×93 维本体感知历史，输出 29 维关节位置目标动作。完整参考关节、参考根节点 XY、motion ID 和 phase 不直接输入 actor。原全身密集跟踪奖励仍保留；未添加外力。保留槽位不是实际测力值，也不代表已训练力控制能力。

这里迁移的是 **CEER 的 EE/root 公开指令语义**。新策略保留 SuperMimic 的 32 维 motion latent、encoder/decoder、世界模型和 BPTT；不是加载旧 CEER 权重，也不是复制它的特权估计器/关节目标估计器。新模型不能直接替换旧 CEER 的 257 维输入或电机包装器。

EE/root 指令对全身姿态存在多解，尤其没有显式指定腿部动作和 root roll/pitch。因此这是全量数据上的稀疏指令控制器实验，不能假定具有论文全关节参考接口的零失败表现。

## 数据

- 下载版本：`bones-studio/seed@2f59b2077b9da34dd4e43618e705c7cb962c9a66`，源文件 SHA-256 已验证。
- 全部 142,220 CSV 转换成功，没有跳过片段；120 Hz → 50 Hz，51,903,579 帧，288.353 小时。
- 沿用仓库名称筛选后 129,115 段；再剔除 126 段高处支撑、46 段穿地、39 段静止悬空。
- **最终 128,904 段，47,032,428 帧，261.291 小时。** 训练遍历这一完整保留集合，不是抽取小子集。
- 与论文 129,514 段 / 264.56 小时不同。报告使用实际数量，不宣称完全相同的 benchmark split。
- `reports/ceer_full/filter.json` 和 `filter_clips.txt` 记录规则、剔除原因和完整保留清单。
- 批量筛选仅改变 FK 执行批量，阈值不变；32 段抽查与逐段 FK 最大差为 0。

## 已完成检查

- 128 个参考帧与原生 MuJoCo FK 对照：手位置最大误差 8.19e-8 m，姿态矩阵最大误差 1.70e-7。
- 世界 XY 平移和共同 yaw 旋转不改变指令。
- 三卡真实训练小测试：4 次联合更新，策略参数更新且有限；策略、世界模型、critic 的参数跨卡逐位一致，学习率一致。
- 正式 batch 编译测试：每卡 4096 环境，WM batch 2048×10、policy batch 4096、H=32，20 iterations / 8 updates 通过；编译后短测约 0.27–0.46 s/iteration，不代表完整数据上的长期速度。
- 真实仿真和想象 rollout 的指令一致，参考关节/XY 输入隔离检查通过。
- 导出的 actor 只有本体感知和 18 维 command 输入，batch 1/4 与原模型数值一致。
- 独立评估入口已实际运行。未训练充分的 smoke 模型 4/4 失败，仅验证执行流程，非行为验收。
- 原 force-transition 模型回归通过，零力动作最大差为 0。

## 正式配置与结果边界

物理 GPU **2、3、4**，每卡 4096 环境，共 12,288 环境训练一个同步策略。从头训练，最多 320,000 iterations 或累计 48 个训练小时。每 4000 iterations 保存模型；每 8000 iterations 在固定 256 段上监控，不能将这组数值当作全量结果。

训练退出后，启动脚本会对 `last.pt` 执行完整保留集合评估，写入 `evaluation_full_final.json`，再导出 `ceer_low_level.pt` 和接口元数据。实际完成迭代数也写入最终评估；时间预算用尽不等于已收敛。

运行目录：`runs/ceer_full_seed0/`；启动与数据清单、源码快照和哈希均保存在其中。运行状态以 `console.log`、`log.jsonl`、`launch.json` 为准。

复现与部署接口说明：`experiments/ceer_full/README.md`。

## 本次启动确认

2026-10-27 15:24 EDT 启动，后台 supervisor PID 为 4069914。三个 rank 均确认加载 128,904 段完整保留数据并完成初始有限值更新；同步策略梯度范数为 0.27797，PyTorch 峰值显存约 29.01 GB/卡。此记录仅证明正式训练已运行，不表示策略已经收敛。

已确认正式 iteration 100 / 40 次联合更新，累计 2,015,232 个三卡环境交互步（含预热），首个日志窗口 0.441 s/iteration。完整训练预计需数十小时，实际以日志和终止预算为准。
