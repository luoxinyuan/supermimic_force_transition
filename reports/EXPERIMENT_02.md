# 实验 2：Force-aware Teacher

日期：2026-10-27（America/New_York）。项目仍在 `~/supermimic_force_transition`。

## 当前结果

**核心验证已通过，训练已主动停止。** 固定模型 `runs/exp02_teacher_seed0/ckpt_0002000.pt` 在真实仿真中完成双向主动切换、撤力保持及连续往返；这不是仅看 loss 或想象回报的结论。

累计 2000 次迭代中，前 1000 次为仅世界模型预热，后 1000 次优化 Teacher（400 次 BPTT 更新）。trainer time 合计约 0.165 小时，含更新中编译，不代表完整工作耗时。原定 2 小时上限未用完，达到核心验收后停止以保存资源。恢复入口仍保留。

## 与原框架的关系

原始跟踪实现不变。新增 `code/supermimic/force_transition/`，复用原来的 2048×3 世界模型、replay、H=32 BPTT、lambda-return critic、优化器循环与 checkpoint。
初始化使用实验 1 的 `ckpt_0036000.pt`（0.90974、0/40 失败），不使用有一次失败的 last.pt。

Teacher 保留原始 motion encoder/decoder 作为冻结的稳定控制器，学习 32 维力 latent 残差。两种静态端点描述始终同时提供，名义 latent 由当前膝姿态插值；不提供环境选中的目的端点。零力、零力历史时残差恒为零。实测该条件下与原始端点跟踪策略动作差为 0。

这是一种受约束的首阶段 Teacher，不是从头学习通用 decoder。静态端点描述使用模拟器状态构造相对几何，因此也不能直接作为未来仅本体感知 Student 的部署输入。

## 已通过的验证

- 两个端点由关节设计生成，10 秒实际闭环保持，64/64 环境无跌倒。
- 初始几何检查：关节限位、足部胶囊碰撞几何、COM 支撑范围、无自碰撞、无明显穿地。
- 当前为浅蹲：稳态髋高约 0.783 / 0.706 m。
- 世界坐标外力，局部施力点，关于 body COM 的力矩；与 MuJoCo `mj_applyFT` 等效。
- 相反力方向、10/20 N 幅值差异、3 接触叠加与置换、环境隔离。
- 原环境与新环境零力动力学差异，与两个未修改环境之间的 GPU 数值差异处于同一数量级；详见 `zero_force_validation.json`。
- 当前步 force/action/next_state 对齐；WM 8 步窗口逐步读取真实力，不广播首步外力。
- 延迟 >=1 控制步，包含历史滤波器；已测试延迟 2 步不会通过 memory 绕过。
- 修改奖励端点标签或未来 force schedule 不改变当前策略输入。
- WM 对状态、动作、外力可微；Teacher 力编码器能通过冻结 decoder 获得梯度。
- 小规模 H=32、k=8 训练及三个优化器更新成功，无 NaN/Inf。

首轮静态行为克隆虽然离线 MSE 很小，却闭环失稳，已被弃用；日志保留在 `smoke.log`，有效版本为 `smoke_v2.log`。

## 世界模型 held-out 验证（Teacher 优化前的预热模型）

256 个新采样有效窗口，不进入训练 replay；包括两个端点上的正/负力及零力，力幅值 12/22/28 N。
比较的是训练后的同一模型：真实力输入 vs 屏蔽力输入，不是另一个独立训练的 force-blind 模型。

| 预测步数 | Force-aware normalized MSE | Force-blind normalized MSE |
|---|---:|---:|
| 1 | 0.00026136 | 0.00062982 |
| 4 | 0.00202393 | 0.00549620 |
| 8 | 0.00661413 | 0.01691747 |
| 16 | 0.02057538 | 0.03872029 |

受力样本一步误差降低约 58.5%。16 步 root position RMSE 约 0.83 mm，joint position RMSE 约 0.0051 rad。该结果限于浅蹲/站立保持附近，不能替代运动切换过程的模型准确性检验；Teacher 训练持续更新模型和 replay。

## 正式 Teacher 训练与验收

GPU 1，256 实际环境，WM batch=256×5、k=8，BPTT batch=256、H=32，UTD=0.4，原 Adam 与 critic 更新。初期关闭 DR，保留原 PD 和 50 Hz 控制。
5–30 N 竖直力作用 torso，0.5 秒平滑起落，随机起始时间/持续时间，25% 无力 episode，混合双向及可选反向第二次脉冲。

同一 checkpoint，从相同起始姿态分别运行：无力、有力、施加力但屏蔽 Teacher 的力输入、只给力 latent 的输入但不施加物理力。
评估时 25 N，1–4.5 秒施力，8–10 秒统计撤力后的状态。
要求双向高度改变 >4.5 cm、膝角变化 >0.25 rad、目标关节 RMSE <0.18 rad、不跌倒，并通过主动性消融。4 个重复是确定性数值重复，不宣称独立统计样本。

完整说明与命令：`experiments/exp02/README.md`；正式训练脚本 `experiments/exp02/train_teacher.sh` 自动恢复，且先检查基础验证结果。

```bash
cd /home/xl521/supermimic_force_transition
tail -f runs/exp02_teacher_seed0/console.log
cat runs/exp02_teacher_seed0/eval.jsonl
```

## 冻结模型的实际结果

独立进程、FP32 推理复测与训练内 BF16 评估均通过。

| 同一策略的场景 | 撤力后髋高 | 跌倒 |
|---|---:|---:|
| 站立起始、无力 | 约 0.783 m | 0/4 |
| 站立起始、向下 25 N | 约 0.700 m | 0/4 |
| 下蹲起始、无力 | 约 0.706 m | 0/4 |
| 下蹲起始、向上 25 N | 约 0.782 m | 0/4 |

屏蔽 Teacher 的力/力历史输入时，真实 ±25 N 不会导致端点切换；保留相同的延迟力信号但抑制物理外力时，策略仍主动切换。说明上述位移不能仅解释成被动压缩或拉起。

同一个 18 秒 episode，1–4.5 秒第一脉冲、9–12.5 秒反向脉冲：

| 幅值 | 站立→下蹲→站立 | 下蹲→站立→下蹲 |
|---|---|---|
| 10 N | 失败：能蹲下，但未能回站 | 失败：起立未触发 |
| 15 N | 通过，无跌倒 | 通过，无跌倒 |
| 25 N | 通过，无跌倒 | 通过，无跌倒 |
| 30 N | 通过，无跌倒 | 通过，无跌倒 |

### 尚未解决的限制

- 10 N 起立触发不足，不能宣称整个 5–30 N 范围都可靠。
- 25 N 下蹲过程存在明显过冲：最低髋高约 **0.587 m**，撤力后回到约 **0.700 m**；需要继续优化轨迹质量和受力持续时的姿态精度。
- 目前验证固定 torso 施力点、竖直方向、平地、关闭 DR。没有声称随机接触点、横向力、未知地形或真实机器人泛化。
- 四次复制是确定性数值重复，不是独立随机种子成功率。
- 冻结原控制器 + 学习力 latent 的受约束 Teacher 已验证，不等于完整通用 decoder 或 sensorless Student 已完成。

### 可查看的产物

- `reports/exp02/teacher_cycles.mp4`：保存的真实仿真状态渲染，四画面比较无力与往返力引导。
- `reports/exp02/teacher_2000.png`：四组核心对照的髋高、膝角、外力曲线。
- `reports/exp02/teacher_2000.json`：独立核心评估。
- `reports/exp02/cycles_2000.json`：连续往返及 10/15/25/30 N 测试。
- `reports/exp02/acceptance.json`：验收摘要与 checkpoint SHA256。
- `runs/exp02_teacher_seed0/STOPPED_AFTER_CORE_PASS.json`：主动停止记录；该目录没有自然训练完成的 DONE 标记。

复查命令：

```bash
source experiments/exp02/env.sh
"$SMIMIC_PY" -m supermimic.force_transition.evaluate --checkpoint runs/exp02_teacher_seed0/ckpt_0002000.pt --out reports/exp02/teacher_2000
"$SMIMIC_PY" experiments/exp02/evaluate_cycles.py
"$SMIMIC_PY" experiments/exp02/plot_evaluation.py reports/exp02/teacher_2000.npz
"$SMIMIC_PY" experiments/exp02/render_cycles.py
```

## 切换轨迹上的世界模型复核与校准

Teacher 2000 checkpoint 在新切换轨迹上的一步 force-aware/force-blind MSE 比为 0.9985，优势很弱（16 步仍有优势）。保持该 Teacher 的全部参数和 buffer **逐位不变**，补充 1000 次仅世界模型训练后，再次使用 seed 9137 的独立切换轨迹评估：

| 步数 | Force-aware MSE | Force-blind MSE |
|---|---:|---:|
| 1 | 0.00135266 | 0.00224278 |
| 4 | 0.00545038 | 0.01465454 |
| 8 | 0.01411514 | 0.04997644 |
| 16 | 0.05973386 | 0.16362505 |

校准后一步误差降低约 **39.7%**，通过预设 >=5% 增益阈值。校准模型保存于 `runs/exp02_wm_refine/last.pt`，其中 policy 与通过行为验收的 `ckpt_0002000.pt` 逐位相同，因此可用于后续继续研究。该校准任务已正常结束，所有本次训练任务均已停止。

复现校准：

```bash
source experiments/exp02/env.sh
"$SMIMIC_PY" -m supermimic.force_transition.train --out runs/exp02_wm_refine --resume runs/exp02_teacher_seed0/ckpt_0002000.pt --num-envs 256 --wm-batch 256 --pi-batch 256 --iterations 3000 --wm-only-updates 100000 --eval-every 0 --ckpt-every 1000 --wall-hours 1
"$SMIMIC_PY" -m supermimic.force_transition.evaluate_wm --checkpoint runs/exp02_wm_refine/last.pt --out reports/exp02/wm_refined.json
```
