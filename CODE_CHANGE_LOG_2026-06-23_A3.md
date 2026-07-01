# PACE-ICRA2026 A3 接入代码修改日志（2026-06-23）

## 阶段 0：阅读项目和 A3 资料，未修改源代码

### 用户目标

用户已经成功复现当前 T1 乒乓任务，命令如下：

```bash
CUDA_VISIBLE_DEVICES=1 python -m legged_lab.scripts.eval \
  --task=t1_tt_eval \
  --num_envs=64 \
  --load_run 2026-06-22_21-09-35 \
  --checkpoint model_9999.pt \
  --predictor \
  --headless
```

新的目标是在保持当前功能不变的前提下，把框架迁移到 A3 机器人。

### 本阶段动作

本阶段只做阅读、梳理和规划：

- 查看仓库状态和项目文件结构。
- 阅读 `t1_tt` 任务配置。
- 阅读 `TTEnv` 关键流程。
- 阅读 Booster T1/T1_TT asset 配置。
- 查看 A3 压缩包内容。
- 解析 A3 URDF 的 joints、links、质量和关键 body 名称。
- 新增迁移规划文档。
- 新增本代码修改日志文件。

### 仓库初始状态

在进行任何 A3 改动前，仓库已有大量 modified 文件：

```text
git diff --summary
```

显示主要是：

```text
mode change 100644 => 100755
```

`git diff --stat` 显示这些文件均为 0 行内容变化。后续不会回滚或整理这些已有权限变化。

### A3 资料结论

资料包：

```text
/mnt/ssd/zxl/a3_t2d5 .zip
```

包含：

- `a3_t2d5/urdf/model.urdf`
- `a3_t2d5/meshes/*.STL`
- `a3_t2d5/README.md`

不包含 USD。

URDF 解析结果：

- robot name：`A3T2.5`
- links：39
- joints：38
- revolute joints：31
- fixed joints：7
- total mass：约 `57.904204 kg`

关键 body / link：

- `pelvis_link`
- `torso_Link`
- `left_ankle_roll_Link`
- `right_ankle_roll_Link`
- `right_hand_Link`
- `head_pitch_Link`

README 说明：腰部和脚踝部分实际为并联结构，但 URDF 中已经做串联等效，训练可按串联处理。

### 当前代码中的 A3 迁移风险点

1. `t1_tt_config.py` 中关节名、body 名、reward 正则均按 T1 命名。
2. `TTEnv.compute_paddle_touch()` 固定 `paddle_index = 15`，A3 body 顺序不同，必须参数化。
3. `TTEnv.compute_paddle_touch()` 固定 paddle offset `[0.0, -0.345, 0.0]`，A3 需要单独设置。
4. `TTEnv.compute_intermediate_values()` 中 `body_height = 0.69`、`paddle_y_offset = -0.60` 和机器人尺寸相关。
5. A3 有 31 个 revolute joints，T1 当前策略为 21 动作维度，A3 不能直接加载 T1 checkpoint。
6. A3 原始 URDF 没有球拍模型，首版只能用虚拟击球点，后续可能需要附加 paddle collision/visual。

### 本阶段新增文件

- `A3_MIGRATION_PLAN_2026-06-23.md`
- `CODE_CHANGE_LOG_2026-06-23_A3.md`

### 本阶段未做事项

- 未修改任何 Python 源代码。
- 未注册 A3 task。
- 未复制或解压 A3 资产到仓库。
- 未运行训练或 eval。

## 下一阶段预定修改

1. 新增 A3 资产目录和 `A3_T2D5_CFG`。
2. 在 `RobotCfg` / `TTEnv` 中把 paddle body、paddle offset、未来目标 body 高度做成配置项，并保留 T1 默认行为。
3. 新增 `a3_tt` / `a3_tt_eval` 配置和 registry 注册。
4. 运行 T1 回归启动验证。
5. 运行 A3 smoke test，检查 URDF 转换、body/joint resolve 和初始姿态。

## 阶段 0.1：补充 A3 与 T1 差异分析，仍未修改源代码

### 本阶段动作

用户要求总结 A3 替换 T1 在本套工作中的不同之处，并分析是否需要单独优化。本阶段继续只修改 Markdown 文档，没有修改 Python 源码。

新增：

- `A3_VS_T1_ANALYSIS_2026-06-23.md`

更新：

- `A3_MIGRATION_PLAN_2026-06-23.md`
- `CODE_CHANGE_LOG_2026-06-23_A3.md`

### 分析结论摘要

1. A3 不能作为 T1 的简单模型路径替换；应新增 A3 专用 task。
2. T1 当前动作维度为 21，A3 URDF 有 31 个 revolute joints，策略网络输入/输出维度会变化，不能直接加载 T1 checkpoint。
3. A3 只有 URDF + STL，没有现成 USD，需要 URDF 导入或生成 USD。
4. 当前 `TTEnv.compute_paddle_touch()` 中的 T1 paddle body index 和 local offset 必须配置化，并为 A3 单独标定。
5. A3 的 body/joint 命名与 T1 完全不同，reward、reset、contact、feet、base mass randomization 都需要 A3 映射。
6. 乒乓球、球桌、空气动力、球路预测、runner 流程可以复用；机器人资产、初始姿态、actuator、动作尺度、reward 正则和击球点需要 A3 专门优化。

### 本阶段未做事项

- 未注册 `a3_tt` / `a3_tt_eval`。
- 未复制 A3 资产进仓库。
- 未修改任何 `.py` 文件。
- 未运行训练或 eval。

## 2026-07-01：新增 A3 stage5i 稳定增强 hitquality 对照版本

### 背景

WebRTC 可视化显示：

- `a3_tt_stage5h_hitquality` 已经能产生明显挥拍动作；
- `stage5g_wide` 和 `stage5h_hitquality` 都没有真正学会站稳，身体会向桌子方向前倾倒；
- 因此保留 `stage5h` 继续长训，同时新增一个独立 task 验证“站稳约束增强”是否能减少前扑式挥拍。

### 修改文件

- `legged_lab/mdp/rewards.py`
- `legged_lab/envs/a3_tt/a3_tt_config.py`
- `legged_lab/envs/__init__.py`
- `A3_TRAINING_COMPARISON_2026-07-01.md`

### 新增内容

新增 reward 函数：

- `penalty_a3_forward_fall_during_strike`

用途：

- 在击球窗口前后，根据 root x 前移、前向速度、躯干倾斜、base 高度下降惩罚“靠前扑接球”的策略；
- 只在新 task 中启用，不影响 T1 和已有 A3 task。

新增 task：

- `a3_tt_stage5i_stable_hitquality`
- `a3_tt_stage5i_stable_hitquality_eval`

新增配置：

- `A3Stage5iStableHitQualityRewardCfg`
- `A3Stage5iStableHitQualityEnvCfg`
- `A3Stage5iStableHitQualityEvalEnvCfg`
- `A3Stage5iStableHitQualityAgentCfg`

### 与 stage5h 的差异

- 保留 `stage5h` 的 A3 几何修正、宽球路、严格 contact、拍面法向、击球窗口、挥拍速度；
- 恢复并加强 stability score 的速度项；
- 增大 `reward_standing_stability`、`termination_penalty`、`flat_orientation_l2`、`penalty_unstable_hit`；
- 将 contact / hitquality / post-hit 奖励改为更严格的 stability gate；
- 降低挥拍速度 shaping 权重，避免继续强化不稳定挥拍。

### 不变项

- 未修改 T1 配置和任务注册；
- 未修改 `a3_tt_stage5h_hitquality` 已有配置；
- 未重新启用 ball curriculum。

## 阶段 1：A3 最小代码接入

### 修改目标

在不改变原有 `t1_tt` / `t1_tt_eval` workflow 的前提下，新增 A3 机器人版本的乒乓任务：

```text
a3_tt
a3_tt_eval
```

本阶段只做最小接入，使 A3 能进入同一套 `TTEnv`、table/ball、predictor runner 和 train/eval 脚本流程。A3 的站姿、击球点、PD、reward 权重仍属于后续仿真验证后的调参项。

### 新增 A3 原始资产

从：

```text
/mnt/ssd/zxl/a3_t2d5 .zip
```

解压到：

```text
legged_lab/assets/a3/t2d5/a3_t2d5/
```

新增内容包括：

- `urdf/model.urdf`
- `meshes/*.STL`
- `README.md`

没有覆盖或修改 T1/Booster 资产。

### 新增 A3 asset cfg

新增文件：

```text
legged_lab/assets/a3/__init__.py
legged_lab/assets/a3/a3.py
```

新增配置：

```python
A3_T2D5_CFG
```

实现要点：

1. 使用 `sim_utils.UrdfFileCfg` 读取 A3 URDF。
2. `root_link_name="pelvis_link"`。
3. `merge_fixed_joints=False`，保留 `right_hand_Link` 作为击球点 anchor。
4. `fix_base=False`。
5. `activate_contact_sensors=True`。
6. URDF 生成的 USD 输出目录为：

```text
legged_lab/assets/a3/t2d5/generated/A3_T2D5.usd
```

7. 使用 `DelayedPDActuatorCfg` 按 A3 URDF effort/velocity 初值分组配置：
   - waist
   - legs
   - feet
   - arms
   - wrists
   - head

### 公共 TTEnv 配置化修改

修改文件：

```text
legged_lab/envs/base/tt_config.py
legged_lab/envs/base/tt_env.py
```

在 `RobotCfg` 中新增字段，默认值保持当前 T1 行为：

```python
paddle_body_name: str = ""
paddle_body_index: int = 15
paddle_local_offset: tuple = (0.0, -0.345, 0.0)
future_body_height: float = 0.69
future_paddle_y_offset: float = -0.60
```

`TTEnv` 修改：

1. 初始化时优先按 `paddle_body_name` 查找 body id。
2. 如果 `paddle_body_name` 为空，则回退到 `paddle_body_index=15`。
3. `compute_paddle_touch()` 使用配置化后的 body id 和 local offset。
4. `compute_intermediate_values()` 使用配置化 `future_body_height` 和 `future_paddle_y_offset`。

边界保证：

- T1 没有设置 `paddle_body_name`，因此仍使用原来的 body index 15。
- T1 默认 offset 仍是 `(0.0, -0.345, 0.0)`。
- T1 默认 future target 参数仍是 `0.69` 和 `-0.60`。

### 新增 A3 task cfg

新增文件：

```text
legged_lab/envs/a3_tt/__init__.py
legged_lab/envs/a3_tt/a3_tt_config.py
```

新增类：

```python
A3TableTennisRewardCfg
A3TableTennisEnvCfg
A3TT_EvalEnvCfg
A3TableTennisAgentCfg
```

核心配置：

- `scene.robot = A3_T2D5_CFG`
- `height_scanner.prim_body_name = "torso_Link"`
- `feet_body_names = ["left_ankle_roll_Link", "right_ankle_roll_Link"]`
- `terminate_contacts_body_names = ["pelvis_link", "torso_Link"]`
- `paddle_body_name = "right_hand_Link"`
- `paddle_local_offset = (0.12, 0.0, 0.0)`
- `future_body_height = 0.90`
- `num_actions = 31`
- `num_joints = 31`
- `experiment_name = "a3_table_tennis"`

A3 关节顺序使用 URDF revolute joint 顺序，共 31 个关节。

### 任务注册

修改文件：

```text
legged_lab/envs/__init__.py
```

新增：

```python
task_registry.register("a3_tt", TTEnv, A3TableTennisEnvCfg(), A3TableTennisAgentCfg())
task_registry.register("a3_tt_eval", TTEnv, A3TT_EvalEnvCfg(), A3TableTennisAgentCfg())
```

原有注册保持不变：

```python
task_registry.register("t1_tt", TTEnv, T1TableTennisEnvCfg(), T1TableTennisAgentCfg())
task_registry.register("t1_tt_eval", TTEnv, T1TT_EvalEnvCfg(), T1TableTennisAgentCfg())
```

### 静态验证

已运行：

```bash
python3 -m py_compile \
  legged_lab/envs/base/tt_config.py \
  legged_lab/envs/base/tt_env.py \
  legged_lab/assets/a3/a3.py \
  legged_lab/assets/a3/__init__.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/a3_tt/__init__.py \
  legged_lab/envs/__init__.py
```

结果：通过。

## 2026-06-27 A3 stage4f/g/h reward 回调与诊断

目标：

- 保持 T1 与已有 A3 task 不变；
- 从当前最稳定的 stage4d `model_2549.pt` 回退接续；
- 验证“低 entropy/std + 强 hit/pass-net 主线 + 温和 own-table/stability 约束”是否能提升真实对面落台。

代码变更：

- `legged_lab/mdp/rewards.py`
  - 新增 `reward_post_hit_ballistic_landing_target()`；
  - 新增 `penalty_post_hit_low_base()`；
  - 新增 `penalty_post_hit_trajectory_excess()`。
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 `a3_tt_stage4f`：stage4d + ballistic landing target；
  - 新增 `a3_tt_stage4g`：降低 ballistic 权重，恢复/增强 hit velocity、net-clear、future-pass-net 主线，加入 low-base dense penalty；
  - 新增 `a3_tt_stage4h`：在 stage4g 上加入 post-hit over-arc / lateral-speed penalty，降低 vz 诱导。
- `legged_lab/envs/__init__.py`
  - 注册 `a3_tt_stage4f/g/h` 与对应 eval task。
- `legged_lab/scripts/a3_hit_outcome_diagnostics.py`
  - 允许 stage4f/g/h 任务；
  - 修正 own-table-after-hit 诊断口径：击球后必须先离开 own-table contact 区域，再次接触才计为 post-hit own-table。

关键实验：

```text
stage4d baseline:
  run: 2026-06-27_21-48-11_a3_stage4d_resume2500_std003_1024_50
  checkpoint: model_2549.pt
  first_hit=226/256
  actual_net_clear=198/256
  opponent_table_after_hit=31/256
  own_table_after_hit=220/256
  reset_low_z=2/256
  hit_vy mean=2.4963
  hit_vz mean=2.5339
  actual_z_at_net mean=1.3874
```

```text
stage4e strong own-table ramp:
  run: 2026-06-27_22-35-16_a3_stage4e_from4d2549_std003_1024_50
  checkpoint: model_2598.pt
  opponent_table_after_hit=22/256
  own_table_after_hit=214/256
  reset_low_z=60/256
```

```text
stage4f ballistic landing target:
  run: 2026-06-27_22-50-08_a3_stage4f_from4d2549_std003_1024_50
  checkpoint: model_2598.pt
  first_hit=239/256
  actual_net_clear=226/256
  opponent_table_after_hit=29/256
  own_table_after_hit=225/256
  reset_low_z=46/256
  hit_vy mean=1.9219
  hit_vz mean=3.3230
  actual_z_at_net mean=1.5643
```

```text
stage4g conservative rollback:
  run: 2026-06-27_23-01-38_a3_stage4g_from4d2549_std003_1024_50
  checkpoint: model_2598.pt
  first_hit=240/256
  actual_net_clear=221/256
  opponent_table_after_hit=32/256
  own_table_after_hit=223/256
  reset_low_z=47/256
  reset_y_high=27/256
  hit_vy mean=1.8617
  hit_vz mean=3.1833
  actual_z_at_net mean=1.5330
```

```text
stage4h arc/lateral penalty:
  run: 2026-06-27_23-09-58_a3_stage4h_from4d2549_std003_1024_50
  checkpoint: model_2598.pt
  first_hit=234/256
  actual_net_clear=220/256
  opponent_table_after_hit=24/256
  own_table_after_hit=217/256
  reset_low_z=41/256
  reset_y_high=22/256
  hit_vy mean=1.9997
  hit_vz mean=3.2941
  actual_z_at_net mean=1.5454
```

判断：

- stage4f/g/h 都保持了高触球和过网，但没有稳定提升真实对面落台；
- stage4g 只比 stage4d 多 1 个对面落台样本，同时引入明显低身体 reset；
- stage4h 的球路惩罚没有成功降低过网高度和横向速度，反而降低了 opponent-table；
- 不建议直接从 stage4f/g/h 的 `model_2598.pt` 长训；
- 当前稳定回退点仍是 stage4d `model_2549.pt`；
- 下一步应转向 A3 专用 curriculum/几何回查：先训练固定或更窄中心球路的低弧线对面落台，再逐步放宽球路随机范围。

验证：

```text
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py \
  legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

结果：通过。

## 2026-06-27 stage4b/c/d/e reward rollback and probes

目标：

- 保留 T1 workflow 不变；
- A3-only 回调 reward：先恢复 hit velocity / pass-net 信号，再逐步加 own-table 和击球后稳定约束；
- 继续使用低 entropy / 低 policy std 接续，避免 A3 窄击球窗口被 PPO 采样噪声冲散。

代码变更：

- `legged_lab/mdp/rewards.py`
  - 新增 `reward_post_hit_net_progress()`；
  - 该 reward 在触拍后、首个桌面接触前提供短窗口轨迹信号；
  - 后续增加 `max_reward_x`、`vz_target`、`vz_weight` 参数，避免 stage4c 的 late-hit 奖励偏置。
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 `a3_tt_stage4b/c/d/e` 对应 reward/env/agent cfg；
  - stage4c：首次加入 post-hit short-horizon shaping；
  - stage4d：限制 late-hit，降低 post-hit 权重，强调 vz / net height；
  - stage4e：在 stage4d 基础上小幅加强 own-table / opponent landing 约束。
- `legged_lab/envs/__init__.py`
  - 注册 `a3_tt_stage4b/c/d/e` 和 eval 任务。
- `legged_lab/scripts/a3_hit_outcome_diagnostics.py`
  - 允许诊断 `a3_tt_stage4b/c/d/e`。

验证：

```text
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py \
  legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

结果：通过。

stage4b 4096-env probe：

```text
run: 2026-06-27_21-20-39_a3_stage4b_resume2500_100
resume: 2026-06-27_00-56-01_a3_stage2_moderated_long_10000/model_2500.pt
reset_policy_noise_std: 0.15

step 2549:
  reward_contact recent10: 0.0219666
  reward_hit_ball_velocity_net recent10: 0.000222071
  reward_future_pass_net recent10: 4.17889e-14
  reward_table_success recent10: 0.0
  termination_penalty recent10: -2.0
```

结论：低强惩罚回调不够，采样训练仍拿不到过网/对桌信号。

stage4b std=0.05 probe：

```text
run: 2026-06-27_21-26-48_a3_stage4b_resume2500_std005_100
step 2529:
  mean_noise_std: 0.0512585
  reward_contact recent10: 0.0203193
  reward_future_pass_net max: 9.70704e-07
  reward_table_success: 0.0
  termination_penalty recent10: -1.97553
```

结论：仅降低 std 不够，需要把 first-hit 一帧式 reward 扩展成短窗口轨迹信号。

stage4c 结果：

```text
smoke: 2026-06-27_21-33-08_a3_stage4c_smoke_20
  reward_post_hit_net_progress avg: 0.132175

probe: 2026-06-27_21-41-18_a3_stage4c_resume2500_std003_1024_50
  reward_future_pass_net recent10: 0.076942
  reward_table_success recent10: 0.0381693
  reward_post_hit_net_progress recent10: 0.443015

diagnostic model_2549:
  first_hit: 113/256
  actual_net_clear: 3/256
  opponent_table_after_hit: 1/256
  own_table_after_hit: 113/256
  hit_x mean: -0.8552
  hit_vz mean: 0.0824
```

结论：TensorBoard 任务曲线变好，但确定性物理结果变差。原因判断为 post-hit x-progress 奖励鼓励 late-hit，已作为失败对照保留。

stage4d 结果：

```text
smoke: 2026-06-27_21-47-19_a3_stage4d_resume2500_std003_128_smoke
  reward_future_pass_net last: 0.091608
  reward_post_hit_net_progress avg: 0.0477198
  reward_table_success avg: 0.012

probe: 2026-06-27_21-48-11_a3_stage4d_resume2500_std003_1024_50
  reward_contact recent10: 0.152745
  reward_hit_ball_velocity_net recent10: 0.470615
  reward_future_pass_net recent10: 0.215799
  reward_table_success recent10: 0.0599294
  reward_future_opponent_landing recent10: 0.0888099
  penalty_actual_own_table_after_hit recent10: -0.117835
  termination_penalty recent10: -0.461193
  undesired_contacts recent10: -51.1829

diagnostic model_2549:
  first_hit: 226/256
  actual_net_clear: 198/256
  opponent_table_after_hit: 31/256
  own_table_after_hit: 226/256
  reset_low_z: 2/256
  hit_x mean: -1.2227
  hit_vz mean: 2.5339
```

结论：stage4d 明显优于 stage4c，恢复了过网能力并显著改善击球后低姿态 reset；但 own-table-first 仍是主失败模式，不能直接长训。

stage4e 初步结果：

```text
smoke: 2026-06-27_21-53-48_a3_stage4e_resume2500_std003_128_smoke
  reward_future_pass_net last: 0.0821198
  reward_future_opponent_landing avg: 0.0454656
  reward_table_success avg: 0.011
  reward_post_hit_net_progress avg: 0.0464933
```

1024-env probe `a3_stage4e_resume2500_std003_1024_50` 未进入训练，Isaac 初始化阶段报：

```text
malloc(): invalid size (unsorted)
```

没有 Python traceback，也没有生成 run 目录；判断为当前 Isaac/Kit 连续启动后的初始化层问题，不作为 stage4e reward 失败证据。当前没有残留训练进程。

当前判断：

- stage4d 是目前最好的可复现实验点；
- stage4e 只完成 128 smoke，尚未完成 1024/4096 probe；
- 下一步应在清理/重启 Isaac 状态后继续 stage4e 1024-env probe，或者先用 stage4d 作为保守基线做 250-500 iter 延长观察；
- 仍不建议直接启动 10000 iter 长训。

## 2026-06-27 A3 stage-2 moderated 长训启动

用户决定：

- 250/500 iter 只能说明早期趋势，不能直接判定 A3 在原始 PACE 流程下无法学会；
- 先按 T1 的长训思路做一次同尺度验证，再决定是否引入 A3 专项 swing curriculum 或更强动作先验。

启动命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage2_moderated_long_10000 \
  --headless \
  --predictor
```

运行方式：

```text
tmux session: a3_stage2_moderated_long_10000
console log: logs/a3_table_tennis/a3_stage2_moderated_long_10000_console.log
run dir: logs/a3_table_tennis/2026-06-27_00-56-01_a3_stage2_moderated_long_10000
tensorboard: http://127.0.0.1:16006
```

早期观察：

```text
iteration: 25/10000
reward_contact: 0.0486
reward_hit_ball_velocity_net: 0.0569
reward_future_landing_x_progress: 0.0623
penalty_actual_own_table_after_hit: -0.2316
reward_hit_net_clearance_progress: 0.0000
reward_future_pass_net: 0.0000
reward_table_success: 0.0000
undesired_contacts: -60.6496
termination_penalty: -0.0583
```

当前判断：

- 训练进程正常，早期已经有触球和向球路推进信号；
- `future_pass_net/table_success` 在前几十 iter 为 0 暂不作为失败结论；
- 下一步继续观察 60-80、250、500、1000、2000 iter 的曲线变化，特别关注是否从“触球/己方落点”过渡到“过网/对面桌成功”。

58 iter 监督更新：

```text
iteration: 58/10000
mean reward: -401.60
mean episode length: 179.25
reward_contact: 0.0430
reward_hit_ball_velocity_net: 0.0519
reward_future_landing_x_progress: 0.0537
penalty_actual_own_table_after_hit: -0.1884
reward_hit_net_clearance_progress: 0.0000
reward_future_pass_net: 0.0000
reward_table_success: 0.0000
undesired_contacts: -42.4535
termination_penalty: -0.4602
```

判断：

- `undesired_contacts` 较 25 iter 明显改善，未出现数值崩溃；
- 触球和击球速度奖励仍在，说明长训入口没有失去任务信号；
- `termination_penalty` 变重，需要在 250/500 iter 再看是否只是早期探索波动；
- 过网和上对面桌仍为 0，当前阶段继续训练，不提前判定失败。

长训完成与结果分析：

```text
run: logs/a3_table_tennis/2026-06-27_00-56-01_a3_stage2_moderated_long_10000
checkpoint: model_9999.pt
event file: events.out.tfevents.1782492961.FdseRobot-02.1862628.0
training wall time: about 10h52m
```

TensorBoard 关键趋势：

```text
iter 25:
  mean_reward=-619.40
  reward_contact=0.04864
  reward_hit_ball_velocity_net=0.05695
  reward_future_pass_net=0.0000004
  reward_table_success=0.0000
  undesired_contacts=-60.65
  termination_penalty=-0.058

iter 2000:
  mean_reward=2.406
  reward_contact=0.1664
  reward_hit_ball_velocity_net=0.2786
  reward_future_pass_net=0.01394
  reward_table_success=0.00842
  undesired_contacts=-0.011
  termination_penalty=-0.675

iter 9999:
  mean_reward=9.509
  reward_contact=0.00145
  reward_hit_ball_velocity_net=0.00378
  reward_future_pass_net=0.00126
  reward_table_success=0.00208
  undesired_contacts=0.000
  termination_penalty=-0.006
```

全程稀疏任务信号峰值：

```text
reward_contact max=0.17683 at iter 2065
reward_hit_ball_velocity_net max=0.30986 at iter 2109
reward_hit_net_clearance_progress max=0.03351 at iter 2416
reward_future_pass_net max=0.07142 at iter 2411
reward_table_success max=0.07136 at iter 2408
```

物理诊断命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_hit_outcome_diagnostics \
  --task=a3_tt \
  --num_envs=128 \
  --max_steps=180 \
  --load_run 2026-06-27_00-56-01_a3_stage2_moderated_long_10000 \
  --checkpoint model_XXXX.pt \
  --predictor \
  --headless \
  --enable_cameras False
```

诊断结果：

```text
model_2000.pt:
  first_hit=104/128
  actual_crossed_net=92/128
  actual_net_clear=19/128
  opponent_table_after_hit=10/128
  own_table_after_hit=104/128
  reset_low_z=121/128
  hit_vx mean=4.7766
  hit_vy mean=1.0214
  hit_vz mean=-8.9834
  actual_z_at_net mean=0.9861

model_2500.pt:
  first_hit=128/128
  actual_crossed_net=128/128
  actual_net_clear=128/128
  opponent_table_after_hit=1/128
  own_table_after_hit=128/128
  reset_low_z=54/128
  reset_y_high=77/128
  hit_vx mean=5.8738
  hit_vy mean=1.5209
  hit_vz mean=3.2743
  actual_z_at_net mean=1.5175

model_9999.pt:
  first_hit=128/128
  actual_crossed_net=128/128
  actual_net_clear=128/128
  opponent_table_after_hit=10/128
  own_table_after_hit=128/128
  reset_low_z=128/128
  hit_vx mean=3.8618
  hit_vy mean=0.1200
  hit_vz mean=4.4343
  actual_z_at_net mean=1.9327
```

结论：

- 10000 iter 长训已完成；
- A3 不是完全没有学会挥拍，固定中心球路下 `model_2500.pt` 和 `model_9999.pt` 都能稳定触球、向前打球、过网且有净空；
- 失败点是落点和身体稳定性：诊断中 `own_table_after_hit=128/128`，说明球几乎总是先落己方桌；`model_9999.pt` 的 `reset_low_z=128/128`，说明策略击球后基本会低姿态/摔倒 reset；
- TensorBoard 中任务稀疏奖励在 2000-2500 iter 出现过峰值，随后被更容易优化的稳定/短 episode 行为压下去，最终 checkpoint 不是最好的任务 checkpoint；
- 直接继续从 `model_9999.pt` 长训意义不大，下一步应围绕 A3-only 的落点奖励、己方落台判定/惩罚时序、击球后站稳约束、以及选择 2000-2500 附近 checkpoint 做接续/对照来推进。

## 2026-06-27 A3 hit-rate cliff 分析与 stable-return 试验

问题：

- 用户观察到训练中击球率突然断崖下跌；
- 需要分析原因并尝试 A3-only 调整，不能影响 T1 和已有 `a3_tt` 基线。

TensorBoard 分析：

```text
reward_contact:
  max=0.17683 at iter 2065
  iter 2315: 0.08567
  iter 2565: 0.01913
  iter 2815: 0.00541

reward_hit_ball_velocity_net:
  max=0.30986 at iter 2109
  iter 2609: 0.02976

reward_future_pass_net:
  max=0.07142 at iter 2411
  iter 2661: 0.01007

reward_table_success:
  max=0.07136 at iter 2408
  iter 2658: 0.01023

Policy/mean_noise_std:
  init: about 0.10
  around 2000-2500: about 1.09-1.18
```

补充随机球路确定性诊断：

```text
model_2500.pt random ball:
  first_hit=253/256
  actual_crossed_net=251/256
  actual_net_clear=230/256
  opponent_table_after_hit=29/256
  own_table_after_hit=253/256
  reset_low_z=108/256

model_9999.pt random ball:
  first_hit=249/256
  actual_crossed_net=247/256
  actual_net_clear=231/256
  opponent_table_after_hit=53/256
  own_table_after_hit=249/256
  reset_low_z=256/256
```

判断：

- TensorBoard 训练曲线使用 PPO 采样动作，会受 policy std 影响；诊断使用 inference policy，反映均值动作；
- A3 并不是完全忘记击球，确定性策略在随机球路仍能高比例触球/过网；
- 训练曲线中的 hit-rate cliff 与 policy std 被 entropy 推高高度相关；
- 真正主失败模式是 `own_table_after_hit` 几乎等于 first-hit，同时击球后低姿态 reset；
- 当前 reward 对“触球/过网趋势/高球”给了足够信号，但对“对面桌真实落点”和“击球后不摔”不够强。

代码变更：

- `legged_lab/mdp/rewards.py`
  - 新增 `reward_opponent_table_after_paddle_hit_target()`；
  - 新增 `penalty_hit_low_base_reset()`。
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 `A3StableReturnRewardCfg`；
  - 新增 `A3StableReturnEnvCfg` / `A3StableReturnEvalEnvCfg`；
  - 新增 `A3StableReturnAgentCfg`，A3 stable PPO 使用 `entropy_coef=0.0005`、`learning_rate=5e-4`、`desired_kl=0.006`。
- `legged_lab/envs/__init__.py`
  - 注册 `a3_tt_stable`；
  - 注册 `a3_tt_stable_eval`。
- `legged_lab/scripts/train.py`
  - 新增 `--no_load_optimizer`；
  - 新增 `--reset_policy_noise_std VALUE`。
- `legged_lab/scripts/a3_hit_outcome_diagnostics.py`
  - 允许诊断 `a3_tt_stable` / `a3_tt_stable_eval`。

验证：

```text
py_compile:
  legged_lab/mdp/rewards.py
  legged_lab/envs/a3_tt/a3_tt_config.py
  legged_lab/envs/__init__.py
  legged_lab/scripts/train.py
  legged_lab/scripts/a3_hit_outcome_diagnostics.py
结果：通过
```

从零 smoke：

```text
run: logs/a3_table_tennis/2026-06-27_19-57-38_a3_stable_smoke_20
task: a3_tt_stable
num_envs: 64
max_iterations: 20
checkpoint: model_19.pt
结果：runner 正常完成，新 reward 项进入日志。
```

从 `model_2500.pt` 接续 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stable \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=5 \
  --run_name=a3_stable_resume2500_smoke_5 \
  --resume True \
  --load_run 2026-06-27_00-56-01_a3_stage2_moderated_long_10000 \
  --checkpoint model_2500.pt \
  --no_load_optimizer \
  --reset_policy_noise_std 0.15 \
  --headless \
  --predictor
```

结果：

```text
run: logs/a3_table_tennis/2026-06-27_19-58-57_a3_stable_resume2500_smoke_5
checkpoint: model_2504.pt
Policy/mean_noise_std: 0.15057 -> 0.15060
reward_contact: 0.0000 -> 0.0742
reward_future_opponent_landing: 0.0000 -> 0.2521
reward_future_pass_net: 0.0000 -> 0.1057
reward_table_success: 0.0000 -> 0.4375
reward_actual_opponent_table_target: 0.0000 -> 0.1669
penalty_actual_own_table_after_hit: 0.0000 -> -0.1208
```

当前判断：

- 新 task 与训练入口可以正常工作；
- 重置 std 后，接续训练没有立即重演 hit-rate cliff；
- 5 iter 只能说明 wiring 和方向可行，不能说明最终改善成功；
- 下一步应跑 `a3_tt_stable` 从 `model_2500.pt` 接续的 4096 env / 500 iter probe，并对比 `own_table_after_hit`、`opponent_table_after_hit`、`reset_low_z`。

注意：

- 一次 256 env 的 `a3_tt_stable` 诊断卡在 Isaac `Building environment...`，已强制清理该诊断进程；
- 因为训练 smoke 已经完成，暂不把这次诊断卡住视为代码失败。

## 2026-06-27 A3 stable 4096-env probe 监督结果

启动命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stable \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_stable_resume2500_500 \
  --resume True \
  --load_run 2026-06-27_00-56-01_a3_stage2_moderated_long_10000 \
  --checkpoint model_2500.pt \
  --no_load_optimizer \
  --reset_policy_noise_std 0.15 \
  --headless \
  --predictor
```

运行信息：

```text
run: logs/a3_table_tennis/2026-06-27_20-34-34_a3_stable_resume2500_500
tmux session: a3_stable_resume2500_500
console log: logs/a3_table_tennis/a3_stable_resume2500_500_console.log
start iteration: 2500/3000
stopped at: about 2573/3000
```

监督结果：

```text
Policy/mean_noise_std:
  last=0.14991 at iter 2573
  max=0.15049

Train/mean_reward:
  last=-14.4892 at iter 2573
  max=6.8988 at iter 2502

reward_contact:
  last=0.01826 at iter 2573
  max=0.03684 at iter 2502

reward_future_opponent_landing:
  last=0.0
  max=0.0

reward_future_pass_net:
  last=0.0
  max=1.0e-10

reward_table_success:
  last=0.0
  max=0.0

reward_actual_opponent_table_target:
  last=0.0
  max=0.0

penalty_actual_own_table_after_hit:
  last=0.0
  max=0.0

termination_penalty:
  last=-2.0
```

处理：

- 在约 2573 iter 提前停止；
- 未生成新的后续 checkpoint，目录中只有 resume 时保存的 `model_2500.pt` 和 event 文件；
- 没有残留训练进程。

判断：

- `--reset_policy_noise_std 0.15` 生效，std 没有再上冲；
- 但 4096-env 大批量训练没有复现 64-env smoke 中的 early `table_success` 信号；
- 当前 stable reward 版本过强地惩罚/约束失败模式后，策略反而快速进入“少触球、短 episode、没有过网/对面桌信号”的状态；
- 下一步不应继续这次 probe，应回调 stable reward：降低接续初期的强惩罚，保留低 entropy/std 控制，并恢复更强的 hit velocity / pass-net curriculum，再分阶段把 own-table 和 post-hit stability 惩罚加上去。

额外检查 A3 action/observation 关节列表与 URDF：

```text
config_joint_count 31
urdf_revolute_count 31
missing_in_urdf []
extra_in_urdf []
same_order True
```

### 尚未完成的运行验证

此前默认 shell 未激活 `zxl-pace`，无法完成 URDF 导入和环境实例化验证。后续已确认应使用：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python
```

用户已在 `zxl-pace` 环境中运行 A3 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --headless \
  --predictor
```

实际命令为：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --headless \
  --predictor
```

结果：用户反馈可以正常运行。

这说明本阶段的 A3 最小接入已经通过第一轮运行验证：

- `a3_tt` task registry 可找到；
- A3 URDF 资产可以被 IsaacLab/IsaacSim 路径加载/转换；
- A3 action/observation joint 配置至少能完成初始化；
- `TTEnv`、table/ball、predictor runner 与 A3 task 可以进入训练流程。

尚未运行 T1 回归 eval。下一步建议运行原 `t1_tt_eval` 命令，确认 T1 workflow 在运行时没有被公共配置化改动影响。

### T1 回归验证

用户随后运行原 T1 eval 命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.eval \
  --task=t1_tt_eval \
  --num_envs=64 \
  --load_run 2026-06-22_21-09-35 \
  --checkpoint model_9999.pt \
  --predictor \
  --headless
```

结果：用户反馈可以正常运行。

这说明本阶段公共代码改动没有破坏原 T1 eval workflow。当前阶段已有两项运行验证：

1. A3 `a3_tt` smoke test 可正常启动训练流程。
2. T1 `t1_tt_eval` 使用原 checkpoint 可正常运行。

## A3 baseline 1000-iter experiment analysis

用户完成一次 A3 训练实验：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1000 \
  --run_name=a3_baseline_500 \
  --headless \
  --predictor
```

日志目录：

```text
logs/a3_table_tennis/2026-06-23_21-49-47_a3_baseline_500
```

保存 checkpoint：

```text
model_0.pt
model_250.pt
model_500.pt
model_750.pt
model_999.pt
```

TensorBoard 关键指标：

```text
Train/mean_reward: step 0 -> 999, -10.878 -> -31.912, last100 mean -32.004
Train/mean_episode_length: step 0 -> 999, 6.667 -> 63.09, last100 mean 62.39
Train/TT_hit_rate: step 999 = 0.001236
Train/TT_success_rate: step 999 = 0.0
Episode_Reward/reward_contact: last100 mean 0.000129
Episode_Reward/reward_table_success: last100 mean 0.0
Episode_Reward/termination_penalty: last100 mean -2.0
Episode_Reward/undesired_contacts: last100 mean -0.456
Loss/value_function: step 0 -> 999, 16.649 -> 0.608
Loss/predictor_mse: step 0 -> 20, 1.630 -> 0.074
Perf/total_fps: last100 mean 882.24
```

对比 T1 参考 run `logs/t1_table_tennis/2026-06-22_21-09-35`：

```text
T1 step999 TT_hit_rate = 0.0550, TT_success_rate = 0.00152
T1 step999 reward_contact = 0.0352, reward_table_success = 0.00279
T1 step999 termination_penalty = -0.255
T1 step999 mean_episode_length = 276.8

A3 step999 TT_hit_rate = 0.00124, TT_success_rate = 0.0
A3 step999 reward_contact = 0.0
A3 step999 reward_table_success = 0.0
A3 step999 termination_penalty = -2.0
A3 step999 mean_episode_length = 63.09
```

结论：

- 训练程序稳定，日志、checkpoint、predictor 训练和 PPO loss 都正常。
- A3 当前不是程序崩溃问题，而是任务链路未打通：几乎没有有效击球，未出现对方台面成功回球。
- A3 的非 timeout reset 频率明显高于 T1，episode length 明显短，说明站位/初始姿态/可行动作空间/终止边界需要优先标定。
- `reward_future_dis_ee` 有正值，但 `reward_contact` 接近零，说明策略有一定朝未来击球点靠近的趋势，但右手 paddle touch point、右臂可达性或初始站位仍未对准。

下一轮建议只做 A3 专用优化，不改 T1 workflow：

1. 标定 `right_hand_Link` 的 `paddle_local_offset`，必要时使用可视化/调试输出来确认虚拟击球点是否在真实球拍中心。
2. 收窄或重设 A3 `reset_base` 初始站位，使 base 不容易触发 `x > -1.35`、`y` 越界或低高度 reset。
3. 检查 A3 初始右臂姿态和 `action_scale=0.20`，提高早期可达击球点概率。
4. 若仍频繁 termination，再单独调 A3 的 `terminate` 边界或稳定性相关 reward/PD，不改公共 `TTEnv` 默认行为。

## A3 ping-pong paddle URDF integration

用户补充新的 A3 乒乓专用 URDF：

```text
/mnt/ssd/zxl/095dabe4-ebcc-4666-80d9-ab49f41fbaa8.zip
```

检查结果：

```text
robot: 0000014503_A3T2.5-URDF-std-pingpang-0409
links: 43
joints: 42
revolute joints: 31
fixed joints: 11
total mass: 58.27723163 kg
```

动作关节仍为原 A3 的 31 个 revolute joints，顺序不变。新增乒乓拍链路：

```text
right_wrist_yaw_Link
  -> right_hand_pingpang_Link
     -> pingpang_red_Link
     -> pingpang_black_Link
     -> pingbang_ball_Link
```

供应方给出的击球点 marker fixed joint：

```text
pingbang_ball_joint
parent = right_hand_pingpang_Link
child = pingbang_ball_Link
origin xyz = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
origin rpy = (0, 0, 0)
```

新增资产目录：

```text
legged_lab/assets/a3/t2d5_pingpang/a3_t2d5_pingpang/
```

处理内容：

- 保留原始 `URDF-JOINT-LINK.urdf`。
- 新增用于 IsaacLab 的 `urdf/model.urdf`。
- 将 `model.urdf` 中的 `package://.../meshes/` 替换为相对路径 `../meshes/`，避免 standalone IsaacLab 运行时依赖 ROS package 解析。

代码修改：

- `legged_lab/assets/a3/a3.py`
  - 新增 `A3_T2D5_PINGPANG_URDF_PATH`。
  - 新增 `A3_T2D5_PINGPANG_CFG`，复用原 A3 PD/初始姿态/actuator 配置，只替换 URDF asset path 和生成 USD 路径。
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - `a3_tt` 改为加载 `A3_T2D5_PINGPANG_CFG`。
  - `paddle_body_name = "right_hand_pingpang_Link"`。
  - `paddle_local_offset = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)`。

设计选择：

- 不直接使用 `pingbang_ball_Link` 作为 `paddle_body_name`，因为该 link 质量为 0，固定 link 在 URDF 转换后是否稳定作为 articulation body 保留需要额外验证。
- 首版使用 `right_hand_pingpang_Link + marker offset`，更接近 T1 的 `right_hand_link + marker_ball offset` 机制。
- 原 T1 workflow 不变；原裸 A3 配置 `A3_T2D5_CFG` 保留。

URDF 清理：

- 只修改复制到仓库中的 `urdf/model.urdf`，保留原始 `URDF-JOINT-LINK.urdf`。
- 将 robot name 从 `0000014503_A3T2.5-URDF-std-pingpang-0409` 改为 `A3T2_5_pingpang`，避免 USD prim path 非法字符警告。
- 将 fixed joint 中空/零 `axis` 改为 `1 0 0`，避免 URDF importer 的 `Could not parse xyz` / `xyz not specified for axis` 错误；fixed joint 的 axis 不参与运动学控制。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/assets/a3/__init__.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/a3_tt/__init__.py \
  legged_lab/envs/__init__.py
```

结果：通过。

Isaac AppLauncher 配置检查：

```text
config_joint_count 31
urdf_revolute_count 31
same_order True
missing []
extra []
right_hand_pingpang_Link exists True
pingbang_ball_Link exists True
cfg_robot_asset legged_lab/assets/a3/t2d5_pingpang/a3_t2d5_pingpang/urdf/model.urdf
cfg_paddle_body_name right_hand_pingpang_Link
cfg_paddle_local_offset (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
cfg_num_actions 31
```

1-iter smoke train：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_pingpang_smoke_clean \
  --headless \
  --predictor
```

结果：退出码 0。修正后的 URDF 不再出现 importer 的 axis/xyz 解析错误。日志中仍有 Isaac/headless 环境的 GLFW/GPU warning，不影响本次资产导入结论。

## 2026-06-24 A3 pingpong baseline 1000 训练复盘

本次未修改代码，仅补充训练诊断和后续优化判断。

用户训练 run：

```text
logs/a3_table_tennis/2026-06-24_16-15-10_a3_pingpang_baseline_1000
```

核心 TensorBoard 指标：

```text
Train/TT_hit_rate last ~= 0.000944584
Train/TT_success_rate = 0
Episode_Reward/reward_contact last = 0
Episode_Reward/reward_table_success = 0
Episode_Reward/termination_penalty tail100 = -2
Episode_Reward/reward_future_dis_ee tail100 ~= 0.022451
Loss/predictor_mse last ~= 0.062777
```

运行时几何检查结论：

- `right_hand_pingpang_Link + paddle_local_offset` 与 `pingbang_ball_Link` 的 world position 完全一致，说明 A3 乒乓 URDF 的 marker/offset 已经正确生效。
- reset 后 `paddle_touch_point` 明显低于预测击球点；零动作 rollout 的最近 `paddle-ball` 距离约 `0.298689m`，远大于当前虚拟接触奖励的有效范围。
- 因此当前训练失败不再主要归因于 URDF 未标定，而是 A3 初始姿态、右臂可达区域、球路/reset 分布和 T1 奖励课程不匹配。

后续建议：

1. 在 A3 配置中先调整右臂/手腕初始姿态，让 reset 后球拍中心接近预测击球点。
2. 初期收窄球路随机化，先训练单一可达球路。
3. 增加 A3-only dense shaping/debug logging，例如 episode 内最小 `paddle-ball distance`、最小 `paddle-future distance`、termination reason。
4. 再根据指标调整 A3 的 reward weights、right-arm regularization、PD/action scale。

## 2026-06-24 A3 pose calibration script

新增 A3-only 姿态标定脚本：

```text
legged_lab/scripts/a3_pose_calibration.py
```

用途：

- 单环境启动 `a3_tt_eval`。
- 固定 base reset、右臂 reset noise 和球路，便于观察姿态。
- 使用已有绿色 `BallFuture` marker 显示 `ball_future_pose`。
- 使用已有黄色 `BallPred` marker 显示 `paddle_touch_point`。
- 周期性打印：

```text
ball_future_pose
paddle_touch_point
ball_pos
paddle_to_future_dist
paddle_to_ball_dist
ball_future_t
right_arm_joint_pos
```

支持通过命令行覆盖右臂默认姿态：

```bash
--joint right_elbow_joint=0.8
--rsp 0.1 --rsr -0.2 --rsy 0.0 --re 0.8 --rwp -0.2
```

后续增强：

- 新增 `--pin_base`，仅用于标定脚本，将 robot root pose/velocity 每步固定在 reset 姿态，避免自由基座倒下阻塞右臂几何观察。
- 状态打印新增 `base_pos_w` 和 `base_rpy(rad)`，用于判断自由站立稳定性。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/scripts/a3_pose_calibration.py
```

结果：通过。

短 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_pose_calibration \
  --task=a3_tt_eval \
  --num_envs=1 \
  --max_steps=3 \
  --print_interval=1 \
  --no_visual_markers \
  --headless
```

结果：退出码 0。终端打印显示当前默认姿态在固定球路下：

```text
paddle_to_future_dist ~= 0.50m
```

这说明 A3 当前默认右臂/球拍预姿态离 `ball_future_pose` 仍明显偏远，后续应使用该脚本迭代右臂候选姿态。

`--pin_base` smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_pose_calibration \
  --task=a3_tt_eval \
  --num_envs=1 \
  --max_steps=3 \
  --print_interval=1 \
  --no_visual_markers \
  --headless \
  --pin_base
```

结果：退出码 0。`base_pos_w` 保持 `[-1.8600, 0.3500, 0.9000]`，`base_rpy(rad)` 保持 `[0, 0, 0]`。

## 2026-06-25 A3 standing calibration task

用户提供一张目标姿态参考图：机器人下蹲、两脚分开、右臂持拍在身体前方。当前 A3 默认姿态会在 Isaac Sim 中摔倒，因此在改训练配置前，先新增 A3-only 站立姿态候选筛选工具。

计划新增：

```text
legged_lab/scripts/a3_standing_calibration.py
```

设计边界：

- 只用于 A3 姿态诊断，不修改 `a3_tt` / `a3_tt_eval` 默认配置。
- 保持 T1 workflow 不变。
- 自动生成接近参考图的下蹲持拍候选姿态。
- 并行测试候选在自由基下是否能站稳。
- 输出排序结果和可复制到 A3 config 的 `joint_pos` 候选。

评估指标：

```text
survival_steps
fallen / reset
max_abs_roll_pitch
root_z_drift
both_feet_contact_ratio
foot_slip
paddle_to_future_dist
```

实际修改：

- 新增 `legged_lab/scripts/a3_standing_calibration.py`。
- 生成参考图风格的 ready-arm crouch candidates。
- 后续补充 zero/soft/light leg priority candidates。
- 后续补充 default/zero/ready arm modes，用于区分站立不稳是否来自持拍姿态。
- 标定脚本内关闭 push/action delay/perception delay，关闭质量随机化，将摩擦随机化固定，避免训练 domain randomization 干扰站姿诊断。
- 输出 CSV 到 `logs/a3_standing_calibration/`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/scripts/a3_standing_calibration.py
```

结果：通过。

最小 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_standing_calibration \
  --task=a3_tt_eval \
  --max_candidates=2 \
  --trial_steps=1 \
  --print_interval=1 \
  --headless
```

结果：退出码 0，输出 Top candidates 和 CSV。

实际筛选结果：

```text
96 ready-arm candidates, 250 steps:
  0 survived
  best: candidate_index=170, mild_z0.90_w+0.00_toe+0.05, 114/250 steps

128 candidates with zero/soft/light legs, 250 steps:
  0 survived
  best: candidate_index=13, zero_leg_z0.96, 122/250 steps

128 candidates with default/zero/ready arms, 250 steps:
  0 survived
  best score: candidate_index=37, zero_leg_default_arms_z0.96, 92/250 steps

candidate_index=37 single replay, 500 steps, relaxed thresholds:
  95/500 steps, reset_seen=True
```

结论：当前脚本已能自动筛选并排序候选姿态，但没有找到可直接用于训练配置的稳定 A3 自由基站姿。下一步应先查 A3 脚底 collision、root 高度、惯量/质心和 leg/feet PD，而不是继续直接长训乒乓球策略。

## 2026-06-25 A3 actuator table update

用户补充 A3 电机参数截图，明确了各电机型号、减速比、扭矩、转速和仿真中使用的等效关节惯量计算方式。

计划修改：

- 仅修改 A3 asset 配置 `legged_lab/assets/a3/a3.py`。
- 保持 T1 asset 和 T1 workflow 不变。
- 将当前统一占位 `armature=0.01` 改为按截图计算得到的关节 armature。
- 将 PFP-59-60 对应的 upper J3/J4/J5 峰值扭矩从保守值 `24 Nm` 调整为截图峰值 `36 Nm`。
- 将 PFP-41-48 对应 wrist/head 的额定速度统一为 `150 rpm = 15.70796 rad/s`。
- 将 `waist_pitch_joint` 等效最大力矩按截图说明调整为 `115 Nm`。

计算值：

```text
PFP-110-75: 0.12034028684
PFP-93-65:  0.06646569891
PFP-78-58:  0.01208336871
PFP-59-60:  0.00496735130
PFP-41-48:  0.00081008933
ankle_pitch: 0.06444060531
ankle_roll:  0.02012630058
waist_pitch: 0.08820859156
waist_roll:  0.01462087613
```

实际修改：

- `legged_lab/assets/a3/a3.py`
  - 新增 `A3_ARMATURE`。
  - 新增 `A3_RATED_SPEED`。
  - `waist`、`legs`、`feet`、`arms`、`wrists`、`head` actuator 的 `armature` 改为按关节/正则匹配的表格值。
  - `waist_pitch_joint` effort limit 从 `118.0` 调整为截图说明的 `115.0`。
  - shoulder_yaw / elbow / wrist_roll effort limit 从 `24.0` 调整为 PFP-59-60 峰值 `36.0`。
  - wrist_pitch / wrist_yaw / head velocity limit 调整为 PFP-41-48 额定速度 `15.707963267948966 rad/s`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/scripts/a3_standing_calibration.py
```

结果：通过。

最小 Isaac smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_standing_calibration \
  --task=a3_tt_eval \
  --max_candidates=2 \
  --trial_steps=1 \
  --print_interval=1 \
  --headless
```

结果：退出码 0，新的 actuator 配置可被 Isaac Lab 解析。

修正 actuator 后复测站姿筛选：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_standing_calibration \
  --task=a3_tt_eval \
  --max_candidates=128 \
  --trial_steps=250 \
  --print_interval=50 \
  --headless
```

结果：

```text
0/128 candidates survived 250 steps
best: candidate_index=52, light_crouch_default_arms_z1.02
survival_steps=116/250
reset_seen=False under script reset flag, but failed roll/pitch threshold
```

结论：按截图修正电机 armature 和部分限幅后，候选评分有所改善，但仍没有得到真正稳定的 A3 初始站姿。下一步应继续查脚底 collision、root 高度/脚底贴地关系、URDF link inertia/CoM，以及 leg/feet PD 增益。

## 2026-06-25 A3 standing physics diagnostics

用户确认继续做 A3 脚底碰撞/root 高度/PD 诊断。

计划新增：

```text
legged_lab/scripts/a3_standing_physics_diagnostics.py
```

目标：

- pinned-root 扫描 root_z，判断脚底是否接触、是否可能悬空或穿地。
- free-base 短 rollout，判断同一组 root_z 在当前 PD 下是否自由基稳定。
- 支持临时 PD multiplier，例如 leg/feet/waist stiffness/damping scale。
- 输出 CSV，方便对比不同 root_z 和 PD 组合。

边界：

- 不修改 T1。
- 不修改 A3 默认训练姿态。
- 诊断参数只通过命令行临时作用，不写入配置。

实际修改：

- 新增 `legged_lab/scripts/a3_standing_physics_diagnostics.py`。
- 脚本分为 `--phase=pinned` 和 `--phase=free` 两种运行模式：
  - `pinned`：固定 root，只扫描脚底接触、脚底高度、脚底滑移和左右脚支撑力。
  - `free`：自由基短 rollout，观察 survival steps、reset、roll/pitch、z drift、脚底滑移。
- 支持 `--pose=current/zero_leg/light_crouch/t1_like/ready_light_crouch/ready_t1_like`。
- 支持命令行临时缩放 PD：
  - `--waist_stiffness_scale`
  - `--waist_damping_scale`
  - `--leg_stiffness_scale`
  - `--leg_damping_scale`
  - `--feet_stiffness_scale`
  - `--feet_damping_scale`
- 输出 CSV 到 `logs/a3_standing_physics/`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/scripts/a3_standing_physics_diagnostics.py

git diff --check
```

结果：通过。

最小 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_standing_physics_diagnostics \
  --task=a3_tt_eval \
  --phase=pinned \
  --root_z_values=0.90,1.02 \
  --contact_steps=1 \
  --print_interval=1 \
  --headless

CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_standing_physics_diagnostics \
  --task=a3_tt_eval \
  --phase=free \
  --root_z_values=0.90,1.02 \
  --free_steps=1 \
  --print_interval=1 \
  --headless
```

结果：均可创建 Isaac 环境并输出 CSV。

实际诊断结果：

```text
pinned current pose, 50 steps:
  root_z=0.90: contact_ratio=0.780, foot_slip=0.6623
  root_z=0.96: contact_ratio=0.980, foot_slip=0.1547
  root_z=1.02: contact_ratio=1.000, foot_slip=0.0399
  root_z=1.08: contact_ratio=0.000, foot_slip=0.0027

free current pose, 250 steps:
  root_z=0.90: survival=92/250, reset_seen=True
  root_z=0.96: survival=54/250, reset_seen=True
  root_z=1.02: survival=101/250, reset_seen=True
  root_z=1.08: survival=74/250, reset_seen=True

free current pose + leg/feet damping x2, 250 steps:
  root_z=0.96: survival=57/250
  root_z=1.02: survival=114/250
  root_z=1.08: survival=114/250

free current pose + leg/feet stiffness x1.5 + damping x2, 250 steps:
  root_z=0.96: survival=88/250
  root_z=1.02: survival=94/250
  root_z=1.08: survival=94/250

free light_crouch pose + leg/feet damping x2, 250 steps:
  root_z=0.96: survival=57/250, reset_seen=True
  root_z=1.02: survival=140/250, reset_seen=False, max_abs_roll_pitch=0.5517
  root_z=1.08: survival=101/250, reset_seen=True
```

关键结论：

- 当前 A3 默认 `root_z=0.90` 不理想，脚底接触滑移很大。
- `root_z=1.02` 明显更接近合理脚底贴地状态；`root_z=1.08` 已出现双脚无接触。
- 仅增加 stiffness 不解决站立问题，反而比 damping-only 更差。
- 当前最好候选是 `light_crouch + root_z=1.02 + leg/feet damping x2`，达到 `140/250` steps 且没有触发 reset，但 roll/pitch 仍略高于严格阈值。
- 因此还不能直接把该姿态写入 A3 训练配置；下一步应通过 WebRTC 可视化确认脚底 collision、身体倾倒方向和 root 高度，再做小范围 PD/pose 微调。

继续确认计划：

- 在 500 step 复测中，`light_crouch + root_z≈1.00-1.04 + leg/feet damping x2` 仍在约 100-140 step 触发环境 reset。
- 下一步只增强诊断脚本输出，不改训练配置：
  - 记录首次 `env.reset_buf` 触发的 step。
  - 记录 TTEnv 使用的 `robot_pos` 在 table frame 下的 x/y/z 范围。
  - 用于判断 reset 是否来自 `robot_pos[..., 2] < 0.50`、x/y 边界，还是脚底/姿态问题间接导致的位置漂移。

继续确认中的新发现：

- `light_crouch + root_z=1.00/1.02/1.04 + damping x2` 在 300-500 step 复测中仍会 reset。
- `base_y=0.35` 时主要表现为 table-frame 下 `y` 漂到 `1.10+` 或 `x` 漂到 `-1.34`。
- `base_y=0.0` 后仍会失败，并且 `robot_pos.z` 会跌到约 `0.47-0.50`，说明不是单纯 reset 边界过窄，而是自由基确实失稳。
- 下一步增加 `zero_light_crouch` 和 `zero_t1_like` 两个诊断姿态，用于判断上肢/球拍非对称是否是主要扰动源。

实际追加修改：

- `legged_lab/scripts/a3_standing_physics_diagnostics.py`
  - `ScanResult` 增加 `first_env_reset_step`。
  - CSV/终端输出增加 table-frame 下 `robot_pos` 的 x/y/z 范围。
  - 新增 `zero_light_crouch` 和 `zero_t1_like` 诊断姿态。

继续确认结果：

```text
light_crouch + root_z=1.00/1.02/1.04 + damping x2:
  约 100-140 step 后 reset
  主要表现为 y 漂移到 1.10+ 或 x 漂移到 -1.34

ready_light_crouch + damping x2:
  约 110-125 step 后 reset
  roll/pitch 约 1.3 rad，明显倒下

current + damping x2:
  约 103-118 step 后 reset
  y 漂移到 1.10+

zero_light_crouch + root_z=1.04 + damping x2:
  500/500 step, no env reset
  max_abs_roll_pitch=0.5584

zero_light_crouch + root_z=1.05/1.06 + damping x2:
  1000/1000 step, no env reset
  root_z=1.05 max_abs_roll_pitch=0.8642
  root_z=1.06 max_abs_roll_pitch=0.6317

zero_light_crouch + root_z=1.045/1.050/1.055 + waist damping x2 + leg/feet damping x3:
  all 500/500 step, no env reset
  root_z=1.050 max_abs_roll_pitch=0.5189

strict 1000 step final candidate:
  zero_light_crouch
  root_z=1.050
  waist_damping_scale=2.0
  leg_damping_scale=3.0
  feet_damping_scale=3.0
  survival=1000/1000
  reset_seen=False
  max_abs_roll_pitch=0.5412
  max_root_z_drift=0.1796
  both_feet_contact_ratio=0.981
```

结论：

- 上肢/球拍非对称是 A3 初始站立不稳的重要因素；默认/ready 手臂都会明显放大侧向或前向失稳。
- 当前最可靠的站立基线候选是 `zero_light_crouch + root_z=1.05 + waist damping x2 + leg/feet damping x3`。
- 该候选只能作为站立基线，不是最终乒乓击球姿态；右臂持拍姿态应在这个稳定基线之上逐步引入。
- 目前仍未修改正式 A3 训练配置。

## 2026-06-25 A3 standing search refinement

用户追问当前候选是否能稳定站住。

判断：

- `zero_light_crouch + root_z=1.05 + waist damping x2 + leg/feet damping x3` 已经能在严格诊断中达到 `1000/1000` 且无 reset。
- 但 `max_abs_roll_pitch=0.5412 rad`，仍接近 31 度；脚底滑移约 `0.1976 m`，只能算“不会 reset 的站立基线”，不能算理想自然站姿。
- 为了后续逐步加入右臂/球拍姿态，应该继续寻找更稳的站姿，目标是把 `max_abs_roll_pitch` 压到约 `0.3-0.4 rad`，并降低脚底滑移。

计划修改：

- 增强 `legged_lab/scripts/a3_standing_calibration.py`，只用于 A3-only 诊断搜索。
- 增加临时 PD scale 参数：
  - `--waist_stiffness_scale`
  - `--waist_damping_scale`
  - `--leg_stiffness_scale`
  - `--leg_damping_scale`
  - `--feet_stiffness_scale`
  - `--feet_damping_scale`
- 增加 `--crouch_arm_modes`，允许 crouch 宽站姿使用 `zero` arms 参与搜索。
- 不修改 A3 正式训练配置，不修改 T1 workflow。

实际修改：

- `legged_lab/scripts/a3_standing_calibration.py`
  - 增加 `--crouch_arm_modes`，可指定 `default,zero,ready`。
  - 增加 waist/leg/feet stiffness/damping scale 参数。
  - crouch 搜索候选现在可使用 zero arms，不再只使用 ready arms。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/scripts/a3_standing_calibration.py \
  legged_lab/scripts/a3_standing_physics_diagnostics.py

git diff --check -- legged_lab/scripts/a3_standing_calibration.py CODE_CHANGE_LOG_2026-06-23_A3.md
```

结果：通过。

搜索命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_standing_calibration \
  --task=a3_tt_eval \
  --root_z_values=1.045,1.05,1.055 \
  --crouch_arm_modes=zero \
  --max_candidates=80 \
  --trial_steps=500 \
  --base_x=-0.40 \
  --base_y=0.35 \
  --waist_damping_scale=2.0 \
  --leg_damping_scale=3.0 \
  --feet_damping_scale=3.0 \
  --headless
```

搜索结果：

```text
best batch candidate:
  candidate_index=122
  mild_zero_arms_z1.05_w+0.10_toe+0.05
  500/500 steps
  max_abs_roll_pitch=0.5113
  max_foot_slip=0.0788

single 1000-step replay:
  candidate_index=122 failed at about 100 steps
  max_abs_roll_pitch=0.5510
  reason: just crossed strict 0.55 roll/pitch threshold
```

因此 `candidate_index=122` 不够鲁棒，不能作为最终候选。

继续复测 batch 第二名：

```text
candidate_index=205
name=mild_zero_arms_z1.05_w+0.10_toe+0.00
root_z=1.055
single replay=1000/1000
reset_seen=False
max_abs_roll_pitch=0.4789
max_root_z_drift=0.1886
max_foot_slip=0.0951
both_feet_contact_ratio=0.991
```

当前结论：

- `candidate_index=205` 比之前的 `zero_light_crouch + root_z=1.05` 更稳：
  - roll/pitch 从 `0.5412` 降到 `0.4789`
  - foot slip 从 `0.1976` 降到 `0.0951`
  - 1000 step 单独复测通过
- 该候选可以作为当前 A3 站立基线候选。
- 它仍是 zero-arm 站立基线，不是最终持拍击球姿态。
- 目前仍未写入正式 A3 训练配置。

## 2026-06-25 A3 WebRTC viewport camera

用户反馈：

- Isaac Lab 官方最小 WebRTC 场景可以正常显示。
- A3 standing calibration 脚本 WebRTC 仍然黑屏。

判断：

- WebRTC 服务本身可用。
- A3 脚本问题更可能是默认 viewport camera 没有对准 A3 场景，或者没有主动设置 viewport camera。
- 官方最小示例中有 `sim.set_camera_view(...)`，而 A3 standing calibration 脚本当前没有设置相机。

计划修改：

- 只修改 A3-only 调试脚本 `legged_lab/scripts/a3_standing_calibration.py`。
- 增加 `--camera_eye` 和 `--camera_target` 参数。
- 创建环境后调用 `env.sim.set_camera_view(...)`，默认看向 A3 当前站位。
- 不修改训练配置，不修改 T1 workflow。

继续反馈：

- 用户观察到：A3 WebRTC 一开始可以看到界面，但在输出 `step 1/...` 后画面变黑。
- 说明 WebRTC 服务和初始 viewport 都能工作，问题发生在仿真 step 循环开始后。

计划追加：

- 给 `a3_standing_calibration.py` 增加可视化慢速模式：
  - `--visualize_sleep`：每个 env step 后 sleep 一小段时间，避免 WebRTC 流被高速仿真循环压住。
  - `--keep_camera_interval`：每隔 N step 重新设置 viewport camera。
  - `--warmup_render_steps`：开始仿真前先渲染若干帧，保证 WebRTC 初始画面稳定。
- 在可视化模式下额外调用 `env.sim.render()` 和 `simulation_app.update()`。

实际修改：

- `legged_lab/scripts/a3_standing_calibration.py`
  - 新增 `--visualize_sleep`。
  - 新增 `--keep_camera_interval`。
  - 新增 `--warmup_render_steps`。
  - step 后可选额外调用 `env.sim.render()` 和 `simulation_app.update()`。
  - 可选按间隔重新执行 `set_camera_view`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/scripts/a3_standing_calibration.py

git diff --check -- legged_lab/scripts/a3_standing_calibration.py CODE_CHANGE_LOG_2026-06-23_A3.md
```

结果：通过。

## 2026-06-25 A3 measured pose hybrid search

用户提供了一组 A3 真机/控制器侧站立姿态数据，包含 `layout` 和 `q`：

- `layout` 与 A3 当前 31 DOF 顺序一致。
- 上半身和双臂姿态较自然，可作为右臂/球拍预姿态参考。
- 下肢比当前稳定候选更直，膝关节约 `0.20 rad`，髋 roll 接近 0。

先用一次性仿真检查该完整姿态：

```text
root_z=1.020: 41/500 steps, max_abs_roll_pitch=0.5732, max_foot_slip=0.0057
root_z=1.050: 41/500 steps, max_abs_roll_pitch=0.5570, max_foot_slip=0.0060
root_z=1.080: 43/500 steps, max_abs_roll_pitch=0.5759, max_foot_slip=0.0058
root_z=1.110: 43/500 steps, max_abs_roll_pitch=0.5720, max_foot_slip=0.0088
```

判断：

- 该完整姿态不是直接可用的稳定 reset 姿态。
- 失败主要是严格 roll/pitch 阈值略超，不是严重脚滑或高度塌陷。
- 更合理的路线是保留真机上半身/双臂，组合当前仿真搜索出的稳定下肢。

计划修改：

- 只修改 A3-only 标定脚本 `legged_lab/scripts/a3_standing_calibration.py`。
- 新增 `--measured_pose_json`，读取用户提供的姿态 JSON。
- 新增 measured 上肢候选和 measured 完整姿态对照候选。
- 不修改正式 A3 训练配置，不修改 T1 workflow。

第一轮 measured-upper 搜索结果：

```text
best measured candidate:
  candidate_index=16
  zero_leg_measured_arms_z1.05
  survival=78/500
  reset_seen=True
  max_abs_roll_pitch=0.5665
  max_foot_slip=0.0027

measured_full_z1.045:
  survival=41/500
  reset_seen=True
  max_abs_roll_pitch=0.5597
  max_foot_slip=0.0057
```

同时，同一轮中 zero-arm 对照仍能出现 `500/500` 的候选，说明当前问题主要来自 measured 上身姿态而不是下肢搜索或仿真命令。

追加计划：

- 继续扩展 `a3_standing_calibration.py` 的 measured 子模式：
  - `measured_torso`
  - `measured_arms`
  - `measured_right_arm`
- 用这些子模式区分失稳来源。

第二轮 measured 子模式结果：

严格阈值 `max_abs_roll_pitch=0.55`：

```text
best measured_torso:
  light_crouch_measured_torso_arms_z1.04
  survival=146/500
  reset_seen=True

best measured_arms:
  zero_leg_measured_arms_arms_z1.05
  survival=98/500
  reset_seen=True

best measured_right_arm:
  zero_leg_measured_right_arm_arms_z1.04
  survival=104/500
  reset_seen=True
```

放宽阈值 `max_abs_roll_pitch=0.70`：

```text
best measured_torso:
  candidate_index=83
  mild_measured_torso_arms_z1.04_w+0.14_toe+0.05
  survival=500/500
  reset_seen=False
  max_abs_roll_pitch=0.6035
  max_root_z_drift=0.1939
  max_foot_slip=0.0295

best measured_right_arm:
  survival=112/500
  reset_seen=True
```

判断：

- 真机腰/头姿态本身不是最大问题，但会让身体更倾斜。
- 真机双臂和真机右臂直接作为 reset 目标仍不稳定。
- 下一步需要右臂姿态插值，而不是一次性使用完整真机右臂角度。

追加计划：

- 增加 `measured_right_arm_blend_<alpha>` 模式。
- 用 `alpha=0.25/0.50/0.75` 搜索稳定右臂预姿态。

## 2026-06-25 A3 fallback stabilization config

用户确认可以开始做 A3 专项托底机制修改。

本阶段目标：

- 把已通过 1000 step 复测的 A3 zero-arm 稳定站姿写入 A3 默认姿态。
- 把站立诊断中有效的 waist/leg/feet damping 放大写入 A3 actuator。
- 缩小 A3 reset 随机范围。
- 降低 A3 初期 action scale 和 PPO 初始动作噪声。
- 初期关闭 A3 的 push_robot 和 base mass randomization，后续站稳后再打开。

计划修改文件：

```text
legged_lab/assets/a3/a3.py
legged_lab/envs/a3_tt/a3_tt_config.py
```

边界：

- 不修改 T1。
- 不修改公共 `TTEnv.step()` 或 reset/reward 语义。
- 不把真机右臂 measured pose 直接写为默认姿态；右臂持拍预姿态后续单独做 curriculum/插值。

实际修改：

- `legged_lab/assets/a3/a3.py`
  - 新增 `A3_STABLE_STANDING_ROOT_POS` 和 `A3_STABLE_STANDING_JOINT_POS`。
  - A3 默认 root 高度从 `0.90` 改为 `1.055`。
  - A3 默认姿态改为 `candidate_index=205` 对应 zero-arm 宽站姿。
  - `waist.damping` 从 `4.0` 改为 `8.0`。
  - `legs.damping` 从 `4.0` 改为 `12.0`。
  - `feet.damping` 从 `4.0` 改为 `12.0`。

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 A3 初期托底常量：
    - `A3_INITIAL_ACTION_SCALE = 0.15`
    - `A3_INITIAL_POLICY_NOISE_STD = 0.5`
    - A3 train/eval base reset pose/velocity ranges
  - A3 `robot.action_scale` 从 `0.20` 改为 `0.15`。
  - A3 `reset_locomotion_joints.position_range` 从 `(0.5, 1.5)` 收紧为 `(0.95, 1.05)`。
  - A3 `reset_manipulation_joints.position_range` 从 `(-0.5, 0.5)` 收紧为 `(-0.05, 0.05)`。
  - A3 train/eval base reset pose/velocity 收紧。
  - A3 `push_robot` 初期关闭。
  - A3 base mass randomization 初期关闭为 `(0.0, 0.0)`。
  - A3 PPO `init_noise_std` 从继承的 `1.0` 改为 `0.5`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/scripts/a3_standing_calibration.py

git diff --check -- \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  A3_MIGRATION_PLAN_2026-06-23.md \
  CODE_CHANGE_LOG_2026-06-23_A3.md
```

结果：通过。

A3 当前默认姿态自由基 500 step 诊断：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_standing_physics_diagnostics \
  --task=a3_tt_eval \
  --pose=current \
  --root_z_values=1.055 \
  --phase=free \
  --free_steps=500 \
  --base_x=-0.40 \
  --base_y=0.35 \
  --headless
```

结果 CSV：

```text
logs/a3_standing_physics/a3_fallback_current_free_500_results.csv
```

关键结果：

```text
survival_steps=500/500
reset_seen=0
max_abs_roll_pitch=0.478885
max_root_z_drift=0.188608
max_foot_slip=0.095079
both_feet_contact_ratio=0.982000
```

A3 训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_fallback_smoke \
  --headless \
  --predictor
```

结果：正常退出，日志目录：

```text
logs/a3_table_tennis/2026-06-25_23-33-53_a3_fallback_smoke
```

T1 轻量语法回归：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/t1_tt/t1_tt_config.py \
  legged_lab/assets/booster/booster.py \
  legged_lab/envs/base/tt_env.py
```

结果：通过。

## 2026-06-25 A3 WebRTC current default visualization fix

用户使用上一条 WebRTC 命令观察后反馈 A3 仍会往后倒。

复盘：

- 上一条命令使用了 `a3_standing_calibration.py --candidate_index=37`。
- 该脚本中的 candidate index 不是全局稳定编号，会随着 `root_z_values`、`crouch_arm_modes`、`measured_pose_json` 等参数变化。
- 因此 `candidate_index=37` 不等价于已写入 A3 asset 的稳定默认姿态。
- 当前正式 A3 默认姿态已经在 headless 诊断中通过：

```text
survival_steps=500/500
reset_seen=0
max_abs_roll_pitch=0.478885
max_root_z_drift=0.188608
max_foot_slip=0.095079
```

计划修改：

- 只修改 A3-only 可视化/标定脚本 `legged_lab/scripts/a3_standing_calibration.py`。
- 新增 `--use_current_default_pose`，直接使用当前环境加载出的 A3 默认关节姿态。
- 不修改正式 A3 asset/task 配置。
- 不修改 T1。

## 2026-06-25 A3 backward fall follow-up

用户继续反馈 A3 可视化仍会往后倒。

本轮处理记录：

- 发现机器上仍有旧 WebRTC 标定进程在运行：
  - `a3_standing_calibration.py --candidate_index=37`
  - 该命令未使用 `--use_current_default_pose`，可能继续显示旧候选姿态。
- 已停止旧进程，避免后续判断混入 stale visualization。

下一步计划：

- headless 批量搜索更前倾、更强托底的 A3 默认姿态/PD 组合。
- 只在搜索通过后修改：
  - `legged_lab/assets/a3/a3.py`
  - 必要时 `legged_lab/envs/a3_tt/a3_tt_config.py`
- 不修改 T1 代码和公共 TT workflow。

第二轮搜索结果：

- 扩展 `legged_lab/scripts/a3_standing_calibration.py`：
  - 支持扫 `waist_pitch_values`、`hip_pitch_values`、`knee_values`、`ankle_pitch_values`、`stance_width_values`、`toe_out_values`；
  - CSV 增加 `final_roll/final_pitch/min_pitch/max_pitch`；
  - 这是 A3-only 标定工具改动，不影响训练流程。
- PD 搜索结论：
  - `leg/feet/waist` stiffness 或 damping 继续增大，会减少下沉但增大姿态冲击；
  - 本轮不把更强 PD 写入 asset。
- 准备写入 A3 默认姿态候选：
  - `hip_pitch=-0.14`
  - `knee=0.38`
  - `ankle_pitch=-0.16`
  - `waist_pitch=0.0`
  - 保持当前 PD。

落地修改：

- `legged_lab/assets/a3/a3.py`
  - A3 默认腿部姿态从较深蹲改为更浅的稳定姿态：
    - `hip_pitch=-0.14`
    - `knee=0.38`
    - `ankle_pitch=-0.16`
    - `waist_pitch=0.0`
  - PD 参数保持当前值，没有继续加硬。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/scripts/a3_standing_calibration.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/t1_tt/t1_tt_config.py \
  legged_lab/assets/booster/booster.py \
  legged_lab/envs/base/tt_env.py
```

结果：通过。

```bash
git diff --check -- \
  legged_lab/assets/a3/a3.py \
  legged_lab/scripts/a3_standing_calibration.py \
  A3_MIGRATION_PLAN_2026-06-23.md \
  CODE_CHANGE_LOG_2026-06-23_A3.md
```

结果：通过。

新默认姿态 1000 step free-base 诊断：

```text
logs/a3_standing_physics/a3_new_default_pose_free_1000_results.csv

survival_steps=1000/1000
reset_seen=0
max_abs_roll_pitch=0.468132
max_root_z_drift=0.192568
max_foot_slip=0.104378
both_feet_contact_ratio=1.000000
final_root_z=0.878292
```

A3 训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_pose2_smoke \
  --headless \
  --predictor
```

结果：正常退出。

## 2026-06-26 TT WebRTC probe

用户运行 T1 `train.py` 和 `preview.py` 的 WebRTC 命令后仍然黑屏。

原因判断：

- 现有 `train.py` / `preview.py` 不适合作为远程 WebRTC 观察工具；
- 它们缺少 A3 标定脚本中已经验证有效的固定相机、render/update、sleep 机制。

计划修改：

- 新增 `legged_lab/scripts/tt_webrtc_probe.py`；
- 只用于可视化观察；
- 不修改 T1/A3 正式训练流程。

## 2026-06-26 A3 right-arm paddle default pose

用户确认当前阶段先为 A3 设置一个好的初始姿态，要求默认姿态带上球拍/右臂预姿态。

复测记录：

```text
logs/a3_standing_calibration/a3_ready_paddle_pose_check_1000.csv

current legs + zero arms:
  survival_steps=1000/1000
  reset_seen=0

current legs + ready arms:
  survival_steps=91/1000
  reset_seen=1
```

结论：不能直接把手写 `READY_ARM_JOINTS` 写入默认姿态。

继续使用用户提供的 measured pose 做右臂插值：

```text
logs/a3_standing_calibration/a3_blend_050_relaxed_2000.csv

measured_right_arm_blend_0.50:
  survival_steps=2000/2000
  reset_seen=0
  final_root_z=0.846580
  max_abs_roll_pitch=0.610165
  max_root_z_drift=0.226591
  max_foot_slip=0.025867
  both_feet_contact_ratio=0.999500
  min_paddle_future_dist=0.184931

logs/a3_standing_calibration/a3_blend_075_relaxed_2000.csv

measured_right_arm_blend_0.75:
  survival_steps=381/2000
  reset_seen=1
```

落地计划：

- 修改 `legged_lab/assets/a3/a3.py`：
  - 右臂默认姿态从 zero arms 改为 measured right-arm 50% blend。
  - 左臂继续保持 zero arms。
- 修改 `legged_lab/envs/a3_tt/a3_tt_config.py`：
  - A3 右臂 reset offset 从 `(-0.05, 0.05)` 收紧到 `(-0.02, 0.02)`。
- 不修改 T1；
- 不修改公共 TTEnv。

实际修改：

- `legged_lab/assets/a3/a3.py`
  - `right_shoulder_pitch_joint=0.1449383158874511`
  - `right_shoulder_roll_joint=-0.053864232177734285`
  - `right_shoulder_yaw_joint=0.004107922210693449`
  - `right_elbow_joint=0.4118487550354004`
  - `right_wrist_roll_joint=-0.0012087522888184487`
  - `right_wrist_pitch_joint=0.006742773132324187`
  - `right_wrist_yaw_joint=-0.0007340335083008132`
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - `reset_manipulation_joints.position_range=(-0.02, 0.02)`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/scripts/a3_standing_calibration.py \
  legged_lab/scripts/tt_webrtc_probe.py
```

结果：通过。

```text
logs/a3_standing_calibration/a3_current_default_paddle_ready_050_2000.csv

current_default_pose:
  survival_steps=2000/2000
  reset_seen=0
  final_root_z=0.842900
  max_abs_roll_pitch=0.601967
  max_root_z_drift=0.222827
  max_foot_slip=0.025901
  both_feet_contact_ratio=0.999500
  min_paddle_future_dist=0.184931
```

A3 训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_paddle_ready_050_smoke \
  --headless \
  --predictor
```

结果：正常退出。

## 2026-06-26 A3 paddle face orientation toward ball path

用户指出：拍子需要横过来，一个拍面正对球路，否则即使右臂持拍，早期也可能因为拍面角度不对而接不到球，导致奖励稀疏。

几何判断：

```text
pingpang_red_Link.STL / pingpang_black_Link.STL extents:
  x ~= 0.1604 m
  y ~= 0.0029 m
  z ~= 0.1604 m
```

因此 A3 球拍拍面平面为局部 `X-Z`，拍面法向为局部 `Y`。

计划修改：

- 扩展 `legged_lab/scripts/a3_pose_calibration.py`：
  - 打印 paddle body 的世界系局部轴；
  - 打印 `+Y/-Y` 哪一面更朝向来球；
  - 打印与 `-ball_linvel` 的对齐度。
- 用该输出搜索新的右臂/手腕默认姿态。
- 只修改 A3 默认姿态，不修改 T1 或公共 `TTEnv`。

实际修改一：

- `legged_lab/scripts/a3_pose_calibration.py`
  - 新增 `paddle_local_x_w`、`paddle_face_+y_w`、`paddle_local_z_w` 打印；
  - 新增 `incoming_ball_dir` 和 `face_alignment(+Y/-Y/best)` 打印；
  - 新增 `reset_count`，用于长步调试时判断候选姿态是否触发 reset。

当前默认姿态诊断：

```text
step=1
ball_linvel=[-5.7804, -0.1976, 1.4846]
incoming_ball_dir=[0.9680, 0.0331, -0.2486]
paddle_face_+y_w=[0.0022, 0.9985, -0.0549]
face_alignment=0.0488
paddle_to_future_dist=0.4262 m
```

`right_wrist_yaw_joint=-1.45`：

```text
step=1
face_alignment=0.9562
paddle_to_future_dist=0.1812 m

2000 step:
reset_count=15
```

结论：几何很好，但会触发 reset，不落地。

`right_wrist_yaw_joint=-1.20`：

```text
step=1
face_alignment=0.9094
paddle_to_future_dist=0.2317 m

2000 step:
reset_count=0
base_pos_w=[-1.4318, 0.2155, 0.8437]
paddle_to_future_dist=0.7315 m
```

当前默认对照：

```text
2000 step:
reset_count=0
base_pos_w=[-1.4325, 0.2562, 0.8389]
face_alignment=0.0106
paddle_to_future_dist=0.9426 m
```

结论：

- `right_wrist_yaw_joint=-1.20` 不增加 reset；
- 初始拍面对齐从 `0.0488` 提升到 `0.9094`；
- 初始击球点距离从 `0.4262 m` 降到 `0.2317 m`；
- 2000 step 静态收敛后仍优于当前默认。

落地计划：

- 修改 `legged_lab/assets/a3/a3.py`：
  - `right_wrist_yaw_joint=-1.2`
  - 其他右臂关节暂不变。

实际修改二：

- `legged_lab/assets/a3/a3.py`
  - `right_wrist_yaw_joint: -0.0007340335083008132 -> -1.2`
  - 注释更新为 measured right-arm blend + wrist yaw turned to meet the ball path。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/scripts/a3_pose_calibration.py \
  legged_lab/scripts/a3_standing_calibration.py \
  legged_lab/scripts/tt_webrtc_probe.py
```

结果：通过。

当前默认姿态一帧几何：

```text
face_alignment=0.9094
paddle_to_future_dist=0.2317 m
right_wrist_yaw_joint=-1.200036
```

当前默认姿态 2000 step：

```text
reset_count=0
base_pos_w=[-1.4318, 0.2155, 0.8437]
paddle_to_future_dist=0.7315 m
```

A3 训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_paddle_face_rwy_120_smoke \
  --headless \
  --predictor
```

结果：正常退出。

## 2026-06-26 T1 reference paddle-face diagnostics

用户指出 A3 球拍仍不像 T1 一样完整横过来正对来球方向。

判断：

- 仅用 A3 的 `paddle_face_+y_w` 和来球方向对齐度不足以说明视觉上等同于 T1；
- 需要先量化 T1 默认姿态的 paddle body 局部轴、击球点和球路关系；
- 再让 A3 向 T1 的几何参照靠拢。

计划修改：

- 新增 `legged_lab/scripts/tt_paddle_pose_diagnostics.py`；
- 通用支持 T1/A3 的 TT task；
- 输出各 paddle local axis 与 `incoming_ball_dir` 的对齐度；
- 不修改 T1；
- 不修改公共 TTEnv。

新增脚本：

- `legged_lab/scripts/tt_paddle_pose_diagnostics.py`

T1 参照输出：

```text
task=t1_tt_eval
paddle_body=right_hand_link
incoming_ball_dir=[0.9680, 0.0331, -0.2486]
axis_+X_w=[0.8648, 0.4552, -0.2118]
align_+X=0.9049
best_axis=+X
```

A3 当前输出：

```text
task=a3_tt_eval
paddle_body=right_hand_pingpang_Link
axis_+Y_w=[0.8000, 0.3344, -0.4982]
align_+Y=0.9094
best_axis=+Y
```

判断：

- A3 当前拍面对来球方向的点积已经高，但 `z=-0.4982`，下倾明显；
- T1 参照轴的 `z=-0.2118`，更接近“整面横过来”；
- 下一步目标不是继续提高 A3 `align_+Y`，而是让 A3 `axis_+Y_w` 更接近 T1 `axis_+X_w`。

细化搜索：

```text
T1 reference:
  axis_+X_w=[0.8648, 0.4552, -0.2118]

A3 candidate:
  right_wrist_roll_joint=-0.25
  right_wrist_pitch_joint=-0.40
  right_wrist_yaw_joint=-1.20
  paddle_face_+y_w=[0.8736, 0.4268, -0.2338]
  face_alignment=0.9179
  paddle_to_future_dist=0.2146 m

2000 step:
  reset_count=0
```

落地计划：

- 修改 `legged_lab/assets/a3/a3.py`
  - `right_wrist_roll_joint=-0.25`
  - `right_wrist_pitch_joint=-0.40`
  - `right_wrist_yaw_joint=-1.20`
- 不修改 T1；
- 不修改公共 TTEnv。

## 2026-06-26 A3 visual stance correction

用户反馈当前 WebRTC 姿态错误：

- 腿并拢；
- 屈膝不足；
- 上身没有轻微前倾。

原因复盘：

- 上一轮写入的 `hip_roll` 符号来自 headless 候选排序，但视觉上对应了收腿方向；
- 本轮需要优先满足 A3 乒乓球准备姿态的视觉要求，再用 headless 诊断排除明显摔倒。

计划修改：

- 只修改 A3 默认姿态；
- 不修改 T1；
- 不修改公共 TT workflow；
- PD 暂不继续改硬，除非新姿态稳定性诊断明确需要。

搜索结果：

```text
logs/a3_standing_calibration/a3_visual_open_crouch_waist_search_1000.csv

selected candidate=176
name=custom_hp-0.22_kn+0.50_ap-0.22_zero_arms_z1.05_waist-0.02_w+0.16_toe+0.00
survival_steps=1000/1000
reset_seen=0
max_abs_roll_pitch=0.440652
max_root_z_drift=0.215386
max_foot_slip=0.106045
final_pitch=0.230375
final_root_z=0.839614
```

准备落地：

- `waist_pitch_joint: 0.0 -> -0.02`
- `hip_pitch: -0.14 -> -0.22`
- `knee: 0.38 -> 0.50`
- `ankle_pitch: -0.16 -> -0.22`
- `left_hip_roll/right_hip_roll` 改为视觉展开方向：`+0.16/-0.16`
- `left_ankle_roll/right_ankle_roll` 成对改为 `-0.072/+0.072`

落地修改：

- `legged_lab/assets/a3/a3.py`
  - `waist_pitch_joint=-0.02`
  - `left_hip_pitch_joint=-0.22`
  - `left_hip_roll_joint=0.16`
  - `left_knee_joint=0.50`
  - `left_ankle_pitch_joint=-0.22`
  - `left_ankle_roll_joint=-0.072`
  - `right_hip_pitch_joint=-0.22`
  - `right_hip_roll_joint=-0.16`
  - `right_knee_joint=0.50`
  - `right_ankle_pitch_joint=-0.22`
  - `right_ankle_roll_joint=0.072`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/scripts/a3_standing_calibration.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/t1_tt/t1_tt_config.py \
  legged_lab/assets/booster/booster.py \
  legged_lab/envs/base/tt_env.py
```

结果：通过。

```text
logs/a3_standing_physics/a3_visual_open_crouch_default_free_1000_results.csv

survival_steps=1000/1000
reset_seen=0
max_abs_roll_pitch=0.485167
max_root_z_drift=0.213199
max_foot_slip=0.072228
both_feet_contact_ratio=0.999000
final_root_z=0.851783
```

A3 训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_visual_open_crouch_smoke \
  --headless \
  --predictor
```

结果：正常退出。

## 2026-06-26 A3 full paddle-frame alignment to T1

用户在 WebRTC 中反馈：A3 球拍仍没有像 T1 那样完全横过来，T1 的拍面是完整一个面正对来球方向。

复盘：

- 上一版 A3 主要让 `paddle_face_+y_w` 对齐来球方向；
- 但视觉上还要对齐拍面内坐标轴，否则会出现法向对了、拍子整体仍然不像 T1 横开的情况；
- 因此本轮用 T1 的完整 paddle frame 作为参照，而不是只看单个点积。

T1 参照：

```text
task=t1_tt_eval
paddle_body=right_hand_link
incoming_ball_dir=[0.9680, 0.0331, -0.2486]
axis_+X_w=[0.8648, 0.4552, -0.2118]
axis_-Y_w=[0.4651, -0.8853, -0.0039]
axis_+Z_w=[0.1892, 0.0952, 0.9773]
best_axis=+X
best_alignment=0.9049
```

新 A3 候选：

```text
right_wrist_roll_joint=-0.05
right_wrist_pitch_joint=-0.33
right_wrist_yaw_joint=-1.10

paddle_local_x_w=[0.4641, -0.8857, -0.0074]
paddle_face_+y_w=[0.8624, 0.4538, -0.2245]
paddle_local_z_w=[0.2022, 0.0978, 0.9745]
face_alignment=0.9056
paddle_to_future_dist=0.2273 m
```

判断：

- A3 `+Y` 基本匹配 T1 `+X`，拍面正对来球；
- A3 `+X` 基本匹配 T1 `-Y`，拍面横向展开方式一致；
- A3 `+Z` 基本匹配 T1 `+Z`，拍面竖直方向一致；
- 该姿态比上一版 `right_wrist_roll=-0.25, right_wrist_pitch=-0.40, right_wrist_yaw=-1.20` 更符合用户的视觉要求。

落地修改：

- `legged_lab/assets/a3/a3.py`
  - `right_wrist_roll_joint: -0.25 -> -0.05`
  - `right_wrist_pitch_joint: -0.40 -> -0.33`
  - `right_wrist_yaw_joint: -1.20 -> -1.10`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/scripts/tt_paddle_pose_diagnostics.py \
  legged_lab/scripts/a3_pose_calibration.py
```

结果：通过。

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.tt_paddle_pose_diagnostics \
  --task=a3_tt_eval \
  --base_x=-0.40 \
  --base_y=0.35 \
  --max_steps=1 \
  --print_interval=1 \
  --headless
```

结果：

```text
axis_+X_w=[0.4641, -0.8857, -0.0074]
axis_+Y_w=[0.8624, 0.4538, -0.2245]
axis_+Z_w=[0.2022, 0.0978, 0.9745]
best_axis=+Y
best_alignment=0.9056
paddle_to_future_dist=0.2273 m
```

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_pose_calibration \
  --task=a3_tt_eval \
  --base_x=-0.40 \
  --base_y=0.35 \
  --print_interval=1000 \
  --max_steps=2000 \
  --no_visual_markers \
  --headless
```

结果：

```text
step=1: reset_count=0, face_alignment=0.9056
step=1000: reset_count=0
step=2000: reset_count=0
```

A3 训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_t1frame_paddle_smoke \
  --headless \
  --predictor
```

结果：正常退出。

## 2026-06-26 A3 stable-base pose and reset-range correction

用户要求继续上移/对齐 A3 击球中心，并允许通过机器人初始平移和右臂姿态调整完成。复测后发现，单纯把 base 往后移可以改善 step 1 击球几何，但会快速触发 reset 或让默认姿态下塌；因此本轮优先落地“站稳边界内的 A3 默认姿态”，并记录仍未解决的几何差距。

分析记录：

- 已更新 `A3_MIGRATION_PLAN_2026-06-23.md`：
  - 记录最新训练为什么 `reward_future_dis_ee` 有信号但 `reward_contact/table_success` 极低；
  - 解释 PD 在本框架中是关节位置伺服，不是外部托举；
  - 记录 reset 随机范围和奖励稀疏之间的矛盾；
  - 记录本轮候选搜索结果和最终边界。

姿态搜索关键结果：

```text
aggressive geometry candidate:
  base_x=-0.62
  rsp=0.00
  re=0.30
  rwp=-0.35
  rwy=-1.20
  step=1 paddle_to_future_dist=0.0228 m
  2000 step reset_count=18
  decision: reject

new A3 stable-base candidate:
  waist_pitch=-0.04
  hip_pitch=-0.05
  knee=0.50
  ankle_pitch=-0.22
  hip_roll=+0.16/-0.16
  ankle_roll=-0.072/+0.072
  base_x=-0.26
```

2000 step standing check:

```text
logs/a3_standing_calibration/a3_candidate547_basex026_2000.csv

survival_steps=2000/2000
reset_seen=0
final_root_z=0.843224
max_abs_roll_pitch=0.344542
max_root_z_drift=0.237392
max_foot_slip=0.044406
both_feet_contact_ratio=1.000000
```

真实球路复测：

```text
new default, base_x=-0.26, base_y=0.35
step=1:
  paddle_to_future_dist=0.3514 m
  face_alignment=0.9055
  paddle_touch_point.z=0.9841
step=1000:
  reset_count=0
  paddle_touch_point.z=0.6265

base_x=-0.28:
  step=1 paddle_to_future_dist=0.3327 m
  step=1000 reset_count=15

base_x=-0.30:
  step=1 paddle_to_future_dist=0.3142 m
  step=1000 reset_count=17

base_x=-0.34:
  step=1 paddle_to_future_dist=0.2778 m
  step=1000 reset_count=18
```

右臂前伸小测试：

```text
base_x=-0.26, rsp=0.30:
  step=1 paddle_to_future_dist=0.3112 m
  paddle_touch_point.z=0.9398
```

结论：单独增加右肩 pitch 收益有限且降低击球高度，本轮不落地。

实际修改：

- `legged_lab/assets/a3/a3.py`
  - `waist_pitch_joint: -0.02 -> -0.04`
  - `left_hip_pitch_joint/right_hip_pitch_joint: -0.22 -> -0.05`
  - `knee/ankle_pitch/hip_roll/ankle_roll` 保持候选 547 的稳定宽站姿组合。
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - `A3_TRAIN_BASE_POSE_RANGE` 收窄到 `x=(-0.265, -0.255), y=(0.34, 0.36), yaw=(-0.02, 0.02)`
  - `A3_EVAL_BASE_POSE_RANGE` 保持在 `x=(-0.26, -0.25), y=(0.34, 0.36), yaw=(-0.02, 0.02)`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/scripts/a3_pose_calibration.py \
  legged_lab/scripts/a3_standing_calibration.py
```

结果：通过。

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_stable_base_pose_smoke \
  --headless \
  --predictor
```

结果：正常退出。

保留问题：

- 稳定 base 下真实球路 step 1 `paddle_to_future_dist` 仍有约 `0.35 m`；
- 继续后移 base 会导致 reset；
- 仅右肩 pitch 前伸不能充分解决 x 方向误差；
- 下一步如果训练仍没有 `reward_contact`，需要做 A3 专用 staged reward/curriculum 或更系统的右臂 IK/前伸姿态搜索。

## 2026-06-26 A3 早期击球学习托底优化

背景：

- 用户完成 4096 env / 10000 iter 的 A3 训练后，TensorBoard 仍显示没有进入有效乒乓球学习；
- 现象符合：A3 稳定默认姿态与 T1 原始球路不完全对齐，早期 `reward_contact/table_success` 过稀疏；
- 当前稳定 A3 姿态不能继续通过后移 base 大幅贴近 `ball_future_pose`，否则会频繁 reset。

方案先写入：

- `A3_MIGRATION_PLAN_2026-06-23.md`
  - 新增 `2026-06-26 A3 早期学习托底优化计划`
  - 记录了真实球路诊断、原因分析、修改边界、拟落地项和训练观察指标。

实际修改：

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - `A3_INITIAL_POLICY_NOISE_STD: 0.5 -> 0.35`
  - `A3_TRAIN_BASE_VELOCITY_RANGE["yaw"]: (-0.05, 0.05) -> (-0.02, 0.02)`
  - 新增 A3 train 专用 easy serve：
    - `A3_TRAIN_BALL_SPEED_X_RANGE = (-5.2, -4.8)`
    - `A3_TRAIN_BALL_SPEED_Y_RANGE = (-0.10, 0.02)`
    - `A3_TRAIN_BALL_SPEED_Z_RANGE = (1.40, 1.60)`
    - `A3_TRAIN_BALL_POS_Y_RANGE = (-0.03, 0.03)`
  - A3 train contact reward 区域：
    - `A3_TRAIN_CONTACT_THRESHOLD = 0.07`
    - eval 保持 `A3_EVAL_CONTACT_THRESHOLD = 0.05`
  - A3 train 每个 robot episode 的 serve 数：
    - `A3_TRAIN_MAX_SERVE_PER_EPISODE = 3`
  - `reward_contact` 权重：
    - `150.0 -> 180.0`
  - 新增 A3-only dense shaping：
    - `reward_paddle_ball_dense = mdp.reward_paddle_distance_terminal`
    - `weight=4.0`
    - `coeff=15.0`
  - `reward_future_dis_ee` 权重：
    - `2.0 -> 6.0`
  - `A3TableTennisEnvCfg.__post_init__` 中覆盖 train 球路/contact/max serve；
  - `A3TT_EvalEnvCfg.__post_init__` 中显式恢复 eval 球路/contact/max serve。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/mdp/rewards.py
```

结果：通过。

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --max_iterations=1 \
  --run_name=a3_early_stage_reward_smoke \
  --headless \
  --predictor
```

结果：正常退出。

下一步训练建议：

- 先跑 250-500 iter 观察是否出现可学习迹象；
- 重点看：
  - `Episode_Reward/reward_paddle_ball_dense`
  - `Episode_Reward/reward_future_dis_ee`
  - `Episode_Reward/reward_contact`
  - reset/termination 相关曲线
  - `Episode_Reward/reward_table_success`
- 如果 dense/future reward 上升但 contact 仍低，继续微调右臂姿态或进一步收窄球路；
- 如果接触开始出现，再逐步恢复更宽球路和 `contact_threshold=0.05`。

## 2026-06-26 TensorBoard 实时监督与 dense reward 修正

监督对象：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_early_stage_dense_500 \
  --headless \
  --predictor
```

TensorBoard：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/tensorboard \
  --logdir logs/a3_table_tennis \
  --host 0.0.0.0 \
  --port 16006
```

实时读取结果：

```text
run: logs/a3_table_tennis/2026-06-26_18-26-19_a3_early_stage_dense_500
iteration ~= 85

Episode_Reward/reward_paddle_ball_dense:
  last=0
  max=6.56363e-07

Episode_Reward/reward_future_dis_ee:
  last ~= 0.1767

Episode_Reward/reward_contact:
  last=0
  max=0

Episode_Reward/reward_table_success:
  last=0
  max=0

Train/mean_reward:
  last ~= -20.02

Train/mean_episode_length:
  last ~= 51.81
```

结论：

- 训练无 NaN；
- 但 `reward_paddle_ball_dense` 基本为 0，没有实现预期的早期 dense shaping；
- 继续训练该 run 意义不大。

已停止：

- 训练 PID `1765130`
- 先后发送 `SIGTERM`、`SIGINT`，最终 `SIGINT` 后退出；
- TensorBoard PID `1765987` 保留。

原因分析：

- `reward_paddle_distance_terminal` 使用真实球当前位置到 `paddle_touch_point` 的距离；
- 发球早期球离球拍约数米，reward 极小；
- 球接近击球区域时又容易被 terminal/window mask 过滤；
- 因此该 reward 不适合作为 A3 early-stage 击球几何引导。

修正方案已落地：

- `legged_lab/mdp/rewards.py`
  - 新增 `reward_future_touch_point_target`
  - 使用 `env.ball_future_pose` 和 `env.paddle_touch_point` 的距离；
  - 使用 `env.mask_invalid` 过滤无效球路；
  - 返回连续指数型 reward。
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 将原 `reward_paddle_ball_dense` 替换为：

```python
reward_future_touch_point = RewTerm(
    func=mdp.reward_future_touch_point_target,
    weight=4.0,
    params={"std_ee": 0.5, "threshold": 0.03},
)
```

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py
```

结果：通过。

几何诊断：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_pose_calibration \
  --task=a3_tt \
  --base_x=-0.26 \
  --base_y=0.35 \
  --ball_vx=-5.0 \
  --ball_vy=-0.05 \
  --ball_vz=1.45 \
  --max_steps=1 \
  --print_interval=1 \
  --no_visual_markers \
  --headless
```

结果：

```text
paddle_to_future_dist=0.0955 m
face_alignment=0.8954
reset_count=0
```

按新 reward 公式估算：

```text
unweighted ~= exp(-0.0955 / 0.5^2) ~= 0.682
weighted ~= 4.0 * 0.682 ~= 2.73
```

这说明新的 `reward_future_touch_point` 在当前 A3 easy serve 下应提供有效非零早期信号。

## 2026-06-26 二次实时监督与坐标系修复

监督对象：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_future_touch_point_500 \
  --headless \
  --predictor
```

约第 99 iter 读取到：

```text
Episode_Reward/reward_future_touch_point:
  last=1.18562e-10
  max=2.28688e-09

Episode_Reward/reward_future_dis_ee:
  last ~= 0.175

Episode_Reward/reward_contact:
  last=0
  max=0

Episode_Reward/reward_table_success:
  last=0
  max=0
```

结论：

- 训练无 NaN；
- `reward_future_touch_point` 仍几乎为 0，说明第一版 future touch-point reward 还有实现问题；
- 已停止 PID `1770702`。

原因：

- `env.ball_future_pose` 是 env-local frame；
- `env.paddle_touch_point` 是 world frame；
- 第一版 reward 直接相减，4096 并行环境中会混入 env origin 偏移。

实际修改：

- `legged_lab/mdp/rewards.py`
  - `reward_future_touch_point_target` 中新增：

```python
paddle_touch_point = env.paddle_touch_point - env.scene.env_origins
distance = torch.linalg.norm(env.ball_future_pose - paddle_touch_point, dim=1)
```

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py
```

结果：通过。

轻量训练探针：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=5 \
  --run_name=a3_future_touch_point_localfix_probe \
  --headless \
  --predictor
```

结果：

```text
Episode_Reward/reward_future_touch_point:
  count=5
  last=0.131393
  max=0.131393
  min=0.0373882
  mean=0.0998148

Episode_Reward/reward_future_dis_ee:
  last=0.191205

Episode_Reward/reward_contact:
  last=0

Episode_Reward/reward_table_success:
  last=0
```

结论：

- 坐标修正后 `reward_future_touch_point` 已经进入 TensorBoard 且明确非零；
- 下一轮 4096 env 训练可以重新开始；
- 早期判断标准应改为：
  - `reward_future_touch_point` 是否稳定非零并逐步上升；
  - `reward_future_dis_ee` 是否同步维持非零；
  - `reward_contact` 可以允许前 100-200 iter 仍为 0，但如果长期为 0，需要继续调球路/姿态。

## 2026-06-26 第三次实时监督与低动作早期策略修正

监督对象：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_future_touch_point_localfix_500 \
  --headless \
  --predictor
```

监督到第 40-43 iter：

```text
Episode_Reward/reward_future_touch_point ~= 0.11
Episode_Reward/reward_future_dis_ee ~= 0.16
Episode_Reward/reward_contact = 0
Episode_Reward/reward_table_success = 0
Episode_Reward/termination_penalty ~= -2.0
Train/mean_episode_length ~= 52
```

结论：

- 坐标系修正有效，touch-point reward 已经稳定非零；
- 但 4096 env 早期探索仍会让 A3 很快 reset；
- episode 长度卡在约 52 step，继续训练会偏向短 episode 下避罚，不利于学击球；
- 已停止该 run。

实际修改：

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - `A3_INITIAL_ACTION_SCALE: 0.15 -> 0.08`
  - `A3_INITIAL_POLICY_NOISE_STD: 0.35 -> 0.20`
  - `A3_TRAIN_BASE_VELOCITY_RANGE`
    - linear/roll/pitch 从 `+-0.02` 收窄到 `+-0.005`
    - yaw 从 `+-0.02` 收窄到 `+-0.01`
  - 新增：
    - `A3_TRAIN_LOCOMOTION_JOINT_RESET_SCALE_RANGE = (0.98, 1.02)`
    - `A3_TRAIN_MANIPULATION_JOINT_RESET_OFFSET_RANGE = (-0.01, 0.01)`
  - `reward_future_touch_point.weight: 4.0 -> 8.0`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/mdp/rewards.py
```

结果：通过。

轻量训练探针：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=10 \
  --run_name=a3_low_action_probe \
  --headless \
  --predictor
```

TensorBoard/event 汇总：

```text
Train/mean_episode_length:
  last=133.974
  max=136.5
  mean=82.9917

Episode_Reward/reward_future_touch_point:
  last=0.245952
  max=0.325355
  mean=0.245145

Episode_Reward/reward_future_dis_ee:
  last=0.176904
  mean=0.178128

Episode_Reward/termination_penalty:
  last=-1.25
  mean=-0.158333

Episode_Reward/undesired_contacts:
  last=-26.42
  mean=-26.0434

Episode_Reward/reward_contact:
  0

Episode_Reward/reward_table_success:
  0
```

结论：

- 低动作版本相对 4096 localfix run 的 `mean_episode_length ~= 52` 有明显改善；
- `reward_future_touch_point` 更强且稳定非零；
- 早期仍未出现真实 contact；
- `undesired_contacts` 在 episode 变长后变大，说明 A3 仍会发生非足部碰撞或姿态接触，需要在下一轮正式 4096 训练中重点监督。

### 2026-06-26 A3 pingpong actuator delay 假设验证

边界：

- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改 reward 成功标准；
- 只临时验证 `A3_T2D5_PINGPANG_CFG` 的 actuator delay。

假设：

- T1 乒乓球 asset 使用 `ImplicitActuatorCfg`，没有显式 actuator delay；
- A3 使用 `DelayedPDActuatorCfg(min_delay=0,max_delay=3)`；
- 如果 A3 来球窗口前球拍下沉主要由 actuator delay 引起，那么把 A3 pingpong delay 固定为 0 后，零动作自由 root 诊断应明显改善。

临时改动：

- 在 `A3_T2D5_PINGPANG_CFG` 上覆盖 actuators，使 `min_delay=0,max_delay=0`。
- 裸 `A3_T2D5_CFG` 未改。

验证命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_pose_calibration \
  --task=a3_tt \
  --base_x=-0.26 \
  --base_y=0.35 \
  --ball_vx=-5.0 \
  --ball_vy=-0.05 \
  --ball_vz=1.45 \
  --max_steps=60 \
  --print_interval=10 \
  --no_visual_markers \
  --headless
```

关键结果：

```text
step 40:
  paddle_to_ball_dist ~= 0.3249 m
  paddle_touch_point.z ~= 0.6510
```

与前一轮未改 delay 的结果几乎一致：

```text
step 40:
  paddle_to_ball_dist ~= 0.3257 m
  paddle_touch_point.z ~= 0.6499
```

结论：

- actuator delay 不是当前 A3 球拍下沉/错过来球的主因；
- 已撤回该临时代码改动；
- 下一步应继续检查 A3 默认站姿力学平衡和 PD gain，而不是继续改球路或 reward。

### 2026-06-26 A3 pingpong candidate 69 默认姿态和 damping 落地

边界：

- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改球路；
- 不修改核心成功 reward；
- 裸 `A3_T2D5_CFG` 保留，只覆盖 `A3_T2D5_PINGPANG_CFG`。

诊断搜索：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_standing_calibration \
  --task=a3_tt \
  --trial_steps=500 \
  --max_candidates=0 \
  --root_z_values=1.055 \
  --crouch_arm_modes=default \
  --hip_pitch_values=-0.18,-0.22 \
  --knee_values=0.38,0.50 \
  --ankle_pitch_values=-0.20,-0.26 \
  --stance_width_values=0.10,0.12,0.16 \
  --toe_out_values=0.0 \
  --waist_pitch_values=-0.04,-0.02,0.0 \
  --waist_damping_scale=2.0 \
  --leg_damping_scale=3.0 \
  --feet_damping_scale=3.0 \
  --output_csv=logs/a3_standing_calibration/a3_default_arm_crouch_search_500.csv \
  --headless
```

最佳候选：

```text
candidate_index=69
survival_steps=500/500
reset_seen=0
max_abs_roll_pitch=0.4123
max_root_z_drift=0.2003
max_foot_slip=0.0620
min_paddle_future_dist=0.0782
```

实际修改：

- `legged_lab/assets/a3/a3.py`
  - 新增 `A3_PINGPONG_READY_JOINT_POS`，基于当前持拍右臂，只覆盖 A3 pingpong 下肢：
    - `hip_pitch=-0.18`
    - `knee=0.50`
    - `ankle_pitch=-0.26`
    - `left_hip_roll=-0.16`
    - `right_hip_roll=0.16`
    - `left_ankle_roll=0.072`
    - `right_ankle_roll=-0.072`
  - 新增 `A3_PINGPONG_DAMPING_SCALES`：
    - waist damping x2
    - legs damping x3
    - feet damping x3
  - `A3_T2D5_PINGPANG_CFG` 覆盖 `init_state` 和 actuator damping。

验证 1：moving-ball zero-action 复测

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_pose_calibration \
  --task=a3_tt \
  --base_x=-0.26 \
  --base_y=0.35 \
  --ball_vx=-5.0 \
  --ball_vy=-0.05 \
  --ball_vz=1.45 \
  --max_steps=60 \
  --print_interval=10 \
  --no_visual_markers \
  --headless
```

关键对比：

```text
before:
  step 40 paddle_to_ball_dist ~= 0.3257 m
  paddle_touch_point.z ~= 0.6499

after:
  step 40 paddle_to_ball_dist ~= 0.0714 m
  paddle_touch_point.z ~= 0.8334
```

训练 contact threshold 当前为 `0.07 m`，因此 after 已经非常接近真实 contact，但没有通过放宽 threshold 达成。

验证 2：默认 zero-action 站立复测

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_standing_physics_diagnostics \
  --task=a3_tt \
  --pose=current \
  --phase=free \
  --root_z_values=1.055 \
  --free_steps=500 \
  --output_csv=logs/a3_standing_physics/a3_pingpong_candidate69_default_free_500_results.csv \
  --headless
```

结果：

```text
survival_steps=500/500
reset_seen=0
max_abs_roll_pitch=0.4298
max_root_z_drift=0.1967
max_foot_slip=0.0552
```

结论：

- A3 当前主要问题确实是默认姿态/控制保持，而不是 T1 workflow 或 reward 主流程；
- 本次修改让 A3 pingpong 默认姿态在来球窗口保持到可接触区附近；
- 下一步可以跑短训练探针，检查 `reward_contact` 是否开始非零。

### 2026-06-26 candidate 69 短训练探针

命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=30 \
  --run_name=a3_candidate69_probe \
  --headless \
  --predictor
```

TensorBoard event 汇总：

```text
run: logs/a3_table_tennis/2026-06-26_19-28-24_a3_candidate69_probe

Train/mean_episode_length:
  last=182.71
  max=197.47
  mean=148.097

Episode_Reward/reward_contact:
  last=0.000569
  max=0.043317
  mean=0.004452

Episode_Reward/reward_future_landing_dis:
  last=0.000822
  max=0.014071
  mean=0.001876

Episode_Reward/reward_future_touch_point:
  last=0.371515
  max=0.435552
  mean=0.370332

Episode_Reward/undesired_contacts:
  last=-37.5733
  min=-94.5801
  mean=-42.4215

Episode_Reward/termination_penalty:
  last=-1.04167
  min=-1.25
  mean=-0.4325

Episode_Reward/reward_table_success:
  0
```

与 T1 原始 run 前 30 iter 对比：

```text
T1 reward_contact mean ~= 0.00234
A3 candidate69 reward_contact mean ~= 0.00445

T1 undesired_contacts mean ~= -3.03
A3 candidate69 undesired_contacts mean ~= -42.42

T1 mean_episode_length mean ~= 73.44
A3 candidate69 mean_episode_length mean ~= 148.10
```

判断：

- `reward_contact` 已从之前 A3 的纯 0 变成偶发非零，说明“球路接触不到球拍”的根因已经被明显缓解；
- `reward_table_success` 仍为 0，说明这还不是完整学会乒乓球，只是解决了第一阶段的稀疏 contact；
- `undesired_contacts` 显著高于 T1，需要作为下一阶段问题处理；
- 由于 T1 也使用非足部 contact 惩罚，当前不能直接删除/大幅降低该 reward；
- 下一步应定位 A3 具体哪些 body 发生非足部接触，再决定是调默认姿态、缩小早期 action/noise，还是 A3-only 排除某些合理接触 body。

### 2026-06-26 candidate 69 4096 env 监督训练与接触来源诊断

监督训练命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_candidate69_supervised_500 \
  --headless \
  --predictor
```

训练在约第 53 iter 手动停止，原因是已经看到主要问题：

```text
run: logs/a3_table_tennis/2026-06-26_19-37-27_a3_candidate69_supervised_500

Train/mean_episode_length:
  last ~= 166.65
  last10_mean ~= 168.90

Episode_Reward/reward_contact:
  last ~= 0.02755
  last10_mean ~= 0.02380

Episode_Reward/reward_table_success:
  0

Episode_Reward/reward_future_landing_dis:
  last10_mean ~= 0.00970

Episode_Reward/reward_future_touch_point:
  last10_mean ~= 0.47733

Episode_Reward/reward_future_dis_ee:
  last10_mean ~= 0.35027

Episode_Reward/undesired_contacts:
  last ~= -46.80
  last10_mean ~= -49.91

Episode_Reward/termination_penalty:
  last10_mean ~= -0.353
```

判断：

- A3 当前已经不是完全碰不到球；4096 env 下 `reward_contact` 稳定非零；
- `reward_table_success` 仍为 0，说明球没有稳定回到对面桌面；
- 主要训练压力来自 `undesired_contacts`，不是大量 timeout/termination；
- 不能直接删除该项，因为 T1 原始任务也依赖非足端接触惩罚作为托底。

接触来源诊断使用当前 A3 配置、256 env、训练 reset/ball 配置，分别跑 zero-action 和 `action_std=0.20` 小随机动作。

zero-action 结果：

```text
first touch around step 40
min ball-paddle distance ~= 0.0503 m
has_touch_paddle step-sum = 744
table_success step-sum = 0

top non-foot contact bodies:
  torso_Link              count=8924
  pelvis_link             count=7964
  left_hand_Link          count=7837
  pingbang_ball_Link      count=4680
  right_wrist_roll_Link   count=3956
  left_wrist_yaw_Link     count=3649
  pingpang_red_Link       count=3397
  right_hand_pingpang_Link count=2271
```

small-random 结果：

```text
first_paddle_step = 40
first_nonfoot_step = 39
first_torso_step = 76
min ball-paddle distance ~= 0.0479 m
has_touch_paddle step-sum = 723
table_success step-sum = 0

top non-foot contact bodies:
  torso_Link              count=8938
  left_hand_Link          count=8276
  pelvis_link             count=7759
  right_wrist_roll_Link   count=4108
  pingbang_ball_Link      count=3895
  pingpang_red_Link       count=3433
  left_wrist_yaw_Link     count=3143
  right_hand_pingpang_Link count=1875
```

击球方向诊断：

```text
first touch observed around step 40
touched_envs = 38/256 by step 90
final opponent_table_prev = 0/256

step 40 tracked touched envs:
  ball_vx mean ~= -0.70, min ~= -2.67, max ~= 0.82
  ball_z mean ~= 0.90
  paddle_z mean ~= 0.84

step 48:
  ball_vx mean ~= -0.69
  ball_z mean ~= 0.65

step 60:
  ball_vx mean ~= -3.78
```

含义：

- A3 candidate69 能把球带到拍区附近，且几何接触信号出现；
- 但是球触拍后没有稳定反向飞向对面桌，`opponent_table` 始终为 0；
- `pingpang_red_Link`、`right_hand_pingpang_Link`、`pingbang_ball_Link` 等 A3 球拍/击球点 attachment link 也被当前 A3 `undesired_contacts` 计入；
- step 76 左右开始出现 `torso_Link` 接触，随后 pelvis/torso/hand/wrist 接触惩罚快速主导训练。

T1 零动作参照：

```text
task: t1_tt_eval
zero-action, 256 env, 90 steps
touched_envs = 0/256
reset_events = 1
top non-foot contact bodies:
  H2
  Trunk
  right_hand_link
```

边界判断：

- “未训练/零动作策略会不稳”不是 A3 独有，T1 也会出现；
- A3 当前的独有问题是：已经能碰到球，但接触后球没有被有效回击，同时 A3 的球拍/标定球/手腕 attachment link 会进入非足端接触惩罚；
- 下一步若修改代码，应优先做 A3-only 兼容：
  1. 先考虑把 A3 球拍和击球点 marker link 从 `undesired_contacts` 排除，因为这些是任务允许/期望的接触体；
  2. 不排除 torso/pelvis/普通手臂摔碰，因为那是真实失稳信号；
  3. 继续通过 A3 默认姿态/右腕拍面/右臂 regularization 改善击球后球速方向；
  4. 不修改 T1、不修改公共 `TTEnv`、不放宽 `reward_table_success`。

### 2026-06-26 A3 球拍/击球点 contact 白名单落地与复测

实际代码修改：

```text
legged_lab/envs/a3_tt/a3_tt_config.py
```

新增 A3-only 白名单，并将 `reward.undesired_contacts` 的 body regex 改为排除这些 body：

```text
left_ankle_roll_Link
right_ankle_roll_Link
pingpang_red_Link
pingpang_black_Link
right_hand_pingpang_Link
pingbang_ball_Link
```

边界：

- 只修改 A3 task 配置；
- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不排除 `torso_Link`、`pelvis_link`、普通手臂/手腕等真实失稳接触；
- 不放宽 `reward_contact` / `reward_table_success`。

静态验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/a3_tt/a3_tt_config.py
```

结果：通过。

运行时解析验证：

```text
resolved_undesired_body_names:
  ^(?!(left_ankle_roll_Link|right_ankle_roll_Link|pingpang_red_Link|pingpang_black_Link|right_hand_pingpang_Link|pingbang_ball_Link)$).*

contact bodies excluded from undesired:
  left_ankle_roll_Link
  right_ankle_roll_Link
  right_hand_pingpang_Link
  pingbang_ball_Link
  pingpang_black_Link
  pingpang_red_Link
```

同一接触诊断中旧/新计数对比：

```text
old_nonfoot_count_total = 45746
new_undesired_count_total = 35724
reduction = 10022

top remaining undesired bodies:
  torso_Link
  left_hand_Link
  pelvis_link
  right_wrist_roll_Link
  left_wrist_yaw_Link
  right_wrist_yaw_Link
  left_elbow_Link
```

这说明白名单只移除了球拍/击球点 attachment 的任务允许接触，摔倒/撞桌/手臂接触仍然保留惩罚。

短训练复测命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=30 \
  --run_name=a3_allowed_paddle_contacts_probe \
  --headless \
  --predictor
```

同口径 TensorBoard 对比：

```text
before: 2026-06-26_19-28-24_a3_candidate69_probe
after:  2026-06-26_19-57-00_a3_allowed_paddle_contacts_probe

reward_contact mean:
  before ~= 0.00445
  after  ~= 0.00274

reward_table_success:
  before = 0
  after  = 0

undesired_contacts mean:
  before ~= -42.42
  after  ~= -42.24

undesired_contacts last10_mean:
  before ~= -44.41
  after  ~= -46.88

termination_penalty mean:
  before ~= -0.4325
  after  ~= -0.4319
```

结论：

- 该修改在语义上是正确的 A3 适配：球拍/击球点 marker 与球接触不应被视为 undesired body contact；
- 但是它没有解决当前训练主问题，因为剩余大头仍是 `torso/pelvis/hand/wrist` 等真实失稳接触；
- `reward_table_success` 仍为 0，下一步不应继续删除接触惩罚；
- 下一步应转向 A3 右腕/拍面/右臂默认姿态和击球后稳定性，使球触拍后能反向飞向对面桌。

### 2026-06-26 A3 右腕 scan101 默认姿态落地与复测

问题：

当前 A3 右腕默认姿态虽然能让球进入 contact 区，但触球后 `ball_vx` 没有稳定变为正值，球没有朝对面桌方向返回。

诊断：

固定 A3 base pose 和 easy serve，保持右臂其他关节不变，只扫描右腕：

```text
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

125 个候选中，43 个候选在触球后 3 step 的 `vx3 > 0`。

选择候选 `scan101`：

```text
right_wrist_roll_joint  = +0.25
right_wrist_pitch_joint = -0.75
right_wrist_yaw_joint   = -1.40

scan result:
  touch_step = 41
  minD ~= 0.0750
  vx3 ~= +1.237
  vz3 ~= +0.135
  x20 ~= -1.224
  z20 ~= +0.894
  vx20 ~= +0.644
  reset = False
```

实际代码修改：

```text
legged_lab/assets/a3/a3.py
```

只在 `A3_PINGPONG_READY_JOINT_POS` 中覆盖右腕，不改裸 `A3_T2D5_CFG` 默认姿态：

```text
right_wrist_roll_joint  -> +0.25
right_wrist_pitch_joint -> -0.75
right_wrist_yaw_joint   -> -1.40
```

静态验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py
```

结果：通过。

固定球路复测：

```text
default right wrist after change:
  right_wrist_roll_joint  = +0.2500
  right_wrist_pitch_joint = -0.7500
  right_wrist_yaw_joint   = -1.4000

128 env fixed easy serve:
  touched = 128/128
  first_touch_step = 41
  reset_events = 0
  vx3 mean ~= +1.2360
  vz3 mean ~= +0.1869
  vx20 mean ~= +0.5850
  z20 mean ~= +0.8953
```

短训练复测：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=30 \
  --run_name=a3_scan101_wrist_probe \
  --headless \
  --predictor
```

同口径 TensorBoard 对比：

```text
candidate69 baseline:
  reward_contact mean ~= 0.00445
  reward_future_landing_dis mean ~= 0.00188
  undesired_contacts mean ~= -42.42
  termination_penalty mean ~= -0.4325

paddle contact whitelist:
  reward_contact mean ~= 0.00274
  reward_future_landing_dis mean ~= 0.00148
  undesired_contacts mean ~= -42.24
  termination_penalty mean ~= -0.4319

scan101 wrist:
  reward_contact mean ~= 0.03989
  reward_future_landing_dis mean ~= 0.01778
  reward_future_touch_point mean ~= 0.39215
  reward_future_dis_ee mean ~= 0.29040
  undesired_contacts mean ~= -45.97
  termination_penalty mean ~= -0.2583
  reward_table_success = 0
```

结论：

- scan101 明显改善了 A3 的触球和击球后落点相关信号；
- 它没有直接产生 `reward_table_success`，说明默认姿态只能解决“反弹方向起点”，完整回球仍需要策略主动挥拍和身体稳定；
- `undesired_contacts` 仍偏高，且 last10 更高，说明下一步不能继续只调拍面，需要处理右臂/身体稳定性；
- 当前建议保留 scan101 作为 A3 pingpong 初始右腕姿态，下一步优化右臂默认姿态、右臂动作探索幅度/regularization 或 staged reward。

### 2026-06-26 A3 初始探索噪声 0.15 探针

目的：

- scan101 已经改善触球和触球后 `ball_vx`，但短训里 `undesired_contacts` 仍偏高；
- 接触来源诊断显示，first touch 之后 pelvis/torso/left hand/right wrist 等不稳定接触开始出现；
- 先做 A3-only 的最小探索约束，检查是否能降低早期身体碰撞。

修改：

```text
legged_lab/envs/a3_tt/a3_tt_config.py

A3_INITIAL_POLICY_NOISE_STD:
  0.20 -> 0.15
```

边界：

- 未修改 T1；
- 未修改公共 `TTEnv`；
- 未修改 `A3_INITIAL_ACTION_SCALE`；
- 未修改 `reward_contact` / `reward_table_success`；
- 未放宽 contact whitelist。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/mdp/rewards.py
```

结果：通过。

短训：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=30 \
  --run_name=a3_scan101_noise015_probe \
  --headless \
  --predictor
```

同口径对比 scan101 noise=0.20：

```text
reward_contact mean:          0.03989 -> 0.02501
reward_contact last10:        0.04433 -> 0.02330
reward_future_landing mean:   0.01778 -> 0.01396
reward_future_touch_point:    0.39215 -> 0.36600
undesired_contacts mean:    -45.96783 -> -41.68706
undesired_contacts last10:  -57.90598 -> -48.25888
termination_penalty mean:    -0.25833 -> -0.38796
reward_table_success:         0       -> 0
```

结论：

- `0.15` 略微降低 `undesired_contacts`，但牺牲了一部分触球和落点信号；
- 这不是解决 `table_success=0` 的根因，只能作为早期稳定性候选保留；
- 下一步继续调 A3 右腕/拍面/右臂默认姿态，使触球后过网高度和对面桌落点更合理。

补充零动作轨迹诊断：

```text
32 env, hold default, 140 step:
  hits = 15/32
  opponent_table_any = 0/32
  own_table_any = 32/32
  reset = 0/32
  first_hit_step mean ~= 41.2
  hit_vx mean ~= +1.09, min ~= -0.38
  hit_vz mean ~= +0.49
  pred_land_x_at_hit mean ~= -1.40
  actual_net_cross_z mean ~= 0.786
```

这说明当前 scan101 的静态几何有触球基础，但默认命中率和过网高度都不够，不能靠单纯降低探索噪声解决。

### 2026-06-26 A3 lift1 右腕与初始噪声 0.10

目的：

- scan101 的固定球路 `ball_vx` 已经改善，但训练随机球路下默认命中率不足；
- 扩展 wrist scan 显示，继续只追求 `vx > 0` 不够，需要提高早期触球覆盖率；
- base x 前移会引入 reset/不稳定，暂不作为默认配置；
- 选择更仰的 A3 右腕候选 lift1，并进一步降低初始策略噪声以减少早期碰撞。

修改：

```text
legged_lab/assets/a3/a3.py

A3_PINGPONG_READY_JOINT_POS:
  right_wrist_roll_joint:   +0.25 -> +0.00
  right_wrist_pitch_joint:  -0.75 -> -1.15
  right_wrist_yaw_joint:    -1.40 -> -1.40

legged_lab/envs/a3_tt/a3_tt_config.py

A3_INITIAL_POLICY_NOISE_STD:
  0.15 -> 0.10
```

边界：

- 未修改 T1；
- 未修改公共 `TTEnv`；
- 未修改球路；
- 未修改 `reward_contact` / `reward_future_pass_net` / `reward_table_success`；
- 未移动 base x/y；
- 未放宽碰撞白名单。

诊断依据：

```text
hold-default, 32 env, 训练随机球路:

scan101:
  hits = 15/32
  opponent_table_any = 0/32
  actual_net_cross_z mean ~= 0.786

lift1:
  hits = 23/32
  opponent_table_any = 0/32
  resets = 0/32
  hit_vz mean ~= +1.120
```

静态验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/mdp/rewards.py
```

结果：通过。

短训验证：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=30 \
  --run_name=a3_lift1_wrist_noise010_probe \
  --headless \
  --predictor
```

对比 scan101 + noise=0.15：

```text
reward_contact mean:          0.02501 -> 0.07834
reward_future_landing mean:   0.01396 -> 0.02743
reward_future_touch_point:    0.36600 -> 0.44659
reward_future_dis_ee:         0.27077 -> 0.31004
undesired_contacts mean:    -41.68706 -> -37.15524
mean_reward:               -381.91359 -> -362.24295
reward_future_pass_net:       0       -> 0
reward_table_success:         0       -> 0
```

对比 lift1 + noise=0.15：

```text
reward_contact mean:          0.06210 -> 0.07834
reward_future_landing mean:   0.02482 -> 0.02743
undesired_contacts mean:    -44.60379 -> -37.15524
action_rate_l2 mean:         -0.01472 -> -0.00791
termination_penalty mean:    -0.51713 -> -0.55602
```

结论：

- 当前保留 `lift1 + A3_INITIAL_POLICY_NOISE_STD=0.10`；
- 它明显改善了 A3 早期触球密度和落点相关信号，并降低了整体 undesired contact；
- `reward_future_pass_net` / `reward_table_success` 仍为 0，下一阶段需要针对 A3 的主动挥拍、过网高度和 staged reward/curriculum 做专项处理，而不是继续只旋转手腕。

### 2026-06-26 A3 hit outcome diagnostics 与 rollneg/pitch125 右腕

目的：

- 训练曲线中 `reward_contact` 已经不再完全稀疏；
- 但 `reward_future_pass_net` 和 `reward_table_success` 仍为 0；
- 需要先确认触拍后 `ball_vx/vz` 和实际过网高度，而不是继续靠长训试错。

新增：

```text
legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

功能：

- A3-only，只允许 `a3_tt` / `a3_tt_eval`；
- zero-action rollout，用于诊断默认姿态/候选右臂姿态；
- 支持 `--joint` / `--rwr --rwp --rwy` 等右臂右腕覆盖；
- 输出首个触拍事件的：
  - `hit_vx/vy/vz`；
  - 预测/实际网口高度；
  - 是否触自己桌、对面桌；
  - reset 数量。

扫描结果：

```text
当前默认 lift1:
  first_hit:            40/64
  hit_vx_positive:      35/64
  hit_vx mean:          +0.560
  hit_vz mean:          +0.915
  actual_z_at_net mean: 0.715
  actual_z_at_net max:  0.997
  opponent_table:       0/64
  reset_seen:           0/64

candidate roll=-0.25, pitch=-1.25, yaw=-1.40:
  first_hit:            47/64
  hit_vx_positive:      46/64
  hit_vx mean:          +0.779
  hit_vz mean:          +0.865
  actual_z_at_net mean: 0.814
  actual_z_at_net max:  1.004
  opponent_table:       0/64
  reset_seen:           0/64
```

修改：

```text
legged_lab/assets/a3/a3.py

A3_PINGPONG_READY_JOINT_POS:
  right_wrist_roll_joint:   +0.00 -> -0.25
  right_wrist_pitch_joint:  -1.15 -> -1.25
  right_wrist_yaw_joint:    -1.40 -> -1.40
```

边界：

- 未修改 T1；
- 未修改公共 `TTEnv`；
- 未修改 reward；
- 未修改球路；
- 未放宽成功条件。

阶段判断：

- 该右腕候选比 lift1 更适合作为 A3 默认起点；
- 但所有 wrist-only 候选仍然 `actual_net_clear=0`、`opponent_table=0`；
- 这说明后续需要 A3-only 主动挥拍/staged shaping，让策略学会把触拍后的球速提高到可过网落台的量级。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/assets/a3/a3.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/mdp/rewards.py \
  legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

结果：通过。

修改后默认姿态诊断：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_hit_outcome_diagnostics \
  --task=a3_tt \
  --num_envs=64 \
  --max_steps=140 \
  --random_ball \
  --headless
```

结果：

```text
first_hit:                 47/64
hit_vx_positive:           46/64
hit_vz_positive:           41/64
actual_crossed_net:        41/64
actual_z_at_net mean:      0.8141
actual_z_at_net max:       1.0042
opponent_table_after_hit:  0/64
reset_seen:                0/64
```

短训验证：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=30 \
  --run_name=a3_rollneg_pitch125_probe \
  --headless \
  --predictor
```

结果：

```text
run: logs/a3_table_tennis/2026-06-26_21-08-40_a3_rollneg_pitch125_probe
reward_contact mean:             0.09754
reward_future_touch_point mean:  0.58099
reward_future_landing_dis mean:  0.03348
reward_future_pass_net mean:     0.00000
reward_table_success mean:       0.00000
undesired_contacts mean:       -45.38344
termination_penalty mean:       -0.18657
```

结论：

- 保留 `right_wrist_roll=-0.25, right_wrist_pitch=-1.25, right_wrist_yaw=-1.40`；
- 这一步提升了触球/对齐相关信号；
- 但 A3 仍需要额外的 A3-only 阶段性奖励或 curriculum 来学习主动挥拍和过网。

### 2026-06-26 A3 staged hit velocity/net reward

目的：

- A3 已经能更稳定触拍，但触拍后球速仍不足以过网；
- `reward_future_pass_net` 和 `reward_table_success` 是结果型稀疏奖励，早期几乎为 0；
- 增加 A3-only staged reward，让策略在触拍事件上先学会把球打向 +x 并带有上抛/过网潜力。

修改：

```text
legged_lab/mdp/rewards.py
  reward_hit_ball_velocity_net_target(...)

legged_lab/envs/a3_tt/a3_tt_config.py
  reward_hit_ball_velocity_net = RewTerm(...)
```

设计边界：

- 只在 `env.ball_landing_dis_rew` 触拍事件上计算；
- 仍要求 `ball_vx > min_vx`；
- 不修改 `reward_future_pass_net`；
- 不修改 `reward_table_success`；
- 不修改 T1 配置；
- 不修改公共 `TTEnv`；
- 不修改球路。

第一次实现观察：

```text
run: logs/a3_table_tennis/2026-06-26_21-13-17_a3_staged_hit_velocity_probe
result: stopped around iter 13
reason: reward_hit_ball_velocity_net was mostly 0 because max_t_net was used as a hard mask.
```

修正：

- 移除 `t_net <= max_t_net` 硬 mask；
- 改为 component score：
  - `vx_score = clamp(vx / vx_target, 0, 1)`；
  - `vz_score = clamp(vz / vz_target, 0, 1)`；
  - `z_score` 使用估算网口高度；
  - `time_score` 使用超出 `max_t_net` 的软惩罚；
- A3 初始权重保持 `20.0`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/assets/a3/a3.py \
  legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

结果：通过。

短训验证：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=10 \
  --run_name=a3_staged_hit_velocity_component_probe \
  --headless \
  --predictor
```

结果：

```text
run: logs/a3_table_tennis/2026-06-26_21-17-04_a3_staged_hit_velocity_component_probe
reward_contact mean:                0.10183
reward_hit_ball_velocity_net mean:  0.00614
reward_hit_ball_velocity_net last:  0.01666
reward_future_landing_dis mean:     0.03553
reward_future_pass_net mean:        0.00000
reward_table_success mean:          0.00000
undesired_contacts mean:          -27.52767
termination_penalty mean:          -0.00833
```

结论：

- 新 staged reward 现在有非零训练信号；
- 量级较保守，未压过原有 landing/stability 项；
- 仍需 250-500 iter 观察它是否推动 `hit_vx`、`actual_z_at_net` 和最终 `reward_future_pass_net`。

### 2026-06-26 A3 staged hit velocity 500-run monitoring

命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_staged_hit_velocity_500 \
  --headless \
  --predictor
```

run：

```text
logs/a3_table_tennis/2026-06-26_21-20-22_a3_staged_hit_velocity_500
```

监控结果：

```text
stopped at iter:                      187/500
reward_contact last50:                0.03991
reward_hit_ball_velocity_net last50:  0.00401
reward_future_landing_dis last50:     0.01994
reward_future_pass_net last50:        0.000002
reward_table_success last50:          0.000000
undesired_contacts last50:           -0.07461
termination_penalty last50:          -1.99833
mean_reward last50:                 -19.00406
mean_episode_length last50:         105.48380
```

结论：

- 当前配置能够把 undesired contact 压低；
- 但任务相关指标没有持续上升，`table_success` 始终为 0；
- `termination_penalty` 接近 `-2.0`，episode length 降低，说明策略可能在走保守规避/短 episode 路线；
- 已停止该 run，避免继续消耗 GPU。

后续方向：

- 需要修改 A3 staged reward 或训练日志指标，让触球后 `hit_vx/hit_vz/actual_z_at_net` 更直接地反馈给策略和监控；
- 不建议用当前权重继续长训。

### 2026-06-26 A3 policy diagnostics and landing reward compatibility

背景：

- `a3_reward_rebalance_120` 训练稳定，但 `reward_future_pass_net` 仍约 `1e-6`，`reward_table_success` 为 0；
- 仅从 TensorBoard 看不清是没有触球，还是触球方向/落点错误；
- 原 `reward_future_landing_dis = threshold - distance` 在 A3 上可能奖励己方落台，因为 `threshold=3.0` 太宽，T1 能越网时这个问题不明显，A3 则会形成局部最优。

修改：

- `legged_lab/scripts/a3_hit_outcome_diagnostics.py`
  - 新增 `--load_run`、`--checkpoint`、`--predictor`；
  - 支持加载训练 checkpoint 做 policy outcome diagnostics；
  - 统计 reset 原因：`reset_low_z`、`reset_x_low`、`reset_x_high`、`reset_y_low`、`reset_y_high`、`reset_timeout`；
  - 输出 `reset_robot_pos`，便于区分摔倒、越界和 serve timeout。

- `legged_lab/mdp/rewards.py`
  - 新增 `reward_future_opponent_landing_target()`；
  - 新增 `penalty_future_own_landing_after_hit()`；
  - 新增 `reward_future_landing_x_progress()`；
  - 这些函数只作为可选 reward term 存在，不改变 T1 默认配置。

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - A3 专项关闭旧的宽松 `reward_future_landing_dis`；
  - 新增 `reward_future_opponent_landing`；
  - 新增 `penalty_future_own_landing`；
  - 新增 `reward_future_landing_x_progress`；
  - 当前 tuned 参数：

```text
reward_future_landing_dis:
  weight: 0.0

reward_future_opponent_landing:
  weight: 120.0
  target_x: 1.15
  target_y: 0.0
  min_x: 0.0
  std: 1.0

reward_future_landing_x_progress:
  weight: 120.0
  min_x: -3.0
  target_x: 1.15
  target_y: 0.0
  y_std: 1.0
  y_weight: 0.25

penalty_future_own_landing:
  weight: -40.0
  max_x: 0.0
```

边界：

- 不修改 `legged_lab/envs/t1_tt/t1_tt_config.py`；
- 不修改公共 `TTEnv`；
- 不改球路；
- 不改 T1 原始 workflow；
- 真实过网和对面落台成功标准保持不变。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

结果：通过。

policy outcome diagnostics：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_hit_outcome_diagnostics \
  --task=a3_tt \
  --num_envs=128 \
  --max_steps=220 \
  --print_interval=50 \
  --random_ball \
  --load_run 2026-06-26_21-41-47_a3_reward_rebalance_120 \
  --checkpoint model_119.pt \
  --predictor \
  --headless
```

结果：

```text
first_hit:                 76/128 (0.594)
hit_vx_positive:           63/128 (0.492)
hit_vz_positive:           70/128 (0.547)
actual_crossed_net:        70/128 (0.547)
actual_net_clear:           0/128
opponent_table_after_hit:   0/128
own_table_after_hit:       76/128 (0.594)
reset_seen:                82/128 (0.641)
reset_low_z:                8/128 (0.062)
reset_timeout:             74/128 (0.578)
hit_x mean:                -1.5547
hit_z mean:                 0.9776
hit_vx mean:                0.5620
hit_vz mean:                1.0069
actual_z_at_net mean:       0.8119
```

结论：

- A3 当前不是完全没触球，而是多数触球后落在己方桌面；
- 真实过网高度不够，`actual_net_clear` 为 0；
- reset 主要来自 timeout/serve limit，不是立即摔倒；
- 奖励修正方向应聚焦“把触球后的落点和速度从己方逐步推向对面”，而不是大改球路或 T1 流程。

短训练验证 1：

```text
run: logs/a3_table_tennis/2026-06-26_22-08-07_a3_opponent_landing_probe
num_envs: 64
max_iterations: 30

reward_contact last10:                 0.06568314
reward_future_landing_dis last10:      0.00000000
reward_future_opponent_landing last10: 0.00000000
penalty_future_own_landing last10:    -0.05894444
reward_hit_ball_velocity_net last10:   0.02984486
reward_future_pass_net last10:         0.00000000
reward_table_success last10:           0.00000000
undesired_contacts last10:           -48.95165749
termination_penalty last10:           -0.52361113
```

判断：opponent-only landing 仍过稀疏，因此加入 x-progress。

短训练验证 2：

```text
run: logs/a3_table_tennis/2026-06-26_22-11-38_a3_landing_x_progress_probe
num_envs: 64
max_iterations: 30

reward_future_landing_x_progress last10:  0.00555782
penalty_future_own_landing last10:       -0.06866667
reward_hit_ball_velocity_net last10:      0.04492416
reward_future_pass_net last10:            0.00000004
reward_table_success last10:              0.00000000
```

判断：x-progress 有信号但偏弱，随后将 `min_x` 调到 `-3.0`、权重调到 `120.0`。

短训练验证 3：

```text
run: logs/a3_table_tennis/2026-06-26_22-14-44_a3_landing_x_progress_tuned_smoke
num_envs: 64
max_iterations: 10

reward_future_landing_x_progress mean/last:  0.05438230 / 0.09806083
penalty_future_own_landing mean/last:       -0.04977778 / -0.08500000
reward_hit_ball_velocity_net mean/last:      0.03170094 / 0.05187657
reward_future_pass_net mean/last:            0.00000004 / 0.00000000
reward_table_success mean/last:              0.00000000 / 0.00000000
```

结论：

- 当前 tuned x-progress 已经能提供可见早期梯度；
- 还没有证明能带来过网/对面落台，需要 4096 env / 250-500 iter 继续观察；
- 如果 `reward_future_landing_x_progress` 不升、`penalty_future_own_landing` 不向 0 收敛，或 `undesired_contacts` 长期约 `-50`，应停止并继续调 A3-only 奖励/姿态。

## 2026-06-26/27 A3 per-joint scale, actual own-table penalty, and swing probes

目标：

- 继续保持 T1 workflow 不变；
- A3-only 调整，解决“能触球但多数先落己方桌、不过网”的局部最优；
- 在进入 1000/10000 长训前，用 250 probe 和物理诊断确认是否真的改善击球。

代码变更：

- `legged_lab/envs/base/tt_env.py`
  - `robot.action_scale` 兼容 list/tuple per-joint scale；
  - 暴露 `has_touch_own_table_just_now`，用于 A3 实际己方落台惩罚；
  - paddle body / touch offset 已改为配置项，避免硬编码 T1 body index。
- `legged_lab/mdp/rewards.py`
  - 新增 `penalty_own_table_after_paddle_hit()`；
  - 保留 A3 迁移用的 `reward_future_landing_x_progress()`、`reward_hit_ball_velocity_net_target()`、`reward_hit_net_clearance_progress()` 等阶段奖励。
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 当前保留 stage-2 moderated 配置：
    - right shoulder action scale: `0.14`
    - right elbow action scale: `0.18`
    - right wrist action scale: `0.20`
    - `reward_contact=110`
    - `reward_hit_ball_velocity_net=160`
    - `reward_hit_net_clearance_progress=30`
    - `penalty_actual_own_table_after_hit=-80`
- `legged_lab/scripts/a3_hit_outcome_diagnostics.py`
  - 支持加载 policy checkpoint；
  - 输出 first-hit、hit velocity、actual crossed/net-clear、own/opponent table、reset cause 等物理指标。
- `legged_lab/scripts/a3_right_arm_pose_grid_search.py`
  - 新增 A3 右臂/右腕静态姿态批量网格搜索脚本。

关键实验：

```text
run: 2026-06-26_23-05-54_a3_perjoint_scale_250
first_hit: 93/128
actual_crossed_net: 88/128
own_table_after_hit: 93/128
hit_vx mean: 0.6792
hit_vz mean: 0.8804
actual_z_at_net mean: 0.8351
```

```text
run: 2026-06-26_23-34-03_a3_actual_own_penalty_250
first_hit: 92/128
actual_crossed_net: 82/128
own_table_after_hit: 92/128
hit_vx mean: 0.6379
hit_vz mean: 0.9581
actual_z_at_net mean: 0.8456
```

```text
run: 2026-06-27_00-04-14_a3_stage2_moderated_250
first_hit: 98/128
actual_crossed_net: 92/128
own_table_after_hit: 98/128
hit_vx mean: 0.6313
hit_vy mean: 0.5092
hit_vz mean: 0.8450
actual_z_at_net mean: 0.8346
```

right-arm grid / zero-action 对照：

```text
grid csv: logs/a3_table_tennis/a3_right_arm_grid_forward_20260627_0026.csv

best deterministic candidate id=307:
  re=0.55, rwr=-0.30, rwp=-1.40, rwy=-1.40
  hit_vx=1.4007, hit_vy=0.0900, actual_z_at_net=0.9350

id=307 random-ball zero-action:
  hit_vx mean=0.7222, hit_vy mean=0.3466, actual_z_at_net mean=0.8117

configured default random-ball zero-action:
  hit_vx mean=0.7428, hit_vy mean=0.2489, actual_z_at_net mean=0.8542
```

stage-3 shoulder/elbow scale smoke：

```text
run: 2026-06-27_00-32-01_a3_stage3_shoulder_elbow_smoke
reward_hit_ball_velocity_net tail10: 0.051363
undesired_contacts tail10: -79.077016
```

该版本比 stage-2 moderated 更不稳且 hit-velocity reward 更低，已回退。

当前判断：

- A3 已经能稳定触球，但仍没有学到真正乒乓球回球；
- 主要失败模式是触拍后速度和高度不足，且横向速度偏大，导致球先落己方桌；
- `reward_future_pass_net` / `reward_table_success` 持续为 0，说明现在不适合直接长训 10000 iter；
- 下一步需要更明确的主动挥拍 curriculum/动作先验，而不是继续单纯增加训练轮数或盲目加大所有右臂动作。

验证：

```text
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/base/tt_env.py \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/scripts/a3_hit_outcome_diagnostics.py \
  legged_lab/scripts/a3_right_arm_pose_grid_search.py
```

结果：通过。

## 2026-06-27 latest A3 reward-stage conclusion

最新结论索引：

- stage4d `model_2549.pt` 仍是当前稳定回退点；
- stage4f/g/h 均已完成 1024-env / 50-iter 与 256-env 随机球诊断；
- stage4g 的真实对面落台最高，但仅 `32/256`，相比 stage4d `31/256` 没有稳定意义，且 `reset_low_z=47/256`；
- stage4h 尝试约束过高弧线/横向速度，但 `opponent_table_after_hit=24/256`，不建议继续该 reward 方向长训；
- 下一步应优先做 A3 专用 curriculum/几何回查，而不是继续增加 reward 权重：
  - 固定或更窄中心球路；
  - 诊断击球瞬间拍面法向、球拍中心、球-拍接触点；
  - 中心球路稳定对面落台后，再逐步放宽球路随机范围。

## 2026-06-29 A3 play WebRTC black-screen fix

问题：

- `play.py --task=a3_tt_stage4d --livestream 2` 加载已训练模型后，WebRTC 客户端黑屏；
- 对比发现 `preview.py` 会显式设置 Isaac viewport camera，而 `play.py` 没有设置；
- 本次只修复可视化入口，不修改训练、奖励、reset、PD、A3/T1 task 注册或 checkpoint 加载行为。

变更：

- 为 `legged_lab/scripts/play.py` 增加可选参数：
  - `--camera_eye x y z`
  - `--camera_target x y z`
- 当非 headless 或启用 livestream 时，在环境创建后调用 `env.sim.set_camera_view(...)`；
- 默认 camera 复用 `preview.py` 视角：eye `[-3.0, 6.0, 1.0]`，target `[-3.0, 0.0, 1.0]`。

验证：

- 已执行 `/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/play.py`，通过；
- 待用原 WebRTC play 命令确认画面恢复。

## 2026-06-29 A3 play WebRTC step-loop render/update follow-up

复测问题：

- 第一次 camera patch 后，初始化阶段 Isaac Sim UI 可见；
- 进入 play / `env.step` 后 WebRTC 仍然黑屏；
- 最新 Isaac 日志没有显示仿真崩溃，WebRTC 是连接后断开；
- `tt_webrtc_probe.py` 已经采用固定 camera、显式 render、显式 `simulation_app.update()` 和短 sleep 的方式规避同类问题。

变更：

- `legged_lab/scripts/play.py` 增加：
  - `--keep_camera_interval`，默认 1，即 GUI/WebRTC 下每步重新固定 camera；
  - `--visualize_sleep`，默认 livestream 下 0.03 秒，本地 GUI 下 0；
- `play.py` 默认 camera 从 `preview.py` 的通用视角调整为 `tt_webrtc_probe.py` 使用的 TT/A3 WebRTC 视角：
  - eye `[-3.2, -2.0, 1.6]`
  - target `[-1.8, 0.35, 0.85]`
- play loop 中 `env.step(actions)` 后，在 GUI/WebRTC 模式下额外执行：
  - 按间隔 `_set_camera_view(env)`；
  - `env.sim.render()`；
  - `simulation_app.update()`；
  - 可选 `time.sleep(...)`。

边界：

- 只影响 `legged_lab.scripts.play` 可视化入口；
- 不影响 `train.py`、headless 训练、reward、PD、reset、A3/T1 task 配置或 checkpoint 行为。

验证：

- 已执行 `/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/play.py`，通过；
- 待 WebRTC 复测。

## 2026-06-29 A3 play/probe WebRTC difference narrowing

复测结论：

- `tt_webrtc_probe.py --task=a3_tt_stage4d --mode=zero --livestream 2` 可正常看到画面；
- `play.py --task=a3_tt_stage4d --load_run ... --checkpoint ... --predictor --livestream 2` 仍黑屏；
- 因此问题不在底层 WebRTC 或 A3 场景加载，而在 `play.py` 特有路径。

变更：

- `legged_lab/scripts/play.py` 增加 `--skip_export`：
  - 显式传入后跳过 predictor/JIT/ONNX 导出；
  - 默认不跳过，避免改变原有 play 导出行为；
- `legged_lab/scripts/play.py` 增加 `--max_play_steps`：
  - 便于 WebRTC 短跑验证；
  - 默认 0，表示持续运行；
- `play.py` 的 `_set_camera_view` 改为与 `tt_webrtc_probe.py` 一致：
  - 将 `--camera_eye` / `--camera_target` 视为 env-local 坐标；
  - 如果存在 `env.scene.env_origins[0]`，自动加到 eye/target 上。

验证：

- 已执行 `/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/play.py`，通过；
- 待使用 `--skip_export` WebRTC 复测。

## 2026-06-29 A3 play action/predictor isolation

复测结论：

- `play.py --skip_export` 后仍黑屏；
- 最新日志中已经没有 ONNX traceback，但 WebRTC 仍断开；
- `play.py` 进程未崩溃，说明不是 Python fatal error，而是策略 play 循环下视频流没有有效画面；
- `tt_webrtc_probe.py` 能看，说明 zero-action 环境 step + render/update 路径可用。

变更：

- `legged_lab/scripts/play.py` 增加 `--action_mode policy|zero|random|sine`：
  - 默认 `policy`，保持原行为；
  - `zero` 用于加载 checkpoint 后仍使用零动作，和 `tt_webrtc_probe.py` 对齐；
- 增加 `--disable_predictor_update`：
  - 仍可用 predictor runner 加载 checkpoint；
  - 但跳过每步 `_record_ball_positions()` / `_maybe_predict_and_update_env()`；
- 增加 `--warmup_render_steps`：
  - livestream 下默认 30；
  - 在进入 step loop 前先固定 camera 并刷新若干帧；
- 增加内部 `_refresh_visualization(...)` 和 `_make_debug_actions(...)`，复用 play 的渲染刷新逻辑。

验证：

- 已执行 `/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/play.py`，通过；
- 下一步复测：
  - 先 `--action_mode zero --disable_predictor_update --skip_export`；
  - 再逐步打开 `policy` 和 predictor update。

## 2026-06-29 A3 play env-headless / runner-load isolation

复测结论：

- `play.py --skip_export --action_mode zero --disable_predictor_update` 仍在 50-100 step 左右黑屏；
- 因此已排除 ONNX/export、策略动作、predictor 每步更新；
- `tt_webrtc_probe.py` 能看，且其环境创建固定 `headless=False`；
- `play.py` 原先用 `env_class(env_cfg, args_cli.headless)`，livestream/no-window 下可能与 probe 的渲染路径不一致。

变更：

- `legged_lab/scripts/play.py` 在 `--livestream > 0` 时创建环境强制使用 `env_headless=False`；
- 增加终端输出：
  - `args_headless`
  - `env_headless`
  - `livestream`
- 增加 `--no_load_runner`：
  - 完全跳过 checkpoint/runner/policy 加载；
  - 仅允许 `--action_mode zero/random/sine`；
  - 用于判断黑屏是否由 runner/checkpoint 加载引入。

验证：

- 已执行 `/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/play.py`，通过；
- 下一步先运行 `--no_load_runner --action_mode zero` 对照。

## 2026-06-29 A3 WebRTC policy probe

复测结论：

- `play.py` 多轮收敛后仍黑屏，包括 zero-action、禁用 predictor update、跳过 export 等情况；
- 终端仍持续输出 `[Play] Success...`，说明仿真循环在运行，但视频流路径不可靠；
- `tt_webrtc_probe.py` 已确认可见，因此改为给 probe 增加 policy 加载能力，用已知可靠的 WebRTC loop 观看训练模型。

变更：

- 扩展 `legged_lab/scripts/tt_webrtc_probe.py`：
  - `--mode` 增加 `policy`；
  - 增加 `--load_run`；
  - 增加 `--checkpoint`；
  - 增加 `--predictor`；
  - 增加 `--disable_predictor_update`；
- 新增内部 `_load_policy(...)`：
  - 使用 task 的 `agent_cfg.experiment_name` 定位 log root；
  - 使用 `OnPolicyRunner` 或 `OnPolicyPredictorRegressionRunner` 加载 checkpoint；
  - 不做 JIT/ONNX export；
  - 返回 inference policy；
- `policy` 模式下仍复用 probe 原有 camera/render/update/sleep 路径。

边界：

- 不修改训练、奖励、reset、PD、A3/T1 task 配置；
- `zero/random/sine` 原 probe 模式保留。

验证：

- 已执行 `/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/tt_webrtc_probe.py`，通过；
- 待 WebRTC policy 模式复测。

## 2026-06-29 A3 policy visualization standing failure

复测结论：

- `tt_webrtc_probe.py --mode=policy` 可以正常通过 WebRTC 看到画面；
- 使用 `a3_tt_stage4d` + `model_2549.pt` 时，A3 会很快摔倒；
- 终端仍持续输出 hit/success 或 probe step 信息，说明仿真没有崩溃，问题是策略行为本身不稳定。

判断：

- 当前问题已从“可视化黑屏”转为“策略站稳失败”；
- stage4d `model_2549.pt` 不是一个可展示的稳定乒乓球策略；
- 后续需要优先验证 standing/ready-pose 能力，再继续调球路、拍面和回球奖励。

建议对照：

- `mode=zero + freeze_ball`：验证默认姿态/PD；
- `mode=policy + freeze_ball + disable_predictor_update`：验证 policy 无球时是否仍摔；
- `model_2500.pt` vs `model_2549.pt`：验证 stage4d 短训是否损伤站稳；
- 若 policy 无球仍摔，应新增或恢复 A3 standing/ready curriculum，而不是继续单纯加乒乓球 reward。

## 2026-06-29 A3 stage5 ready 下肢站姿

背景：

- WebRTC policy probe 已能稳定显示画面；
- `a3_tt_stage4d + model_2549.pt` 可视化时 A3 很快摔倒；
- 用户观察认为上身/拍面暂时可以，主要问题是下肢初始姿态；
- 当前 A3 pingpong ready 下肢的髋 roll / 踝 roll 符号与 stable standing 相反，视觉上容易表现为双腿偏并拢。

变更：

- 更新 `A3_MIGRATION_PLAN_2026-06-23.md`，记录本轮只改下肢、保护 T1 和已有 A3 workflow 的边界；
- 在 `legged_lab/envs/a3_tt/a3_tt_config.py` 新增：
  - `A3_STAGE5_READY_LOWER_BODY_JOINT_POS`；
  - `A3_STAGE5_READY_ROOT_POS`；
  - stage5 ready 的 base/reset 随机范围；
  - `A3Stage5ReadyEnvCfg`；
  - `A3Stage5ReadyEvalEnvCfg`；
  - `A3Stage5ReadyAgentCfg`；
- 在 `legged_lab/envs/__init__.py` 注册：
  - `a3_tt_stage5_ready`；
  - `a3_tt_stage5_ready_eval`；
- 在 `legged_lab/scripts/a3_hit_outcome_diagnostics.py` 的 A3 task 白名单中加入 stage5 ready 任务。

下肢姿态原则：

- 复用 stage4d 的上肢、球拍、球路、reward 和 agent；
- 髋 roll / 踝 roll 使用 stable standing 方向，并增大外展；
- 膝关节略微加屈；
- 根节点高度从 1.055 下调到 1.025，减少屈膝后脚底悬空风险；
- reset 初期随机范围收窄，避免一开始探索就破坏站姿。

验证：

- 语法检查通过：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/a3_tt/a3_tt_config.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/__init__.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

- 普通 Python 直接 import `legged_lab.envs` 仍会因 Isaac `carb` 未由 SimulationApp 初始化而失败，这是当前工程既有行为；
- 通过 AppLauncher 的 2-step headless smoke test 已通过：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.tt_webrtc_probe \
  --task=a3_tt_stage5_ready \
  --num_envs=1 \
  --mode=zero \
  --freeze_ball \
  --trial_steps=2 \
  --headless
```

下一步：

- 先用 WebRTC 观察 `a3_tt_stage5_ready + mode=zero + freeze_ball` 是否能自然站住；
- 如果仍明显后倒或脚底接触不对，只继续微调下肢，不进入训练；
- 如果 zero-action 能站，再做 500-1000 iter 短训，run name 建议 `a3_stage5_ready_stance_500`。

## 2026-06-29 A3 stage5 ready 1024-env 短训监督

命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5_ready \
  --logger=tensorboard \
  --num_envs=1024 \
  --max_iterations=500 \
  --run_name=a3_stage5_ready_stance_500 \
  --headless \
  --predictor
```

实际运行：

- run 目录：`logs/a3_table_tennis/2026-06-29_12-37-39_a3_stage5_ready_stance_500`
- 手动停止在 151 iter；
- 未达到 250 iter 保存间隔，因此只有 `model_0.pt`，本次主要作为曲线诊断，不作为可视化策略 checkpoint。

关键 TensorBoard 指标，step 151：

- `Train/mean_reward = -18.2262`
- `Train/mean_episode_length = 73.96`
- `Episode_Reward/undesired_contacts = -0.1110`
- `Episode_Reward/termination_penalty = -2.0`
- `Episode_Reward/penalty_hit_low_base_reset = -0.0029`
- `Episode_Reward/reward_contact = 0.0864`
- `Episode_Reward/reward_future_touch_point = 0.3898`
- `Episode_Reward/reward_future_landing_x_progress = 0.0586`
- `Episode_Reward/reward_hit_ball_velocity_net = 0.0415`
- `Episode_Reward/reward_future_pass_net ~= 0`
- `Episode_Reward/reward_table_success = 0`
- `Episode_Reward/reward_actual_opponent_table_target = 0`

判断：

- 下肢 ready stance 明显改善了身体非脚接触问题：
  - `undesired_contacts` 从早期最差约 `-55` 收敛到约 `-0.1`；
  - `penalty_hit_low_base_reset` 始终很小；
  - 因此主要不是摔倒/低 base 导致的失败。
- 任务学习没有打开：
  - `future_pass_net` 和 `table_success` 在 151 iter 仍为 0；
  - 策略更像收敛到“站稳/少动/偶尔接触”的局部解；
  - `termination_penalty=-2` 不是 time-out，由 `TTEnv.check_reset()` 可知更可能来自机器人 x/y 边界 reset，而不是低 base reset。

下一步方向：

- 保留 stage5 ready 下肢姿态；
- 不再继续硬跑同一配置；
- 新增后续 stage 时应优先解决早期任务信号：
  - 放宽或修正 x/y reset 边界与 base 初始位置的关系；
  - 适度提高 right-arm exploration 或 action scale；
  - 恢复/加强早期 hit velocity、pass-net、net-progress shaping；
  - 保持 low-base/undesired-contact 托底，但避免它把策略压成保守站立。

## 2026-06-29 A3 stage5b 早期任务 curriculum

背景：

- `a3_tt_stage5_ready` 151 iter 证明下肢 ready stance 有效压低非脚接触；
- 但任务项未打开，`reward_future_pass_net` 与 `reward_table_success` 仍为 0；
- 失败模式更像“站稳/少动/偶尔接触”的保守局部解。

变更：

- 新增 `A3_STAGE5B_ACTION_SCALE_BY_JOINT`：
  - 只提高右臂和右腕 action scale；
  - 腿部 action scale 保持不变；
- 新增 `A3Stage5bRewardCfg`：
  - `reward_contact` 从 stage4d 的 125 提到 150；
  - 增强并放宽 `reward_hit_ball_velocity_net`；
  - 增强 `reward_hit_net_clearance_progress`；
  - 增强并降低高度要求的 `reward_future_pass_net`；
  - 增强 `reward_post_hit_net_progress`；
  - 暂时降低 own-table 惩罚，避免初期一碰球就回到保守策略；
  - 保留 low-base reset 托底；
- 新增 `A3Stage5bEnvCfg` / `A3Stage5bEvalEnvCfg`：
  - 继承 stage5 ready 下肢姿态；
  - 使用 stage5b action scale；
- 新增 `A3Stage5bAgentCfg`：
  - 初始策略噪声从 0.10 提到 0.16；
  - entropy 与学习率小幅提高；
- 注册任务：
  - `a3_tt_stage5b`
  - `a3_tt_stage5b_eval`
- 更新 `a3_hit_outcome_diagnostics.py` 的 A3 task 白名单。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/a3_tt/a3_tt_config.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/__init__.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.tt_webrtc_probe \
  --task=a3_tt_stage5b \
  --num_envs=1 \
  --mode=zero \
  --freeze_ball \
  --trial_steps=2 \
  --headless
```

结果：

- 语法检查通过；
- `a3_tt_stage5b` Isaac smoke test 通过。

## 2026-06-29 A3 stage5b 250-iter 训练与诊断

训练命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5b \
  --logger=tensorboard \
  --num_envs=1024 \
  --max_iterations=250 \
  --run_name=a3_stage5b_tasksignal_250 \
  --headless \
  --predictor
```

结果：

- run 目录：`logs/a3_table_tennis/2026-06-29_12-50-06_a3_stage5b_tasksignal_250`
- checkpoint：`model_249.pt`
- 训练跑完 250 iter。

最终 TensorBoard 指标，step 249：

- `Train/mean_reward = -12.3302`
- `Train/mean_episode_length = 73.88`
- `Episode_Reward/undesired_contacts = -0.1023`
- `Episode_Reward/termination_penalty = -2.0`
- `Episode_Reward/penalty_hit_low_base_reset = -0.0034`
- `Episode_Reward/reward_contact = 0.1249`
- `Episode_Reward/reward_hit_ball_velocity_net = 0.0999`
- `Episode_Reward/reward_future_pass_net = 9.75e-05`
- `Episode_Reward/reward_post_hit_net_progress = 0.4550`
- `Episode_Reward/reward_table_success = 0`
- `Episode_Reward/reward_actual_opponent_table_target = 0`

对比 `stage5_ready`：

- 身体稳定性保持：`undesired_contacts` 和 `penalty_hit_low_base_reset` 与 stage5 ready 接近；
- 任务中间信号明显更强：
  - `reward_contact` 从约 `0.086` 提到约 `0.125`；
  - `reward_hit_ball_velocity_net` 从约 `0.042` 提到约 `0.100`；
  - `reward_post_hit_net_progress` 从约 `0.059` 提到约 `0.455`；
- 但 `table_success` 仍为 0。

checkpoint 诊断命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_hit_outcome_diagnostics \
  --task=a3_tt_stage5b \
  --num_envs=64 \
  --max_steps=160 \
  --print_interval=40 \
  --load_run 2026-06-29_12-50-06_a3_stage5b_tasksignal_250 \
  --checkpoint model_249.pt \
  --predictor \
  --random_ball \
  --headless
```

诊断结果：

- `first_hit = 63/64 (0.984)`
- `hit_vx_positive = 39/64 (0.609)`
- `hit_vz_positive = 53/64 (0.828)`
- `actual_crossed_net = 63/64 (0.984)`
- `actual_net_clear = 0/64 (0.000)`
- `opponent_table_after_hit = 0/64 (0.000)`
- `own_table_after_hit = 63/64 (0.984)`
- `reset_seen = 0/64 (0.000)`
- `hit_z mean = 0.9788`
- `hit_vx mean = 0.1627`
- `hit_vz mean = 0.8092`
- `actual_z_at_net mean = 0.8184`

判断：

- stage5b 已经把问题从“站不稳/碰不到球”推进到“能碰到球，但回球轨迹太低、太弱”；
- 机器人没有在诊断窗口内 reset，说明此 checkpoint 的短时稳定性可接受；
- 几乎所有球都发生 `own_table_after_hit`，且 `actual_net_clear=0`，说明当前击球后的球仍落在己方或低于网；
- 下一步应集中处理击球方向：
  - 拍面可能仍偏关闭或偏向下；
  - 需要更强的向前 `vx` 和向上/过网 `z_at_net` shaping；
  - 可考虑新增 stage5c：降低 contact/own-table 的相对吸引，强化 net-clear 和 positive-vx/vz，并测试右腕 pitch/roll 小范围打开拍面。

## 2026-06-29 A3 stage5c 软课程任务

背景：

- 用户明确 stage/curriculum 不应理解为孤立单项训练；
- 正确方向是在完整乒乓球任务中调整 reward 主次：
  - 初期稳定性权重大；
  - 稳定性改善后逐步提高向球路/击球点移动；
  - 随后逐步提高触球、向前向上击球、过网、落点 reward。

修改：

- 更新 `A3_MIGRATION_PLAN_2026-06-23.md`，新增 stage5c 软课程任务计划；
- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 引入 `CurriculumTermCfg as CurrTerm` 和 `CurriculumCfg`；
  - 新增 `A3Stage5cRewardCfg`：
    - 初始更重 `termination_penalty`、`undesired_contacts`、脚姿态、脚过近、低 base 等稳定性惩罚；
    - 保留完整乒乓球任务信号；
    - 初始降低高阶过网/落点 reward 和 own-table 惩罚，避免早期策略被稀疏成功目标压回保守解；
  - 新增 `A3Stage5cCurriculumCfg`：
    - 4000-24000 action steps：稳定惩罚逐步回到正常强度，靠近球路/触球 reward 增强；
    - 12000-60000 action steps：击球速度、过网高度、post-hit progress、table success 等 reward 增强；
    - 24000-60000 action steps：own-table 惩罚逐步恢复；
  - 新增 `A3Stage5cEnvCfg` / `A3Stage5cEvalEnvCfg`；
  - 新增 `A3Stage5cAgentCfg`，降低初始策略噪声和 entropy，减少早期因随机动作导致的摔倒；
- `legged_lab/envs/__init__.py`
  - 注册 `a3_tt_stage5c`；
  - 注册 `a3_tt_stage5c_eval`；
- `legged_lab/scripts/a3_hit_outcome_diagnostics.py`
  - 将 `stage5c` 加入 A3 诊断脚本白名单。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/a3_tt/a3_tt_config.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/__init__.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

结果：

- 全部通过。

Isaac smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.tt_webrtc_probe \
  --task=a3_tt_stage5c \
  --num_envs=1 \
  --mode=zero \
  --freeze_ball \
  --trial_steps=2 \
  --headless
```

结果：

- 通过，`a3_tt_stage5c` 能正常实例化并执行最小环境循环；
- 终端仍有 Isaac/Omniverse 常见图形和 CUDA warning，但进程 exit code 为 0。

## 2026-06-29 A3 stage5c 长训前击球质量优化

背景：

- 用户明确 T1 是基线，不是 A3 的上限；
- T1 曲线显示任务真正打开在 1500-2500 iter 以后，最佳区间在 4000-5500 iter；
- A3 stage5b 诊断显示瓶颈已从“碰不到球”转为“击球后轨迹太低、太弱、own-table-after-hit 多”；
- 因此长训前应提高 A3 stage5c 对 net-clear / hit-vz / z-at-net 的引导，而不是只提高 contact。

修改：

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - `A3Stage5cRewardCfg.reward_hit_ball_velocity_net`
    - 初始 weight `60 -> 75`；
    - `vx_target 2.4 -> 2.6`；
    - `vz_target 1.2 -> 1.35`；
    - `z_target 1.02 -> 1.08`；
    - 权重从偏 `vx` 调为更重 `vz/z_at_net`；
  - `reward_hit_net_clearance_progress`
    - 初始 weight `35 -> 50`；
    - `target_z 1.00 -> 1.07`；
    - `min_z 0.76 -> 0.78`；
  - `reward_future_pass_net`
    - 初始 weight `50 -> 60`；
    - target 从 `table + 0.25` 调到 `table + 0.30`；
  - `reward_post_hit_net_progress`
    - 初始 weight `20 -> 35`；
    - `vz_target 1.1 -> 1.3`；
    - `net_z_target 1.00 -> 1.08`；
    - 增加 `vz_weight` 和 `z_weight`；
  - `reward_table_success`
    - 初始 weight `30 -> 35`；
  - `reward_actual_opponent_table_target`
    - 初始 weight `20 -> 25`；
  - curriculum 后期目标：
    - `reward_hit_ball_velocity_net 280 -> 310`；
    - `reward_hit_net_clearance_progress 170 -> 220`；
    - `reward_post_hit_net_progress 130 -> 170`；
    - `reward_future_pass_net 280 -> 320`；
    - `reward_table_success 220 -> 260`；
    - `reward_actual_opponent_table_target 100 -> 140`；
    - own-table 后期惩罚稍恢复到 `-25/-30`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/a3_tt/a3_tt_config.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/envs/__init__.py
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.tt_webrtc_probe \
  --task=a3_tt_stage5c \
  --num_envs=1 \
  --mode=zero \
  --freeze_ball \
  --trial_steps=2 \
  --headless
```

结果：

- 语法检查通过；
- `a3_tt_stage5c` 最小 Isaac smoke test 通过；
- 可以进入长训。

训练启动：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5c \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5c_netclear_long_10000 \
  --headless \
  --predictor
```

运行方式：

- tmux session：`a3_stage5c_long_20260629`
- run 目录：`logs/a3_table_tennis/2026-06-29_13-33-06_a3_stage5c_netclear_long_10000`
- console log：`logs/a3_table_tennis/a3_stage5c_netclear_long_10000.console.log`
- TensorBoard：`http://127.0.0.1:16006`

初始观察：

- 训练已进入 PPO loop；
- 速度约 `19k-20k steps/s`；
- ETA 约 13 小时；
- 前几 iter `undesired_contacts` 较大，但到 iter 35 已快速降到约 `-0.94`；
- `reward_future_touch_point`、`reward_future_dis_ee`、`reward_contact` 均为非零；
- 早期 `termination_penalty` 仍偏大，episode length 约 76，需要继续观察到 250/500/1000 iter。

停止与阶段诊断：

- 用户判断当前版本可以停止，原因是：
  - 球路过窄，hit rate 过于乐观；
  - 1000+ iter 后机器人仍没有真正学会稳定处理多个球；
  - hit rate 上升过快但 success 长时间为 0，说明训练目标可能过早偏向触球；
- 已停止 tmux session：`a3_stage5c_long_20260629`；
- 最后训练到约 iter `1257`；
- 已保存 checkpoint：
  - `model_0.pt`
  - `model_250.pt`
  - `model_500.pt`
  - `model_750.pt`
  - `model_1000.pt`
  - `model_1250.pt`

关键配置问题：

- A3 当前训练球路明显窄于 T1：
  - A3 ball speed y `(-0.1, 0.02)`，T1 为 `(-0.8, 0.4)`；
  - A3 ball pos y `(-0.03, 0.03)`，T1 为 `(-0.1, 0.1)`；
  - A3 contact threshold `0.07`，T1 为 `0.05`；
- A3 reset 也明显更窄：
  - base x `(-0.265, -0.255)`；
  - base y `(0.34, 0.36)`；
  - yaw `(-0.005, 0.005)`；
  - locomotion joint reset scale `(0.995, 1.005)`；
- 因此高 hit rate 不能说明策略已学会乒乓球任务。

曲线结论：

- `Train/TT_hit_rate`：
  - 250-500 iter 均值约 `0.915`；
  - 500-750 iter 均值约 `0.895`；
  - 750-1000 iter 均值约 `0.889`；
  - 1000-1260 iter 均值约 `0.890`；
- `Train/TT_success_rate` 全程基本为 `0`；
- `Train/mean_episode_length` 在 250 iter 后长期约 `73-74`，没有继续拉长；
- `Episode_Reward/termination_penalty` 到 1000 iter 后仍约 `-2.0`；
- `Episode_Reward/undesired_contacts` 从早期大值降到约 `-0.1`，说明非脚接触问题改善；
- `Episode_Reward/penalty_hit_low_base_reset` 降到约 `-0.002`，说明击球时低 base reset 问题改善；
- 但机器人还没有真正学会长时间站稳和处理多个球。

关于 hit-ball 相关 reward：

- `reward_contact` 持续升高并在约 `0.12` 附近平台；
- `reward_hit_ball_velocity_net` 没有动态变少：
  - 0-250 iter 均值约 `0.020`；
  - 250-500 iter 均值约 `0.029`；
  - 500-750 iter 均值约 `0.037`；
  - 750-1000 iter 均值约 `0.050`；
  - 1000-1260 iter 均值约 `0.059`；
- `reward_post_hit_net_progress` 也升到约 `0.32`；
- 但 `reward_hit_net_clearance_progress`、`reward_future_pass_net` 仍接近 0；
- `reward_table_success` 始终为 0。

判断：

- 这一版不是物理/训练崩溃；
- 问题是训练入口过于容易触球，导致策略过早追求 hit/contact；
- 站稳能力没有先成为足够硬的前提；
- 下一版应先降低 hit 虚高：
  - 缩小 `contact_threshold`；
  - 扩大球路但分 curriculum 增加；
  - 降低早期 `reward_contact`/`reward_future_touch_point` 的支配性；
  - 更重视 episode length、termination、稳定站姿；
  - 只有站稳指标达到门槛后再逐步提高触球和过网奖励。

## 2026-06-29 A3 stage5d：严格触球判定与 ball curriculum

用户要求本轮先只处理球路和 hit/contact 判定问题，站稳 reward/PD 专项后续单独对比。因此本次没有修改 T1、没有覆盖 `a3_tt_stage5c`，新增 A3-only stage：

```text
a3_tt_stage5d
a3_tt_stage5d_eval
```

设计原因：

- stage5c 的 `hit_rate` 很高但 `success_rate` 长期为 0，且 `mean_episode_length` 没有继续增长；
- stage5c 训练球路过窄：
  - `ball_speed_y_range=(-0.10, 0.02)`
  - `ball_pos_y_range=(-0.03, 0.03)`
  - `contact_threshold=0.07`
- 这会让策略在固定窄球路上刷触球，导致 hit/contact 奖励掩盖真正的击球质量。

修改：

- `legged_lab/mdp/curriculums.py`
  - 新增 `modify_ball_ranges_piecewise_linear()`；
  - 支持按照 action step 分段线性修改：
    - `ball_speed_x_range`
    - `ball_speed_y_range`
    - `ball_speed_z_range`
    - `ball_pos_y_range`
  - 默认不启用，只有显式挂到 curriculum 的 task 才会改变球路。

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 `A3_STAGE5D_CONTACT_THRESHOLD = 0.05`；
  - 新增 `A3_STAGE5D_BALL_START_RANGES`：
    - `ball_speed_x_range=(-5.15, -4.85)`
    - `ball_speed_y_range=(-0.12, 0.04)`
    - `ball_speed_z_range=(1.42, 1.62)`
    - `ball_pos_y_range=(-0.04, 0.04)`
  - 新增 `A3_STAGE5D_BALL_CURRICULUM_PHASES`：
    - 0-12000 step：轻微放宽，保持 A3 初始击球中心附近；
    - 12000-36000 step：扩大横向球路和起始 y；
    - 36000-84000 step：增加横向速度和高度变化；
    - 84000-144000 step：逐步接近 T1 随机球路范围；
  - 新增 `A3Stage5dRewardCfg`：
    - `reward_contact` 从 stage5c 的 `90` 降到 `55`；
    - `reward_future_touch_point` 从 `10` 降到 `8`；
    - `reward_future_dis_ee` 从 `8` 降到 `7`；
    - 保留 stage5c 的击球后速度、过网高度、落点质量奖励；
  - 新增 `A3Stage5dCurriculumCfg`：
    - 启用 ball range curriculum；
    - `reward_contact` 后续只升到 `90`，不再像 stage5c 一样升到 `150`；
  - 新增 `A3Stage5dEnvCfg` 和 `A3Stage5dEvalEnvCfg`。

- `legged_lab/envs/__init__.py`
  - 注册：
    - `a3_tt_stage5d`
    - `a3_tt_stage5d_eval`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/curriculums.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py
```

结果：

- 语法检查通过；
- 直接在普通 shell 中 `import legged_lab.envs` 会因为缺少 Isaac Sim runtime 里的 `carb` 模块失败，这是预期的 Isaac 环境限制，不代表训练脚本不可用。

建议训练命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5d \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5d_ballcurr_stricthit_10000 \
  --headless \
  --predictor
```

监督重点：

- `Train/TT_hit_rate` 不再单独作为好指标；
- 更重要的是：
  - `Train/mean_episode_length`
  - `Train/TT_success_rate`
  - `Episode_Reward/reward_future_pass_net`
  - `Episode_Reward/reward_hit_net_clearance_progress`
  - `Episode_Reward/reward_table_success`
  - `Episode_Reward/reward_contact`
- 如果 hit rate 在球路放宽后明显下降但 success/net-clear 开始上升，说明 curriculum 正在打破“刷窄球路 hit”的局部最优；
- 如果 hit rate 仍高但 success 仍为 0，则下一步应转向站稳 gate、击球质量 gate 和 A3 PD/稳定性专项。

训练启动记录：

- 首次 4096 env 启动在 Isaac 初始化阶段长时间无 run 目录，已停止；
- `num_envs=64, max_iterations=2` smoke test 成功创建 run 目录：
  - `logs/a3_table_tennis/2026-06-29_15-47-40_a3_stage5d_smoke_64_2`
- 随后重新启动 4096 env 长训，已进入 PPO loop：
  - tmux session：`a3_stage5d_ballcurr_20260629`
  - run 目录：`logs/a3_table_tennis/2026-06-29_15-48-30_a3_stage5d_ballcurr_stricthit_10000`
  - console log：`logs/a3_table_tennis/a3_stage5d_ballcurr_stricthit_10000_retry.console.log`
  - iter 19 时速度约 `19k steps/s`，ETA 约 `13h 53m`
  - 早期观察：`mean_episode_length` 约 `100`，`reward_contact` 非零但明显低于 stage5c，`reward_future_pass_net` 和 `reward_table_success` 仍为 0，需要继续观察 250/500/1000 iter。

## 2026-06-29 A3 stage5e：soft stability gate 与 dense standing reward

背景：

- stage5d 到约 1359 iter 时：
  - `TT_hit_rate` 随 ball curriculum 从约 `0.64` 降到约 `0.36`；
  - `TT_success_rate` 仍为 0；
  - `mean_episode_length` 仍约 `73.4`；
  - `reward_future_pass_net` 只有极小非零；
  - `reward_table_success` 仍为 0；
- 结论：stage5d 成功压低了虚高 hit rate，但没有自然形成“稳定站立下完成乒乓球任务”的链路。

设计目标：

- 不做硬门控，避免 `不站稳 -> 所有任务奖励为 0` 造成奖励稀疏；
- 用 soft stability gate 让不稳定状态仍保留少量探索梯度，但高价值任务奖励必须稳定才能拿满；
- 新增 dense standing reward，让站稳每一步都有正向学习信号；
- 新增 unstable-hit penalty，让“倒着碰球”不再划算。

新增/修改：

- `legged_lab/mdp/rewards.py`
  - 新增 `_a3_stability_scores()`：
    - `height_score`
    - `upright_score`
    - `support_score`
    - `low_velocity_score`
    - `contact_clean_score`
    - `feet_width_score`
  - 新增 `_a3_stability_raw_score()`：
    - 使用加权几何均值；
    - 默认权重：
      - height `0.28`
      - upright `0.24`
      - support `0.20`
      - velocity `0.14`
      - clean contact `0.09`
      - feet width `0.05`
  - 新增 `a3_stability_gate()`：
    - `stability_gate = gate_floor + (1 - gate_floor) * raw_score`
  - 新增 `reward_standing_stability()`：
    - 使用同一组稳定分数的加权算术均值；
  - 新增 `penalty_unstable_hit()`：
    - `ball_contact_rew * (1 - raw_score)`；
  - 新增 gated reward wrapper：
    - `reward_contact_stability_gated`
    - `reward_future_touch_point_target_stability_gated`
    - `reward_future_ee_target_stability_gated`
    - `reward_future_landing_x_progress_stability_gated`
    - `reward_hit_ball_velocity_net_target_stability_gated`
    - `reward_hit_net_clearance_progress_stability_gated`
    - `reward_post_hit_net_progress_stability_gated`
    - `reward_future_pass_net_stability_gated`
    - `reward_table_success_stability_gated`
    - `reward_opponent_table_after_paddle_hit_target_stability_gated`

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 `_a3_stage5e_stability_params()`；
  - 初始参数：
    - `min_base_z=0.72`
    - `max_base_z=1.18`
    - `height_std=0.18`
    - `upright_std=0.35`
    - `lin_vel_std=1.20`
    - `ang_vel_std=2.00`
    - `contact_force_threshold=1.0`
    - `force_balance_std=0.65`
    - `bad_contact_threshold=1.0`
    - `bad_contact_std=1.0`
    - `target_feet_width=0.42`
    - `feet_width_std=0.22`
  - 新增 `A3Stage5eRewardCfg`：
    - `reward_standing_stability` weight `12.0`；
    - `penalty_unstable_hit` weight `-60.0`；
    - 接近球/触球类 reward gate floor `0.30`；
    - 击球质量/过网/轨迹类 reward gate floor `0.15`；
    - 成功/落点类 reward gate floor `0.10`；
  - 新增 `A3Stage5eEnvCfg` 和 `A3Stage5eEvalEnvCfg`；
  - 保留 stage5d 的 ball curriculum 和 `contact_threshold=0.05`。

- `legged_lab/envs/__init__.py`
  - 注册：
    - `a3_tt_stage5e`
    - `a3_tt_stage5e_eval`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py
```

结果：

- 语法检查通过；
- 当前 `a3_tt_stage5d` 仍在 GPU1 上长训，因此没有立刻启动 stage5e Isaac smoke test，避免抢占训练资源。

建议 stage5e 训练命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5e \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5e_stability_gate_10000 \
  --headless \
  --predictor
```

如果 stage5d 继续跑，stage5e 可等 stage5d 结束后再跑；如需并行，建议先确认 GPU0 没有 T1/WebRTC 可视化任务。

### 2026-06-29 18:10 stage5e 实现修正与 smoke test

本次在真正启动 stage5e smoke test 后发现两个 IsaacLab 参数/解析边界，已修正。

公式确认：

```text
gate_raw = exp(sum_i normalized_weight_i * log(clamp(score_i, min=1e-3, max=1.0)))
stability_gate = gate_floor + (1 - gate_floor) * gate_raw

reward_standing_stability =
  weighted_arithmetic_mean(height, upright, support, velocity, clean_contact, feet_width)

penalty_unstable_hit = ball_contact_rew * (1 - gate_raw)
```

参数确认：

```text
height/upright/support/velocity/clean/feet_width weights = 0.28/0.24/0.20/0.14/0.09/0.05
min_base_z = 0.72
max_base_z = 1.18
height_std = 0.18
upright_std = 0.35
lin_vel_std = 1.20
ang_vel_std = 2.00
contact_force_threshold = 1.0
force_balance_std = 0.65
bad_contact_threshold = 1.0
bad_contact_std = 1.0
target_feet_width = 0.42
feet_width_std = 0.22
gate_floor = 0.30 for approach/contact, 0.15 for hit quality/net/traj, 0.10 for success/landing
```

代码修正：

- `legged_lab/mdp/rewards.py`
  - 公开 reward wrapper 不再使用 `**score_kwargs` / `**kwargs` 作为函数签名；
  - 改为显式 `score_kwargs: dict | None = None`，避免 RewardManager 将其识别为未提供的强制参数；
  - `reward_post_hit_net_progress_stability_gated()` 改为 `reward_kwargs` 与 `score_kwargs` 分离；
  - `_resolve_scene_entity_cfg()` 只在 `body_ids is None` 或 `body_ids == slice(None)` 时解析，避免对已解析的 regex cfg 重复 resolve；
  - `feet_width_score` 直接根据取出的 body position 数量判断，兼容 `body_ids` 为 list 或 slice。
  - 删除不再使用的 `_pack_a3_stability_kwargs()`，避免后续误用旧的扁平参数传法。

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 `_a3_stage5e_score_kwargs()`；
  - `_a3_stage5e_stability_params()` 现在返回 `{"score_kwargs": ...}`；
  - 新增 `_a3_stage5e_gated_params()` 和 `_a3_stage5e_post_hit_params()`；
  - stage5e gated reward 的普通 reward 参数留在顶层，稳定性参数统一放入 `score_kwargs`，post-hit 参数放入 `reward_kwargs`。

验证记录：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py
```

结果：通过。

smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5e \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1 \
  --run_name=a3_stage5e_smoke_64_1_retry3 \
  --headless \
  --predictor
```

结果：退出码 0，生成：

```text
logs/a3_table_tennis/2026-06-29_18-10-23_a3_stage5e_smoke_64_1_retry3/events.out.tfevents.1782727823.FdseRobot-02.2415195.0
logs/a3_table_tennis/2026-06-29_18-10-23_a3_stage5e_smoke_64_1_retry3/model_0.pt
```

中间失败与处理：

- `a3_stage5e_smoke_64_1_retry`：`body_ids` 为 `slice` 时不能 `len()`，已改为按 `feet_pos.shape[1]` 判断；
- `a3_stage5e_smoke_64_1_retry2`：已解析过的 regex `SceneEntityCfg` 重复 resolve 触发一致性检查，已改为只对默认/未解析 body ids 解析；
- 最终 `a3_stage5e_smoke_64_1_retry3` 通过。

边界：

- 本次只新增/修正 `a3_tt_stage5e` 相关新分支；
- T1 配置、T1 任务注册、stage5d 长训任务未修改。

## 2026-06-30 A3 stage5f：能力触发 ball curriculum 与解除早期稳定门控

背景：

- stage5d/stage5e 训练完成后，WebRTC 可视化显示机器人仍然站不稳、身体动作僵硬、没有形成有效移动和击球策略；
- stage5d 的 `TT_hit_rate` 从早期高位掉到约 `0.003`；
- stage5e 最终 `TT_hit_rate` 约 `0.038`，但 `TT_success_rate` 仍为 0；
- 两版最终 `mean_episode_length` 约 `75`，说明还没有稳定站住；
- 结论：按 step 推进 ball curriculum 会在策略能力不足时自动加难；stage5e 过早 gate 早期任务奖励，也会削弱靠近球路和触球学习。

计划文档：

- 已在 `A3_MIGRATION_PLAN_2026-06-23.md` 新增 `2026-06-30 A3 stage5f：能力触发 curriculum 与解除早期门控计划`。

新增任务：

```text
a3_tt_stage5f
a3_tt_stage5f_eval
```

代码改动：

- `legged_lab/mdp/curriculums.py`
  - 新增 `modify_ball_ranges_by_ability()`；
  - 维护窗口指标：
    - `stage`
    - `steps`
    - `serves`
    - `hit_rate`
    - `success_rate`
    - `mean_episode_length`
    - `reset_rate`
  - 只有窗口内能力达标才推进球路；
  - 如果升阶段后能力过低，则回退一档；
  - 每次 compute 会把当前 stage 的 ball range 写回 `env.cfg.ball`。

- `legged_lab/envs/a3_tt/a3_tt_config.py`
  - 新增 `A3_STAGE5F_ACTION_SCALE_BY_JOINT`，只在 stage5f 小幅放大下肢 action scale；
  - 新增 `A3_STAGE5F_BALL_ABILITY_PHASES`；
  - 新增 `_a3_stage5f_score_kwargs()`：
    - `height_weight=0.32`
    - `upright_weight=0.30`
    - `support_weight=0.23`
    - `velocity_weight=0.00`
    - `clean_weight=0.10`
    - `feet_width_weight=0.05`
    - 目的：减少对“低速度僵住”的奖励；
  - 新增 `A3Stage5fRewardCfg`：
    - 早期探索项不再 gate：
      - `reward_future_touch_point`
      - `reward_future_dis_ee`
      - `reward_contact`
      - `reward_future_landing_x_progress`
    - 高阶回球质量项继续 soft gate：
      - `reward_hit_ball_velocity_net`
      - `reward_hit_net_clearance_progress`
      - `reward_future_pass_net`
      - `reward_post_hit_net_progress`
      - `reward_table_success`
      - `reward_actual_opponent_table_target`
    - `penalty_unstable_hit` 从 stage5e 的强惩罚降低为弱约束；
  - 新增 `A3Stage5fCurriculumCfg`，使用能力触发 ball curriculum；
  - 新增 `A3Stage5fEnvCfg` / `A3Stage5fEvalEnvCfg`；
  - 新增 `A3Stage5fAgentCfg`。

- `legged_lab/envs/__init__.py`
  - 注册：
    - `a3_tt_stage5f`
    - `a3_tt_stage5f_eval`

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/curriculums.py \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py
```

结果：通过。

smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5f \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1 \
  --run_name=a3_stage5f_abilitycurr_smoke_64_1 \
  --headless \
  --predictor
```

结果：退出码 0，生成：

```text
logs/a3_table_tennis/2026-06-30_17-41-57_a3_stage5f_abilitycurr_smoke_64_1/events.out.tfevents.1782812517.FdseRobot-02.3028972.0
logs/a3_table_tennis/2026-06-30_17-41-57_a3_stage5f_abilitycurr_smoke_64_1/model_0.pt
```

smoke event 已确认：

```text
Episode_Reward/reward_standing_stability
Train/TT_hit_rate
Train/TT_success_rate
Train/mean_episode_length
```

注意：

- 1 iteration smoke 太短，还不会产生完整的 ability curriculum 窗口日志；
- 正式训练时应重点看 `Curriculum/ball_range_curriculum/stage`、`hit_rate`、`mean_episode_length`、`reset_rate`，确认球路不是按时间盲目加难。

建议 stage5f 训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5f \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5f_abilitycurr_10000 \
  --headless \
  --predictor
```

WebRTC 可视化训练后 checkpoint：

```bash
PUBLIC_IP=10.176.65.212 \
CUDA_VISIBLE_DEVICES=0 \
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.tt_webrtc_probe \
  --task=a3_tt_stage5f_eval \
  --num_envs=1 \
  --mode=policy \
  --load_run <RUN_DIR> \
  --checkpoint model_9999.pt \
  --predictor \
  --trial_steps=100000 \
  --visualize_sleep=0.03 \
  --livestream 2
```

## 2026-06-30 A3 stage5g：精确 FK 几何目标替换 T1 偏移

目标：只修正 A3 的身体目标几何，不加入 HITTER 的挥拍/恢复动作先验。

计算依据：

- 使用 `zxl-pace` 环境和 Isaac FK；
- task：`a3_tt_stage5f_eval`；
- 训练站位：`base_x=-0.26, base_y=0.35`；
- 固定球路：`ball_vx=-5.0, ball_vy=-0.04, ball_vz=1.5`；
- 得到当前 A3 ready 姿态下 `paddle_touch_point - root_pos ≈ (0.224, -0.397, 0.060)`。

修改：

- `RobotCfg` 新增：
  - `future_paddle_x_offset`，默认 `0.10`，保持 T1 行为；
  - `future_invalid_robot_xy`，默认 `(-1.80, 0.30)`，保持 T1 fallback；
- `TTEnv` 的 actor 相对身体目标和 `robot_future_pos` 改用配置项，不再写死 `[-0.1, +0.6]`；
- 新增 `a3_tt_stage5g` / `a3_tt_stage5g_eval`：
  - `future_paddle_x_offset = 0.224`
  - `future_paddle_y_offset = -0.397`
  - `future_invalid_robot_xy = (-1.86, 0.35)`

对照诊断：

- stage5f：`robot_future_pos=(-1.7000, 0.5766, 0.9000)`，距离当前 root xy 约 `0.277m`；
- stage5g：`robot_future_pos=(-1.8240, 0.3736, 0.9000)`，距离当前 root xy 约 `0.043m`；
- `paddle_to_future_dist` 约 `0.049m`。

边界：

- 不修改 T1 默认行为；
- 不修改 stage5f，保留失败/对照版本；
- 暂不加入挥拍和恢复动作先验。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/base/tt_config.py \
  legged_lab/envs/base/tt_env.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py \
  legged_lab/scripts/tt_paddle_pose_diagnostics.py
```

结果：通过。

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5g \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1 \
  --run_name=a3_stage5g_fkgeometry_smoke_64_1 \
  --headless \
  --predictor
```

结果：退出码 0，生成：

```text
logs/a3_table_tennis/2026-06-30_23-38-20_a3_stage5g_fkgeometry_smoke_64_1/events.out.tfevents.1782833900.FdseRobot-02.3202695.0
logs/a3_table_tennis/2026-06-30_23-38-20_a3_stage5g_fkgeometry_smoke_64_1/model_0.pt
```

## 2026-07-01 A3 stage5g_fixedball：关闭 ball curriculum 的几何验证

目标：隔离验证 A3 FK 几何修正，不让 ball curriculum 变化影响判断。

修改：

- 新增 `a3_tt_stage5g_fixedball` / `a3_tt_stage5g_fixedball_eval`；
- 继承 stage5g 的 A3 FK 几何 offset；
- `curriculum = CurriculumCfg()`，关闭 ability-based ball curriculum；
- 球路固定为 stage5f phase0 初始范围。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py
```

结果：通过。

smoke：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5g_fixedball \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1 \
  --run_name=a3_stage5g_fixedball_smoke_64_1 \
  --headless \
  --predictor
```

结果：退出码 0，生成：

```text
logs/a3_table_tennis/2026-07-01_00-19-29_a3_stage5g_fixedball_smoke_64_1
```

训练：

- 4096 env 在当前 Isaac/GPU 状态下初始化阶段出现 `malloc(): invalid size (unsorted)`，未进入训练；
- 已改用 1024 env 启动几何验证短训：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5g_fixedball \
  --logger=tensorboard \
  --num_envs=1024 \
  --max_iterations=1000 \
  --run_name=a3_stage5g_fixedball_geomonly_1000_1024 \
  --headless \
  --predictor
```

当前状态：tmux session `a3_stage5g_fixedball_1024` 正在运行，GPU0 占用约 4.1GB，GPU1 空闲。

## 2026-07-01 A3 stage5g_wide：A3 适配宽球路长训入口

目的：结束 `stage5g_fixedball` 的窄球路几何验证后，新增一个宽球路对照入口，避免固定窄球路把 hit rate 统计推高。

修改：

- 新增 `A3_STAGE5G_WIDE_BALL_RANGES`：
  - `ball_speed_x_range=(-5.90, -4.95)`
  - `ball_speed_y_range=(-0.42, 0.18)`
  - `ball_speed_z_range=(1.42, 1.85)`
  - `ball_pos_y_range=(-0.09, 0.09)`
- 新增 `A3_STAGE5G_WIDE_CONTACT_THRESHOLD=0.04`，比 stage5d/f 的 `0.05` 更严格；
- 新增 `a3_tt_stage5g_wide` / `a3_tt_stage5g_wide_eval`；
- 继续继承 stage5g 的 A3 FK 几何 offset；
- 关闭 ball curriculum，固定使用 A3 适配宽球路；
- 不修改 T1、不覆盖 `a3_tt_stage5g` 和 `a3_tt_stage5g_fixedball`。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py
```

结果：通过。

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5g_wide \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1 \
  --run_name=a3_stage5g_wide_smoke_64_1 \
  --headless \
  --predictor
```

结果：通过。

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5g_wide \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=1 \
  --run_name=a3_stage5g_wide_smoke_4096_1 \
  --headless \
  --predictor
```

结果：通过。因此当前机器状态下，该任务可以使用 `num_envs=4096` 长训。

长训已启动：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5g_wide \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5g_wide_a3adapted_10000_4096 \
  --headless \
  --predictor
```

tmux session：`a3_stage5g_wide_4096`

run dir：

```text
logs/a3_table_tennis/2026-07-01_00-53-29_a3_stage5g_wide_a3adapted_10000_4096
```

早期监督：

- step 0-1：`reward_contact` 近 0，未出现 fixedball 式 hit rate 虚高；
- step 8-9：`undesired_contacts` 一度到约 `-90`，说明宽球路探索下身体接触问题明显；
- step 19-23：`undesired_contacts` 回到约 `-25` 到 `-40`，`mean_episode_length` 约 `115-126`，训练未发散；
- `TT_hit_rate` / `TT_success_rate` 当前仍为 0，任务侧尚未打开，需要继续看到 250/500 iter 后再判断宽球路是否过难。

## 2026-07-01 A3 stage5h_hitquality：拍面法向、击球速度、击球窗口对照实验

目的：在 GPU0 的 `a3_tt_stage5g_wide` 持续训练不停止的前提下，新增 GPU1 对照实验，验证 A3 几何修正后的击球质量 shaping 是否能改善接触少、出球质量弱的问题。

与 GPU0 `stage5g_wide` 相同：

- A3 FK 几何 offset：
  - `future_paddle_x_offset=0.224`
  - `future_paddle_y_offset=-0.397`
- A3 适配宽球路：
  - `ball_speed_x_range=(-5.90, -4.95)`
  - `ball_speed_y_range=(-0.42, 0.18)`
  - `ball_speed_z_range=(1.42, 1.85)`
  - `ball_pos_y_range=(-0.09, 0.09)`
- `contact_threshold=0.04`
- 关闭 ball curriculum。

新增差异：

- 新增 `a3_tt_stage5h_hitquality` / `a3_tt_stage5h_hitquality_eval`；
- 根据诊断脚本确认 A3 球拍正面法向为本地 `-Z`，配置 `A3_STAGE5H_PADDLE_NORMAL_AXIS=(0,0,-1)`；
- 新增 reward：
  - `reward_strike_window_touch_point_stability_gated`
  - `reward_paddle_normal_alignment_stability_gated`
  - `reward_paddle_swing_velocity_target_stability_gated`
- 加强并收紧击球后质量 reward：
  - `reward_hit_ball_velocity_net`
  - `reward_hit_net_clearance_progress`
  - `reward_future_pass_net`
  - `reward_post_hit_net_progress`

预期对照：

- 如果 `stage5h_hitquality` 的 contact 仍低但稳定性更差，说明击球质量 shaping 过早或过强；
- 如果 contact 稍慢上升但 post-hit/net/table 指标好于 `stage5g_wide`，说明拍面/挥拍/窗口约束有效；
- 如果 hit rate 明显高但 success 仍为 0，需要继续收紧出球落点或拍面倾角目标。

验证：

```bash
/home/zxl/miniconda3/envs/zxl-pace/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/a3_tt/a3_tt_config.py \
  legged_lab/envs/__init__.py
```

结果：通过。

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5h_hitquality \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1 \
  --run_name=a3_stage5h_hitquality_smoke_64_1 \
  --headless \
  --predictor
```

结果：通过，生成：

```text
logs/a3_table_tennis/2026-07-01_01-06-30_a3_stage5h_hitquality_smoke_64_1
```

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5h_hitquality \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=1 \
  --run_name=a3_stage5h_hitquality_smoke_4096_1 \
  --headless \
  --predictor
```

结果：通过，生成：

```text
logs/a3_table_tennis/2026-07-01_01-07-08_a3_stage5h_hitquality_smoke_4096_1
```

长训已启动：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5h_hitquality \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5h_hitquality_a3adapted_10000_4096 \
  --headless \
  --predictor
```

tmux session：`a3_stage5h_hitquality_4096`

run dir：

```text
logs/a3_table_tennis/2026-07-01_01-08-13_a3_stage5h_hitquality_a3adapted_10000_4096
```

对照日志：

```text
A3_TRAINING_COMPARISON_2026-07-01.md
```

早期监督：

- 已进入 PPO；
- 前几轮新增 reward 均正常出现：
  - `reward_strike_window_touch_point`
  - `reward_paddle_normal_alignment`
  - `reward_paddle_swing_velocity`
- GPU0 的 `a3_tt_stage5g_wide` 仍在运行，没有停止。
