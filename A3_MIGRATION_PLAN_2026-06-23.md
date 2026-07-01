# PACE-ICRA2026 A3 机器人迁移任务规划（2026-06-23）

## 目标

在不破坏现有 T1 复现流程的前提下，把当前 PACE/LeggedLab 乒乓球任务框架迁移到 A3T2.5 机器人上，新增 A3 专用资产、任务配置和必要的通用配置钩子。

当前必须保持可用的基线命令：

```bash
CUDA_VISIBLE_DEVICES=1 python -m legged_lab.scripts.eval \
  --task=t1_tt_eval \
  --num_envs=64 \
  --load_run 2026-06-22_21-09-35 \
  --checkpoint model_9999.pt \
  --predictor \
  --headless
```

## 保护原则

1. 不直接改写 `t1_tt` / `t1_tt_eval` 的既有语义。
2. 优先新增 A3 专用模块和 task，例如 `a3_tt` / `a3_tt_eval`。
3. 如必须修改公共环境代码，只加入带默认值的配置项，并让默认值严格等价于当前 T1 行为。
4. 不回滚当前工作树已有变化。开始分析前，仓库已有大量 `100644 => 100755` 文件模式变化，`git diff --stat` 显示 0 行内容差异。
5. 每次代码修改同步更新 `CODE_CHANGE_LOG_2026-06-23_A3.md`。

## 2026-07-01 A3 stage5g_wide 计划

目的：在已验证 A3 几何 offset 可以让窄球路接近击球中心后，取消窄球路对 hit rate 的放大效应，改用 A3 当前站位和右臂可达范围内的宽球路做长训。

修改边界：

- 新增 `a3_tt_stage5g_wide` / `a3_tt_stage5g_wide_eval`，不覆盖现有 T1、stage5g、stage5g_fixedball；
- 继续复用 stage5g 的 A3 FK 几何 offset；
- 关闭 ball curriculum，固定为 A3 适配宽球路；
- 球路宽度和球速不直接照搬 T1，而从 stage5f 后段 A3 可达范围选取；
- 将 contact threshold 收紧，用于降低“擦到就算 hit”的乐观统计。

验证顺序：

1. `py_compile` 检查配置和注册。
2. `num_envs=64` 做 1 iter smoke。
3. 尝试 `num_envs=4096` 做 1 iter smoke；若 Isaac/A3 资产初始化失败，长训改用 2048 或 1024。

## 2026-07-01 A3 stage5h_hitquality 对照计划

目的：在 GPU0 的 `stage5g_wide` 保持不动的前提下，新增 GPU1 对照实验，验证拍面法向、击球速度和击球窗口 shaping 是否能改善“接触少、成功率打不开”的问题。

对照关系：

- GPU0：`a3_tt_stage5g_wide`，A3 FK 几何修正 + A3 适配宽球路 + 严格 contact；
- GPU1：新增 `a3_tt_stage5h_hitquality`，完全继承同一几何、同一宽球路、同一 contact threshold，只增加击球质量相关 reward。

新增优化：

- 拍面法向：根据诊断脚本确认 A3 球拍正面为本地 `-Z`，奖励该法向在击球窗口内朝向来球方向 `-ball_linvel`；
- 击球速度：保留并加强 post-hit ball velocity/net-clearance reward，同时增加击球窗口内球拍朝对面桌方向的挥拍速度 shaping；
- 击球窗口：使用 `ball_future_t` 和 `ball_future_pose` 构造时间窗口与未来击球点接近奖励，降低“任意时间碰球”对学习的误导。

边界：

- 不停止 GPU0 的 `stage5g_wide`；
- 不修改 T1；
- 新增 task，不覆盖已有 A3 任务；
- 长训使用 GPU1，`num_envs=4096` 先 smoke 再启动。

## 当前代码理解

## 补充分析文档

A3 与 T1 的模型差异、控制差异、奖励差异以及是否需要 A3 专用优化，已单独记录在：

```text
A3_VS_T1_ANALYSIS_2026-06-23.md
```

该文档结论：A3 接入不是简单替换机器人模型文件，而应新增 A3 专用 task。球、球桌、空气动力、球路预测和训练 runner 可复用；机器人资产、动作维度、关节/body 命名、击球点、初始姿态、actuator 和 reward 正则需要 A3 专门适配与调参。

### 任务注册链路

- `legged_lab/envs/__init__.py` 注册：
  - `t1_tt`
  - `t1_tt_eval`
- `legged_lab/scripts/train.py` / `legged_lab/scripts/eval.py` 通过 `task_registry.get_cfgs(task)` 和 `task_registry.get_task_class(task)` 获取环境类与配置。
- 当前乒乓任务环境类为 `TTEnv`。

### T1 机器人配置链路

- `legged_lab/envs/t1_tt/t1_tt_config.py`
  - `T1TableTennisEnvCfg.__post_init__()` 设置机器人、球桌、球、地形、接触 body、关节观测顺序、动作关节顺序。
  - 使用 `BOOSTER_T1_TT_P2_CFG`。
- `legged_lab/assets/booster/booster.py`
  - 定义 T1/T1_TT 的 `ArticulationCfg`。
  - T1_TT 使用 USD：`legged_lab/assets/booster/T1_TT/T1_TT.usd`。

### 需要参数化的 T1 硬编码

`legged_lab/envs/base/tt_env.py` 中存在 A3 迁移必须处理的机器人相关硬编码：

- `compute_paddle_touch()` 使用 `paddle_index = 15`，并假设该 body 是 T1 的右手/球拍 body。
- `compute_paddle_touch()` 使用固定局部偏移 `[0.0, -0.345, 0.0]` 计算虚拟击球点。
- `compute_intermediate_values()` 中有 `body_height = 0.69` 和 `paddle_y_offset = -0.60`，这些和机器人尺寸/击球姿态相关。

迁移时应把这些改成配置项，并给出 T1 当前默认值，避免改变 T1 行为。

### 观测和动作维度

当前 `TTEnv` 根据 `cfg.robot.num_actions`、`cfg.actions.joint_names`、`cfg.observations.joint_names` 自动解析动作关节和观测关节。

使用 perception 观测时，单帧 actor obs 长度约为：

```text
3 * num_actions + 18
```

T1 当前是 21 个动作关节，A3 URDF 有 31 个 revolute 关节。如果 A3 使用全部 31 个关节训练，新策略网络输入/输出维度会变化，不能直接复用 T1 checkpoint。

## A3 资料结论

资料包：

```text
/mnt/ssd/zxl/a3_t2d5 .zip
```

内容：

- `a3_t2d5/urdf/model.urdf`
- `a3_t2d5/meshes/*.STL`，共 39 个 mesh
- `a3_t2d5/README.md`
- 没有现成 USD

README 重要说明：

- 实际机器人腰部 `waist_pitch_joint`、`waist_roll_joint` 以及脚踝 `ankle_roll_Link` / `ankle_roll_joint` 为并联结构。
- URDF 中转速、扭矩等参数已经做过串联等效。
- 训练时可以按串联进行计算和训练，部署时并联解算由部署代码完成。

URDF 结构：

- robot name：`A3T2.5`
- links：39
- joints：38
- revolute joints：31
- fixed joints：7
- 解析总质量约：`57.904204 kg`

关键 body / link 候选：

- root/base：`pelvis_link`
- torso：`torso_Link`
- left foot contact body：`left_ankle_roll_Link`
- right foot contact body：`right_ankle_roll_Link`
- right hand / paddle anchor：`right_hand_Link`
- head：`head_pitch_Link` 或 `head_yaw_Link`

## A3 初版迁移方案

### 1. 资产接入

新增 A3 资产目录，不覆盖 T1：

```text
legged_lab/assets/a3/t2d5/
  meshes/
  urdf/model.urdf
  README.md
```

首选实现：

- 使用 `sim_utils.UrdfFileCfg` 从 URDF 生成/加载 USD。
- `fix_base=False`。
- `merge_fixed_joints=False`，保留 `right_hand_Link` 等 fixed child body，方便配置球拍 anchor。
- `activate_contact_sensors=True`。
- 如运行时 URDF 转 USD 不稳定，再改为显式运行 IsaacLab 的 `scripts/tools/convert_urdf.py` 生成 `A3_T2D5.usd`，并在 `ArticulationCfg` 中引用 USD。

新增：

```text
legged_lab/assets/a3/__init__.py
legged_lab/assets/a3/a3.py
```

定义：

```text
A3_T2D5_CFG
```

## 2026-06-25 A3 站立姿态标定补充计划

用户反馈 A3 在 Isaac Sim 中初始姿态不符合物理世界，释放自由基后会摔倒。结合 T1 配置复盘，当前 A3 初始腿部姿态主要是参考 T1 的下蹲站姿迁移而来，root 高度和腿部角度尚未经过 A3 自身的自由基站立标定。

本阶段目标不是直接修改 `a3_tt` 训练默认姿态，而是新增一个 A3-only 的站立姿态筛选脚本，自动测试一批接近用户参考图的候选姿态：

- 宽站距、屈膝、重心降低。
- 右臂持拍位于身体前方。
- 左臂保持平衡/辅助持拍的自然弯曲姿态。
- root 高度在合理范围内扫描。
- 髋/膝/踝 pitch 采用成组扫描，不逐个关节盲猜。
- hip roll / ankle roll 采用左右镜像候选，用于寻找能让双脚更自然接触地面的宽站距。

筛选脚本应只做离线诊断：

- 不修改 T1。
- 不修改当前 A3 训练 task 的默认行为。
- 不加载策略、不训练。
- 通过零速度 position target 持住候选姿态。
- 记录每个候选的存活时间、root roll/pitch、root 高度漂移、双脚接触率、脚底滑移、球拍击球点到 `ball_future_pose` 的距离。
- 输出排序后的候选 joint pose，作为下一步人工 WebRTC 可视化确认和正式配置修改的依据。

只有当某个候选在自由基下稳定，并且右臂/球拍几何合理后，才考虑把该姿态写入 `legged_lab/assets/a3/a3.py` 或新增 A3 专用 prepose config。

## 2026-06-25 A3 站立物理诊断补充计划

用户补充 A3 电机参数后，已修正 A3 actuator armature 和部分限幅，但自动姿态筛选仍没有找到能稳定站满 250 step 的候选。下一阶段目标是区分以下问题：

- root 高度是否导致脚底悬空或穿地；
- 脚底 collision 是否真实可用；
- 当前 leg/feet/waist PD 是否足以维持静态站立；
- 当前 link inertia / CoM 是否让自由基站立明显不稳定。

计划新增 A3-only 诊断脚本：

```text
legged_lab/scripts/a3_standing_physics_diagnostics.py
```

诊断内容：

1. pinned-root contact scan：
   - 并行扫描一组 `root_z`。
   - 每步固定 root pose，观察左右脚接触率、接触力、脚 body 高度。
   - 用来判断 root 高度和脚底接触窗口。
2. free-base rollout：
   - 对同一组 `root_z` 取消 pin base。
   - 记录 survival steps、root roll/pitch、root z drift、脚滑移和接触率。
   - 用来判断在当前 PD 下是否真正站稳。
3. PD multiplier：
   - 支持命令行临时放大/缩小 waist/leg/feet stiffness/damping。
   - 不写回训练配置，只用于诊断。

边界：

- 只新增 A3 脚本和文档。
- 不修改 T1。
- 不直接把任何诊断候选写成默认训练姿态。

## 2026-06-29 A3 ball curriculum 与严格触球判定计划

用户根据 stage5c 长训曲线提出：当前 A3 高 hit rate 不可信，主要因为训练球路过窄、`contact_threshold` 偏宽，策略可能在固定窄球路上刷触球，而没有学到可泛化的击球任务。站稳 reward/PD 专项暂时不在本次修改范围内，留到后续对比实验。

本阶段目标：

- 新增 A3-only task，不覆盖 `t1_tt`、`a3_tt_stage5c` 或已有 checkpoint workflow。
- 将 A3 训练触球判定从 `0.07` 收紧到 `0.05`，与 T1 默认严格度对齐。
- 新增 ball curriculum，让球路从 A3 当前初始姿态附近开始，但随训练 step 逐渐扩大横向范围、速度范围和纵向高度范围，避免策略长期依赖单一窄球路。
- 降低纯 `reward_contact` 在早中期的支配性，保留并继续强调击球后速度、过网高度和落点质量，避免“碰到球就够了”的局部最优。

拟新增任务：

```text
a3_tt_stage5d
a3_tt_stage5d_eval
```

ball curriculum 设计：

1. 初期仍围绕当前 A3 击球中心附近发球，避免一开始完全奖励稀疏；
2. 前 500 iter 左右只做轻微扩展，先验证严格触球条件下是否还能建立真实接触；
3. 约 500-1500 iter 扩大横向 `y` 和 `pos_y`，不让策略只记住一条固定球线；
4. 约 1500-3500 iter 增加 `x` 速度跨度和 `z` 速度跨度，让策略学习预测与挥拍时序；
5. 约 3500-6000 iter 逐步接近 T1 的球路随机范围，但仍保留 A3 专用上限，避免突然恢复到过难分布。

实现边界：

- curriculum 只修改 `cfg.ball.ball_speed_x_range`、`ball_speed_y_range`、`ball_speed_z_range`、`ball_pos_y_range`；
- 发球物理、空气动力、`ball_future_pose` 计算、predictor 和 T1 配置均不变；
- 本次不新增站稳 gate、不修改 reset 条件、不改 A3 PD；
- 训练监控时不再单看 `Train/TT_hit_rate`，必须同时看 `mean_episode_length`、`TT_success_rate`、`reward_future_pass_net`、`reward_table_success` 和 strict hit 后的 hit 曲线。

## 2026-06-29 A3 soft stability gate 计划

stage5d 的阶段曲线显示：ball curriculum 和严格触球判定能压低虚高 hit rate，但 `mean_episode_length` 仍长期约 `73`，`TT_success_rate` 仍为 0，说明策略没有自然形成“稳定站立下打球”的链路。

本阶段目标是新增 A3-only stage，不覆盖 T1、stage5c、stage5d：

```text
a3_tt_stage5e
a3_tt_stage5e_eval
```

核心原则：

- 站稳是所有高价值动作的前提；
- 不能做硬门控：`不站稳 -> 所有任务奖励为 0`，否则早期奖励稀疏；
- 使用 soft stability gate：不稳定时任务奖励仍保留少量梯度，但拿满高价值击球奖励必须稳定；
- 同时新增 dense standing reward，让策略每一步都知道如何站稳，而不是只在摔倒后被惩罚。

稳定门控公式：

```text
gate_raw = exp(sum_i normalized_weight_i * log(clamp(score_i, min=1e-3)))
stability_gate = gate_floor + (1 - gate_floor) * gate_raw
```

默认 gate score 组成：

```text
height_score         weight 0.28
upright_score        weight 0.24
support_score        weight 0.20
low_velocity_score   weight 0.14
contact_clean_score  weight 0.09
feet_width_score     weight 0.05
```

各 score 定义：

```text
height_score =
  exp(-low_deficit^2 / height_std^2) * exp(-high_excess^2 / height_std^2)

upright_score =
  exp(-projected_gravity_xy_l2 / upright_std^2)

support_score =
  0.05, if no foot contact
  0.65, if single foot contact
  0.75 + 0.25 * force_balance_score, if both feet contact

force_balance_score =
  exp(-(abs(F_left - F_right) / (F_left + F_right + eps))^2 / force_balance_std^2)

low_velocity_score =
  exp(-(root_lin_vel_xy^2 / lin_vel_std^2 + root_ang_vel_xy^2 / ang_vel_std^2))

contact_clean_score =
  exp(-bad_contact_count / bad_contact_std)

feet_width_score =
  exp(-(feet_xy_distance - target_feet_width)^2 / feet_width_std^2)
```

初始参数：

```text
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
```

门控参数：

- 接近球/触球类 reward 使用 `gate_floor=0.30`，避免早期完全没探索梯度；
- 击球质量/过网/落点类 reward 使用 `gate_floor=0.15`，让高质量任务收益更依赖稳定；
- 稀疏成功类 reward 使用 `gate_floor=0.10`，防止不稳定偶然成功被过度放大。

新增 dense standing reward：

```text
reward_standing_stability =
  0.28 * height_score
  + 0.24 * upright_score
  + 0.20 * support_score
  + 0.14 * low_velocity_score
  + 0.09 * contact_clean_score
  + 0.05 * feet_width_score
```

新增 unstable-hit 惩罚：

```text
penalty_unstable_hit = ball_contact_rew * (1 - gate_raw)
```

设计含义：

- 不稳定状态下仍可获得一部分探索 reward；
- 不稳定状态下触球会被专门惩罚；
- 高价值击球、过网、落台奖励必须在稳定门控下才能拿满；
- `termination_penalty`、`undesired_contacts`、`flat_orientation_l2`、能耗、动作平滑、关节偏离等基础约束不被 gate。

拟被 gate 的任务项：

```text
reward_contact
reward_future_touch_point
reward_future_dis_ee
reward_future_landing_x_progress
reward_hit_ball_velocity_net
reward_hit_net_clearance_progress
reward_future_pass_net
reward_post_hit_net_progress
reward_table_success
reward_actual_opponent_table_target
```

保留 stage5d 的 ball curriculum 和 `contact_threshold=0.05`，确保 stage5e 只在“严格触球 + 球路逐步放宽”的基础上验证稳定门控效果。

实现确认（2026-06-29 18:10）：

- stage5e 的公式和参数保持上述设计不变；
- 代码实现采用显式 `score_kwargs` 字典向 reward wrapper 传入稳定性参数，避免 IsaacLab RewardManager 把 `**kwargs` 当作缺失强制参数；
- `SceneEntityCfg` 在稳定门控内部按需解析，兼容已经由 RewardManager 解析过的 cfg，也兼容 `body_ids=slice(None)` 的情况；
- 最小 smoke test 已通过：

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

生成目录：

```text
logs/a3_table_tennis/2026-06-29_18-10-23_a3_stage5e_smoke_64_1_retry3
```

### 2. 公共配置最小扩展

在 `legged_lab/envs/base/tt_config.py` 的 `RobotCfg` 中新增带 T1 默认值的字段：

```python
paddle_body_name: str | None = None
paddle_body_index: int = 15
paddle_local_offset: tuple = (0.0, -0.345, 0.0)
future_body_height: float = 0.69
future_paddle_y_offset: float = -0.60
```

在 `TTEnv.init_buffers()` 或附近解析：

- 如果 `paddle_body_name` 非空，使用 `robot.find_bodies()` 获取 body id。
- 否则回退到 `paddle_body_index = 15`，保持 T1 旧行为。

在 `TTEnv.compute_paddle_touch()` 使用配置后的 body id 和 offset。

在 `TTEnv.compute_intermediate_values()` 使用 `future_body_height` / `future_paddle_y_offset`。

### 3. A3 任务配置

新增：

```text
legged_lab/envs/a3_tt/__init__.py
legged_lab/envs/a3_tt/a3_tt_config.py
```

注册：

```python
task_registry.register("a3_tt", TTEnv, A3TableTennisEnvCfg(), A3TableTennisAgentCfg())
task_registry.register("a3_tt_eval", TTEnv, A3TT_EvalEnvCfg(), A3TableTennisAgentCfg())
```

A3 配置要点：

- `self.scene.robot = A3_T2D5_CFG`
- `self.scene.height_scanner.prim_body_name = "torso_Link"`
- `self.robot.terminate_contacts_body_names = ["torso_Link", "pelvis_link"]`
- `self.robot.feet_body_names = ["left_ankle_roll_Link", "right_ankle_roll_Link"]`
- `self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]`
- `self.robot.num_actions = 31`
- `self.robot.num_joints = 31`
- `self.robot.paddle_body_name = "right_hand_Link"`
- `self.robot.paddle_local_offset` 先用保守初值，之后通过可视化/打印微调。
- `self.robot.future_body_height` 先按 A3 身高提高到约 `0.9` 附近，实际值通过站立姿态验证后调整。

### 4. A3 关节顺序

初版使用全部 31 个 revolute joints，动作顺序建议和 URDF revolute 顺序一致：

```text
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
head_yaw_joint
head_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
```

后续如训练不稳，可做第二版关节子集策略，例如固定头部或左手腕，减少动作维度。

### 5. A3 reward 命名适配

新增 A3 专用 `A3TableTennisRewardCfg`，不要改 T1 reward。

需要替换的 T1 body/joint 正则：

- `.*_foot_link*` -> `left_ankle_roll_Link|right_ankle_roll_Link` 或显式列表
- `left_foot_link*` -> `left_ankle_roll_Link`
- `right_foot_link*` -> `right_ankle_roll_Link`
- `.*H2*` -> `head_pitch_Link` 或先降低/禁用该项
- `.*_Ankle_Pitch` -> `.*ankle_pitch_joint`
- `.*_Ankle_Roll` -> `.*ankle_roll_joint`
- `.*_Hip_Yaw` -> `.*hip_yaw_joint`
- `.*_Hip_Roll` -> `.*hip_roll_joint`
- `Left_Shoulder_.*` -> `left_shoulder_.*_joint`
- `Left_Elbow_.*` -> `left_elbow_joint`
- `Right_Shoulder_.*` -> `right_shoulder_.*_joint`
- `Right_Elbow_.*` -> `right_elbow_joint`
- `Waist` -> `waist_.*_joint`

### 6. 初始姿态和 actuator

`A3_T2D5_CFG` 中需要配置：

- base 初始位置，先估计 `pos=(-1.6, 0.0, 0.9)`，通过仿真检查是否脚底落地。
- 站立关节初值，参考 T1 但使用 A3 命名：
  - hips pitch 约 `-0.20`
  - knees 约 `0.42`
  - ankle pitch 约 `-0.23`
  - hip/ankle roll 和 hip yaw 约 `0.0`
  - 右臂摆到可击球姿态，左臂保守下垂或平衡姿态。
- actuator effort/velocity limit 参考 URDF limit。
- PD 参数先保守复用 T1 delayed PD 思路，A3 训练稳定后再调。

### 7. 验证计划

#### 代码静态验证

- 检查新增 task 是否可被 registry 找到。
- 检查 A3 joint/body 正则能 resolve。
- 检查 actor/action 维度和 `robot.num_actions` 一致。

#### T1 回归验证

先跑当前用户已复现命令，或至少用短 timeout 验证启动流程不变：

```bash
CUDA_VISIBLE_DEVICES=1 python -m legged_lab.scripts.eval \
  --task=t1_tt_eval \
  --num_envs=64 \
  --load_run 2026-06-22_21-09-35 \
  --checkpoint model_9999.pt \
  --predictor \
  --headless
```

#### A3 smoke test

无 checkpoint 时不能直接 eval。先做训练启动 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --headless \
  --predictor
```

如需要快速退出，使用外部 timeout，只观察是否完成环境构建、body/joint resolve 和第一步 rollout。

#### A3 可视化检查

在非 headless 模式或通过临时调试打印确认：

- A3 初始姿态没有穿地/漂浮。
- feet body 是左右脚实际接触 body。
- `right_hand_Link` / paddle offset 给出的击球点位置合理。
- 球桌、发球范围和 A3 站位没有明显冲突。

## 主要风险

1. A3 没有现成 USD，URDF 转换可能受 IsaacSim 环境影响。
2. A3 原始模型没有球拍 mesh，虚拟击球点可训练奖励，但真实球-拍碰撞可能不足。可能需要后续给右手附加 paddle collision/visual。
3. A3 31 动作维度会导致策略网络维度变化，不能复用 T1 checkpoint。
4. 初始站立高度和 PD 参数需要仿真调参。
5. 如果 `merge_fixed_joints=True`，`right_hand_Link` 可能被合并消失，因此初版应保持 `False`。

## 待确认问题

1. A3 是否需要右手真实球拍模型和碰撞体？如果有官方 paddle/末端夹具模型，最好补充。
2. 是否希望首版使用全部 31 个关节，还是先固定头部/部分腕关节降低训练难度？
3. 是否有 A3 推荐站立姿态、PD 参数、关节电机参数文档？当前只能从 URDF effort/velocity 和 T1 经验初值开始。
4. 你希望 A3 日志目录命名为 `a3_table_tennis` 还是沿用某个项目命名规范？

## 2026-06-24 新增任务：A3 右臂+球拍预姿态标定脚本

### 目标

新增一个 A3-only 调试脚本，用于通过 Isaac Sim WebRTC 观察并量化 A3 reset 后的球拍中心与预测击球点之间的距离。

### 脚本范围

新增：

```text
legged_lab/scripts/a3_pose_calibration.py
```

不修改：

```text
t1_tt / t1_tt_eval 注册和配置
公共 TTEnv reward 行为
训练脚本 train.py
评估脚本 eval.py
```

### 功能要求

1. 默认运行 `a3_tt_eval` 单环境。
2. 支持 `--livestream 2`，用于 Isaac Sim WebRTC Streaming Client。
3. 固定或收窄球路、base reset 和右臂 reset noise，便于观察。
4. 绿色球显示 `ball_future_pose`。
5. 黄色球显示 `paddle_touch_point`。
6. 终端周期性打印：

```text
ball_future_pose
paddle_touch_point
paddle_to_future_distance
paddle_ball_distance
right arm joint values
```

7. 支持命令行覆盖右臂关节初值，便于重启对比候选姿态。

### 边界

该脚本只用于调试标定，不改变训练流程；所有姿态候选确认后，再单独决定是否写入 A3 pingpong 专用配置。

## 2026-06-25 新增任务：A3 初始站立物理诊断

### 背景

A3 1000 iter 训练稳定运行，但 TensorBoard 中乒乓球相关指标没有明显学习迹象。结合可视化观察，A3 在 reset 后不能稳定站立，导致球拍难以进入球路附近。这个问题优先级高于 reward/curriculum 调参。

### 已完成

1. 新增 A3-only 自动站姿筛选脚本：

```text
legged_lab/scripts/a3_standing_calibration.py
```

用途：批量测试接近参考图的下蹲持拍姿态、zero-leg 姿态和 light-crouch 姿态。

2. 按用户补充的 A3 电机参数表修正 A3 actuator：

```text
legged_lab/assets/a3/a3.py
```

内容：armature、部分 effort limit 和 velocity limit。T1 配置未改。

3. 新增 A3-only 站立物理诊断脚本：

```text
legged_lab/scripts/a3_standing_physics_diagnostics.py
```

用途：

- `--phase=pinned`：固定 root，扫描 root_z 下脚底接触、支撑力和滑移。
- `--phase=free`：自由基短 rollout，评估 reset、roll/pitch、z drift 和 survival steps。
- 命令行临时调整 leg/feet/waist PD scale，不写入默认训练配置。

### 当前关键结果

```text
pinned current pose:
  root_z=0.90: contact_ratio=0.780, foot_slip=0.6623
  root_z=1.02: contact_ratio=1.000, foot_slip=0.0399
  root_z=1.08: contact_ratio=0.000

free current pose:
  root_z=0.90: 92/250 steps
  root_z=1.02: 101/250 steps

free light_crouch pose + leg/feet damping x2:
  root_z=1.02: 140/250 steps, reset_seen=False, max_abs_roll_pitch=0.5517
```

### 当前判断

- A3 当前默认 `root_z=0.90` 不适合作为乒乓训练初始高度。
- `root_z=1.02` 更接近脚底稳定接触区间。
- 单纯增大 stiffness 不解决问题；leg/feet damping 对短时稳定性更有帮助。
- 继续确认后，最佳候选更新为 `zero_light_crouch + root_z=1.05 + waist damping x2 + leg/feet damping x3`。
- 该候选通过严格 1000 step 站立诊断：

```text
survival=1000/1000
reset_seen=False
max_abs_roll_pitch=0.5412
max_root_z_drift=0.1796
both_feet_contact_ratio=0.981
```

- 上肢/球拍非对称会明显放大 A3 初始失稳；右臂持拍姿态不能一开始就激进设置。
- 继续搜索 zero-arm 宽站姿后，当前更优候选更新为：

```text
candidate_index=205
root_z=1.055
hip_pitch=-0.18
knee=0.38
ankle_pitch=-0.20
left_hip_roll=0.10
right_hip_roll=-0.10
left_ankle_roll=-0.045
right_ankle_roll=0.045
arms=zero arms
waist_damping_scale=2.0
leg_damping_scale=3.0
feet_damping_scale=3.0
```

该候选单独 1000 step 复测：

```text
survival=1000/1000
reset_seen=False
max_abs_roll_pitch=0.4789
max_root_z_drift=0.1886
max_foot_slip=0.0951
both_feet_contact_ratio=0.991
```

因此当前优先使用 `candidate_index=205` 作为 A3 站立基线候选。

### 下一步

1. 用 Isaac Sim WebRTC 观察 `candidate_index=205`，确认脚底碰撞和身体姿态是否符合预期。
2. 如果可视化确认合理，把该候选写入 A3 pingpong 专用配置，不影响 T1。
3. 在稳定基线之上逐步加入右臂/球拍预姿态，采用 500-1000 step 站立诊断筛选，而不是一次性设置完整击球姿态。
4. 确认站立和右臂预姿态都稳定后，再重新做 250-500 iter 小规模训练，观察 contact/hit/future distance 指标是否提升。

### 暂不做

- 暂不继续长时间训练。
- 暂不修改 T1 workflow。
- 暂不把 pin-base 或未通过可视化确认的候选姿态写入正式训练配置。
- 暂不优先改 predictor；当前瓶颈是 A3 reset 站姿和击球空间不可达。

## 2026-06-25 真机姿态输入与混合站姿搜索计划

用户提供了一组 A3 真机/控制器侧的 31 DOF 姿态数据，字段包含 `layout` 和 `q`。该 `layout` 与当前 A3 31 个关节顺序一致，说明可以作为仿真标定输入。

初步仿真判断：

- 该姿态的上半身和双臂较自然，可作为后续右臂/球拍预姿态的参考。
- 下肢比当前仿真稳定候选更直，膝关节约 `0.20 rad`，而 `candidate_index=205` 的膝关节为 `0.38 rad`。
- 髋 roll 接近 0，站距偏窄；`candidate_index=205` 使用左右髋 roll 打开到约 `+0.10/-0.10`。
- 直接用该完整姿态做自由基短 rollout 时，脚底滑移很小，但 roll/pitch 会略超过严格阈值 `0.55 rad`，因此不能直接作为稳定训练 reset 姿态。

本阶段计划：

1. 扩展 A3-only 离线站姿脚本 `legged_lab/scripts/a3_standing_calibration.py`。
2. 新增 `--measured_pose_json` 参数，读取用户提供的 `layout + q` 姿态文件。
3. 新增 `measured` 上肢模式，用真机的腰、头、双臂关节作为候选上半身。
4. 生成两类候选：
   - `measured_full`：完整真机姿态，只用于对照。
   - `measured_upper_*`：真机上半身 + 扫描后的稳定下肢候选。
5. 不修改 `a3_tt` 或 `a3_tt_eval` 默认 reset 姿态。
6. 不修改 T1 任务、T1 资产或公共训练/eval workflow。

筛选目标：

- 优先找到能稳定站满 500-1000 step 的候选。
- `max_abs_roll_pitch` 尽量低于当前 `candidate_index=205` 的 `0.4789` 或至少不明显变差。
- 保持脚底滑移较低，目标小于约 `0.10 m`。
- 在稳定后再把右臂/球拍姿态逐步引入训练配置。

第一轮 `measured` 上半身搜索结果显示，完整真机上半身直接叠加到稳定下肢后仍会在约 50-80 step 内失败。因此需要继续把真机姿态拆分为更小的子模式：

- `measured_torso`：只使用真机腰部/头部，双臂清零。
- `measured_arms`：只使用真机双臂，腰部/头部中立。
- `measured_right_arm`：只使用真机右臂，左臂和腰部/头部中立。

这样可以区分失稳主要来自腰部前倾、双臂质心前移，还是右臂/球拍侧的姿态本身。

第二轮结果表明：

- `measured_torso` 在放宽 roll/pitch 阈值到 `0.70 rad` 后可以找到 500 step 存活候选，但姿态倾斜仍偏大。
- `measured_arms` 和 `measured_right_arm` 仍会触发环境 reset 或明显失稳。
- 因此当前不能直接采用完整真机右臂姿态作为 reset 预姿态。

下一步改为渐进插值：

- 新增类似 `measured_right_arm_blend_0.25` 的模式。
- 只将右臂从 zero posture 朝真机右臂姿态插值一部分。
- 先搜索 `0.25 / 0.50 / 0.75 / 1.00` 等比例，找能稳定站住且球拍更接近击球区域的折中点。

## 2026-06-25 A3 训练托底机制专项修改计划

用户确认可以开始做 A3 专项托底机制修改。本阶段只修改 A3 asset/task/agent 配置，不修改 T1，不修改公共 `TTEnv` 行为。

目标：

1. 把已经通过 1000 step 站立复测的 A3 zero-arm 宽站姿写为 A3 默认姿态。
2. 把站立诊断中有效的 waist/leg/feet damping 放大写入 A3 actuator。
3. 缩小 A3 训练初期 reset 随机范围，让随机策略在稳定姿态附近探索。
4. 降低 A3 初期 action scale 和 PPO 初始动作噪声，减少 31 DOF 随机策略早期把机器人推倒的概率。

拟采用的 A3 默认姿态：

```text
root_z=1.055
waist/head=0
arms=zero arms
left_hip_pitch=-0.18
left_hip_roll=0.10
left_hip_yaw=0.0
left_knee=0.38
left_ankle_pitch=-0.20
left_ankle_roll=-0.045
right_hip_pitch=-0.18
right_hip_roll=-0.10
right_hip_yaw=0.0
right_knee=0.38
right_ankle_pitch=-0.20
right_ankle_roll=0.045
```

拟采用的 A3 PD 变化：

- `waist.damping`: `4.0 -> 8.0`
- `legs.damping`: `4.0 -> 12.0`
- `feet.damping`: `4.0 -> 12.0`
- stiffness、effort limit、velocity limit 暂不继续扩大，避免在脚底接触处引入过强弹跳。

拟采用的 A3 初期训练范围：

- `robot.action_scale`: `0.20 -> 0.15`
- locomotion reset scale: `(0.5, 1.5) -> (0.95, 1.05)`
- right-arm manipulation reset offset: `(-0.5, 0.5) -> (-0.05, 0.05)`
- base reset pose/velocity 收窄到接近当前 eval 范围。
- A3 PPO `init_noise_std`: `1.0 -> 0.5`
- A3 初期关闭 `push_robot` 外力扰动。
- A3 初期关闭 base mass randomization，后续站稳后再作为 robustness curriculum 打开。

边界：

- 这些改动只对 `a3_tt` / `a3_tt_eval` 生效。
- 当前默认姿态是站立基线，不是最终右臂持拍击球预姿态。
- 后续右臂/球拍预姿态应在该基线上通过插值或 curriculum 渐进引入。

## 2026-06-25 A3 WebRTC 当前默认姿态可视化修正

用户使用 WebRTC 观察后反馈机器人仍会往后倒。复盘发现上一条可视化命令使用了 `candidate_index=37`，而 `a3_standing_calibration.py` 的候选编号会随 `root_z_values` 和 `crouch_arm_modes` 变化；该编号并不等价于已经写入 A3 配置的 `candidate_index=205` 稳定默认姿态。

本阶段先修正可视化工具，避免继续通过不稳定编号误判：

- 给 `legged_lab/scripts/a3_standing_calibration.py` 增加 `--use_current_default_pose`。
- 该模式不从候选列表猜编号，而是直接使用环境加载后的 `robot.data.default_joint_pos`。
- root_z 仍由 `--root_z_values` 的第一个值指定，默认可传 `1.055`。
- 该模式只用于 A3 可视化/诊断，不改变训练配置。

如果用户用该模式仍然观察到真实 A3 默认姿态往后倒，再进入第二轮姿态/PD 修改：

- 尝试轻微前移质心：
  - 腰 pitch 小幅前倾；
  - 髋 pitch / 膝 / 踝 pitch 联动调整；
  - 检查脚底接触和 root pitch。
- 优先调 damping，不优先放大 stiffness。

## 2026-06-25 A3 往后倒第二轮姿态/PD 调整计划

用户继续 WebRTC 观察后反馈 A3 仍然会往后倒。本轮把它作为正式 A3 专项稳定性问题继续处理，不影响 T1。

已确认的风险点：

- 当前机器上还残留过一个旧 `a3_standing_calibration.py` WebRTC 进程，命令未带 `--use_current_default_pose`，可能仍在显示旧候选姿态。
- 但即使排除可视化命令误差，A3 训练 reset 后仍需要更强的 A3 专用托底：默认姿态、reset 随机范围和低层 PD 要一起匹配。

调整方向：

1. 先停止旧可视化进程，避免 WebRTC 继续观察到旧姿态。
2. 用 headless 物理诊断批量搜索更抗“往后倒”的姿态：
   - pelvis/root 高度围绕 `1.04-1.08` 搜索；
   - hip/knee/ankle pitch 做联动，让躯干初始力矩更接近脚掌支撑面；
   - 允许腰 pitch 小幅前倾；
   - 保持左右 hip/ankle roll 宽站姿，避免侧向倒。
3. PD 方向：
   - 继续优先加强 damping，尤其 ankle/leg/waist；
   - 如仍然慢慢后仰，再小幅提高 leg/ankle stiffness；
   - 不扩大 effort/velocity limit，避免引入额外训练行为变化。
4. 通过 500-1000 step free-base 站立诊断后，再写入 A3 asset。

验收标准：

- A3 单环境 free-base 站立至少 `1000` step 不触发 reset。
- 观察到 `max_abs_roll_pitch`、`max_root_z_drift`、`max_foot_slip` 不比上一版更差。
- WebRTC 命令必须使用 `--use_current_default_pose`，不能再依赖候选编号。

第二轮搜索结论：

- 当前默认姿态 1000 step 不 reset，但最终 pitch 约 `0.45 rad`，root 高度会从 `1.055` 下沉到约 `0.88`，WebRTC 上容易表现为明显后仰/塌下去。
- 单纯加硬 PD 不合适：
  - 强 PD 能减少高度下沉和脚滑，但会引入较大姿态冲击，并可能触发 reset；
  - 中等 PD 也会短暂超过 roll/pitch 诊断阈值；
  - 因此本轮不继续提高 stiffness/damping。
- 更有效的是把腿部默认姿态从当前深蹲姿态往浅一点调整：
  - `hip_pitch`: `-0.18 -> -0.14`
  - `knee`: 保持 `0.38`
  - `ankle_pitch`: `-0.20 -> -0.16`
  - `waist_pitch`: 保持 `0.0`
- 候选复测结果：

```text
candidate 69
hip_pitch=-0.14
knee=0.38
ankle_pitch=-0.16
root_z=1.055
1000/1000 steps, reset=0
max_abs_roll_pitch=0.509425
max_root_z_drift=0.196764
max_foot_slip=0.098541
final_pitch=0.269789
final_root_z=0.875572
```

对比当前默认姿态：

```text
current candidate-like baseline
1000/1000 steps, reset=0
max_abs_roll_pitch=0.492823
max_root_z_drift=0.180558
max_foot_slip=0.088509
final_pitch=0.450143
final_root_z=0.885630
```

解释：

- 新姿态不是所有指标都更小，但它明显降低最终后仰 pitch；
- 这更符合用户在 WebRTC 中看到的主要问题；
- PD 保持当前托底值，避免把问题变成高 stiffness 冲击。

## 2026-06-26 A3 WebRTC 视觉姿态纠正计划

用户 WebRTC 观察后指出当前姿态完全不对：

- 双腿是并拢的，而不是展开站姿；
- 膝盖屈曲不足，不像乒乓球准备姿态；
- 上半身没有轻微前倾。

复盘：

- 上一轮候选主要按 `final_pitch` 和稳定性指标筛选；
- `hip_roll` 的正负方向没有以 WebRTC 视觉结果为准，导致写入了视觉上收腿的方向；
- 因此 headless 稳定指标不能替代可视化姿态验收。

本轮目标：

1. 恢复/加强视觉上的宽站姿：
   - 左右 hip_roll 使用 WebRTC 可见的展开方向；
   - ankle_roll 与 hip_roll 成对调整，尽量让脚底平放。
2. 增加屈膝：
   - 膝盖从当前 `0.38` 提高到更明显屈曲；
   - hip_pitch/ankle_pitch 联动，避免只弯膝导致身体后倒。
3. 上半身轻微前倾：
   - waist_pitch 只做小幅调整；
   - 以 WebRTC 视觉为主、headless 不摔为底线。
4. 验证顺序：
   - 先 1000 step headless free-base 保证不立刻摔；
   - 再给用户 WebRTC 命令确认视觉姿态；
   - 如果视觉姿态仍不对，以 WebRTC 反馈继续微调。

本轮候选选择：

```text
candidate 176
hip_pitch=-0.22
knee=0.50
ankle_pitch=-0.22
waist_pitch=-0.02
left_hip_roll=+0.16
right_hip_roll=-0.16
left_ankle_roll=-0.072
right_ankle_roll=+0.072
root_z=1.055
1000/1000 steps, reset=0
max_abs_roll_pitch=0.440652
max_root_z_drift=0.215386
max_foot_slip=0.106045
final_pitch=0.230375
final_root_z=0.839614
```

选择理由：

- `w+0.16` 是当前 WebRTC 反馈的反方向，目标是让腿视觉上展开；
- `knee=0.50` 比上一版 `0.38` 更明显屈膝；
- `waist_pitch=-0.02` 是轻微前倾，不采用更大的腰 pitch，避免姿态冲击；
- 虽然 root 会下沉，但不触发 reset，且 roll/pitch 峰值比上一版更低。

## 2026-06-26 TT WebRTC probe 计划

用户运行 T1 `train.py` 和 `preview.py` WebRTC 后反馈：

- Isaac 界面一开始可见；
- 一进入训练/仿真循环就黑屏。

复盘：

- `train.py` / `preview.py` 不是 WebRTC 观察专用脚本；
- 它们没有持续固定 camera，也没有显式 `simulation_app.update()` 和 `visualize_sleep`；
- A3 标定脚本之所以能稳定显示，是因为专门做了相机刷新、render/update 和节流。

本轮计划：

- 新增通用 TT WebRTC 观察脚本 `legged_lab/scripts/tt_webrtc_probe.py`；
- 支持 `zero` 和 `random` action：
  - `zero` 用来看默认姿态 + PD 是否托住身体；
  - `random` 用来近似训练初期随机策略扰动；
- 支持固定 camera、warmup render、每步 update、sleep；
- 不修改 T1/A3 训练代码；
- 不修改公共环境逻辑。

## 2026-06-26 A3 持拍默认姿态落地计划

用户确认 T1 在可视化下也可能摔倒，因此当前阶段不再以 T1 的早期训练可视化为参照，而是先为 A3 建立一个自洽的 reset 默认姿态：

目标：

1. 保留已经验证过的 A3 宽站姿和屈膝下肢默认姿态。
2. 在默认姿态中加入右臂持拍预姿态，使 reset 后球拍不是完全零位悬挂。
3. 右臂姿态必须经过自由基站立诊断，不能直接采用激进 ready arms 或完整真机右臂姿态。
4. 只修改 A3 asset/task 配置，不修改 T1、不修改公共 TTEnv workflow。

候选复测结论：

```text
full ready arms:
  当前腿部 + ready arms 只存活约 91/1000 step，触发 reset，不采用。

measured_right_arm_blend_0.50:
  2000/2000 step
  reset_seen=0
  final_root_z=0.846580
  max_abs_roll_pitch=0.610165
  max_root_z_drift=0.226591
  max_foot_slip=0.025867
  both_feet_contact_ratio=0.999500

measured_right_arm_blend_0.75:
  1000 step 可通过，但 2000 step 复测只存活 381/2000 并触发 reset，不采用。

measured_right_arm_blend_1.00:
  约 108/1000 step 后触发 reset，不采用。
```

落地选择：

- 采用 `measured_right_arm_blend_0.50` 作为第一版 A3 持拍默认姿态；
- 具体右臂角度来自用户提供的 `layout + q` 姿态中右臂关节的 50% 插值；
- 左臂暂时保持 zero arms，避免双臂质心前移导致站立不稳；
- reset 中右臂随机偏移从 `(-0.05, 0.05)` 收紧到 `(-0.02, 0.02)`，先保证训练早期围绕持拍默认姿态探索。

## 2026-06-26 A3 拍面朝向球路计划

用户指出当前仅有“持拍”还不够，拍子需要横过来，让一个拍面正对球路，降低早期接球奖励稀疏。

几何分析：

- A3 新 URDF 中 `pingpang_red_Link` / `pingpang_black_Link` 的 mesh 很薄；
- mesh 包围盒尺寸约为 `0.1604 x 0.0029 x 0.1604 m`；
- 因此拍面平面是局部 `X-Z` 平面，拍面法向是 `right_hand_pingpang_Link` 的局部 `Y` 轴；
- A3 评估球速默认约为：
  - `vx=(-6.5, -5.2)`
  - `vy=(-0.6, 0.2)`
  - `vz=(1.5, 1.9)`
- 来球主要沿世界系 `-X` 方向运动，因此拍面法向应尽量朝世界系 `+X`，让正面迎向来球。

本阶段计划：

1. 扩展 A3-only 调试脚本 `legged_lab/scripts/a3_pose_calibration.py`，打印：
   - 拍面局部 `+Y/-Y` 在世界系方向；
   - 与来球反方向的对齐度；
   - 拍面切向与世界竖直方向的关系。
2. 先用脚本量化当前默认姿态的拍面方向。
3. 搜索/微调右腕和右臂关节：
   - 目标一：`local +Y` 或 `local -Y` 与 `-ball_velocity` 高对齐；
   - 目标二：`paddle_touch_point` 靠近 `ball_future_pose`；
   - 目标三：free-base 站立不触发 reset。
4. 将通过验证的姿态写入 A3 默认姿态。

边界：

- 不修改 T1；
- 不修改公共 TTEnv；
- 不修改 reward；
- 只改变 A3 默认右臂/球拍预姿态和 A3-only 调试输出。

筛选结果：

```text
当前默认:
  step=1 face_alignment=0.0488
  step=1 paddle_to_future_dist=0.4262 m
  2000 step reset_count=0
  2000 step paddle_to_future_dist=0.9426 m

right_wrist_yaw_joint=-1.45:
  step=1 face_alignment=0.9562
  step=1 paddle_to_future_dist=0.1812 m
  2000 step reset_count=15

right_wrist_yaw_joint=-1.20:
  step=1 face_alignment=0.9094
  step=1 paddle_to_future_dist=0.2317 m
  2000 step reset_count=0
  2000 step paddle_to_future_dist=0.7315 m
```

选择：

- 落地 `right_wrist_yaw_joint=-1.20`；
- 不使用 `-1.45`，因为它虽然拍面对齐更好，但会导致 reset；
- 其他右臂关节保持上一版稳定持拍姿态。

## 2026-06-26 T1 参照拍面诊断计划

用户继续观察后指出：A3 球拍仍然不像 T1 那样“完全横过来，一个完整面正对来球方向”。因此不能只看 A3 的 `face_alignment`，还需要直接量化 T1 默认姿态下的球拍局部轴和击球点位置，作为 A3 的几何参照。

本阶段计划：

1. 新增通用 TT 拍面诊断脚本 `legged_lab/scripts/tt_paddle_pose_diagnostics.py`；
2. 支持 `--task=t1_tt_eval` 和 `--task=a3_tt_eval`；
3. 打印：
   - paddle body 名称；
   - paddle local X/Y/Z 在世界系的方向；
   - `paddle_local_offset` 对应的击球点；
   - 来球反方向 `incoming_ball_dir`；
   - 三个局部轴与来球反方向的对齐度；
   - `paddle_to_future_dist`。
4. 用 T1 输出判断“视觉上正对球路”的实际轴向；
5. 再据此调整 A3，而不是只根据 A3 自身的局部 `Y` 轴判断。

边界：

- 新脚本只做诊断；
- 不修改 T1；
- 不修改公共环境逻辑；
- A3 默认姿态只有在和 T1 参照对齐后再修改。

诊断结果：

```text
T1:
  paddle_body=right_hand_link
  best_axis=+X
  axis_+X_w=[0.8648, 0.4552, -0.2118]
  best_alignment=0.9049

A3 previous:
  paddle_body=right_hand_pingpang_Link
  best_axis=+Y
  axis_+Y_w=[0.8000, 0.3344, -0.4982]
```

用户的视觉反馈与诊断一致：A3 虽然 `align_+Y` 较高，但 `z=-0.4982` 下倾明显，不像 T1 那样横。

最终候选：

```text
right_wrist_roll_joint=-0.25
right_wrist_pitch_joint=-0.40
right_wrist_yaw_joint=-1.20

step=1:
  paddle_face_+y_w=[0.8736, 0.4268, -0.2338]
  face_alignment=0.9179
  paddle_to_future_dist=0.2146 m

2000 step:
  reset_count=0
```

选择：

- 用 A3 `+Y` 拍面法向匹配 T1 `+X` 有效迎球轴；
- 落地 `right_wrist_roll=-0.25`、`right_wrist_pitch=-0.40`、`right_wrist_yaw=-1.20`；
- 该姿态比上一版视觉上更横，也比当前 T1 参照更接近。

## 2026-06-26 A3 完整拍面坐标系对齐计划

用户在 WebRTC 中继续观察后指出：A3 球拍仍没有像 T1 那样完全横过来。复盘后确认上一版只主要约束了“拍面法向正对球路”，但视觉上还需要约束拍面内部的横向轴和竖直轴，否则会出现数学上朝向球路、画面中仍然没有完全横开的情况。

T1 参照坐标系：

```text
T1 right_hand_link:
  face normal: axis_+X_w=[0.8648, 0.4552, -0.2118]
  in-plane horizontal: axis_-Y_w=[0.4651, -0.8853, -0.0039]
  in-plane vertical: axis_+Z_w=[0.1892, 0.0952, 0.9773]
  incoming alignment=0.9049
```

A3 新目标是把 `right_hand_pingpang_Link` 的局部坐标系整体贴近 T1：

```text
A3 candidate:
  right_wrist_roll_joint=-0.05
  right_wrist_pitch_joint=-0.33
  right_wrist_yaw_joint=-1.10

  face normal: paddle_face_+y_w=[0.8624, 0.4538, -0.2245]
  in-plane horizontal: paddle_local_x_w=[0.4641, -0.8857, -0.0074]
  in-plane vertical: paddle_local_z_w=[0.2022, 0.0978, 0.9745]
  incoming alignment=0.9056
  paddle_to_future_dist=0.2273 m
```

对比判断：

- A3 `+Y` 几乎等于 T1 `+X`，说明拍面正对来球；
- A3 `+X` 几乎等于 T1 `-Y`，说明拍面横向展开方式一致；
- A3 `+Z` 接近 T1 `+Z`，说明拍面竖直方向一致；
- 这版更符合用户说的“完整一个面正对球来的方向”，比上一版 `right_wrist_roll=-0.25` 更适合作为训练初始持拍姿态。

边界：

- 只修改 A3 默认右腕姿态；
- 不修改 T1；
- 不修改 TTEnv、奖励、球路、训练 workflow；
- 修改后必须跑 A3 headless 诊断和短步稳定性验证。

## 2026-06-26 A3 击球中心初始几何对齐计划

用户确认下一步先优化 A3 初始姿态和初始站位，让击球中心更接近任务球路。当前问题不是球拍面背对球，而是 A3 reset 后的 `paddle_touch_point` 和 `ball_future_pose` 空间差距偏大，导致 `reward_contact` 和 `reward_table_success` 极稀疏。

当前 deterministic 诊断结果：

```text
ball_future_pose(env)=[-1.8742, -0.1236, 1.0862]
paddle_touch_point(env)=[-1.6863, -0.0560, 0.9774]
future_minus_paddle=[-0.1879, -0.0676, 0.1087]
future_to_paddle_dist=0.2273 m

first-serve static closest distance to initial paddle center=0.1063 m
contact threshold ~=0.0700 m
```

调整策略：

1. 水平误差主要通过 A3 reset base 平移修正：
   - `base_x` 往负方向移动，使击球中心更靠近预测击球点的 x；
   - `base_y` 往负方向移动，使击球中心更靠近预测击球点的 y。
2. 垂直误差主要通过右臂默认姿态修正：
   - 小幅调整 `right_shoulder_pitch/roll/yaw`、`right_elbow` 和 wrist pitch；
   - 目标是把 `paddle_touch_point.z` 上移约 8-11 cm；
   - 保持拍面法向仍然朝向来球，避免为抬高击球中心破坏上一阶段的 T1 拍面坐标系对齐。
3. 搜索判据：
   - `paddle_to_future_dist` 优先小于 0.10 m，理想小于 0.07 m；
   - 首发球路到初始击球中心的最近距离小于接触阈值或接近接触阈值；
   - `face_alignment` 不能明显低于 T1 参照，目标保持约 0.85 以上；
   - 站立稳定性短步验证不能产生 reset。

边界：

- 只修改 A3 默认姿态和 A3 reset base 范围；
- 不修改 T1；
- 不修改球路和奖励函数，先把初始几何对齐做好；
- 修改前后都记录在 A3 迁移日志中。

## 2026-06-26 A3 最新训练问题与托底机制分析

用户最新一次 4096 env、10000 iter 训练后，TensorBoard 曲线表现为：

- `reward_future_dis_ee` 有一定学习迹象；
- `reward_contact`、`TT_hit_rate`、`reward_table_success` 长期接近 0；
- 机器人没有形成稳定的乒乓球击球策略。

结合代码流程判断，这不是单纯训练步数不足，而是任务奖励链条在 A3 初始几何下被切断：

1. `TTEnv.compute_intermediate_values()` 根据当前球位置、速度和反弹模型计算 `ball_future_pose`。
2. `reward_future_ee_target()` 奖励 `paddle_touch_point` 靠近 `ball_future_pose`，这是密集 shaping。
3. `compute_paddle_touch()` 只有在球心到 `paddle_touch_point` 的距离小于接触阈值时，才会产生 `ball_contact_rew`。
4. `reward_contact()` 直接读取 `ball_contact_rew`，因此它本质上仍然是接触附近才有效的稀疏奖励。
5. `reward_table_success()` 还要求 `has_touch_paddle` 后球落到对侧台面，因此比 `reward_contact` 更稀疏。

因此，A3 如果初始击球中心和球路相差 10-20 cm 以上，策略虽然可能先学到“往未来点靠近一点”，但很难自然跨过真实接触阈值；一旦没有真实接触，后续过网、落台相关奖励也不会启动。

PD 控制器作用：

- 代码中的底层控制不是外部悬挂或固定机器人，而是 `DelayedPDActuatorCfg` 对每个关节执行位置 PD；
- actor 输出 action 后，经 `processed_actions = action * action_scale + default_joint_pos` 转成每个关节的位置目标；
- PD 用 stiffness/damping、effort limit、velocity limit 把关节拉向这个目标；
- 这相当于真实机器人低层伺服控制的仿真近似，不是“仿真作弊”。

sim-to-real 影响：

- 如果 PD 参数接近真实电机和控制器，sim-to-real 更可信；
- 如果为了仿真站稳而把 stiffness/damping 或力矩限幅调得远强于真实硬件，策略会依赖真实机器人做不到的关节响应，sim-to-real 风险会明显变大；
- 当前 A3 已经用用户提供的电机惯量、力矩、速度信息设置了 effort/velocity/armature，后续 PD 只做小范围保守调参，不能把它当成固定身体的外力。

reset/策略随机范围权衡：

- 缩小 reset 随机范围可以让早期状态集中在“能站住、球能经过拍面附近”的区域，减少奖励稀疏；
- 如果球路和击球中心本身差距很远，缩小随机范围反而会把策略困在错误初始几何附近；
- 过大随机范围会让未训练策略频繁站不稳、摔倒、越界，episode 很短，学习信号被终止惩罚和姿态惩罚淹没；
- 所以正确顺序是：先用默认几何把 `paddle_touch_point` 对齐到代表性 `ball_future_pose` 附近，再从小范围 reset 开始训练，等 `reward_contact/TT_hit_rate` 有稳定非零信号后再逐步扩大球路和 reset 范围。

本轮落地准则：

1. `base_x/base_y` 负责水平平移，让击球中心的 x/y 靠近 `ball_future_pose`；
2. 右臂默认姿态负责把击球中心 z 上移，避免为了升高球路而修改任务本身；
3. 保持 A3 球拍完整坐标系近似 T1：拍面朝向来球、拍面横向和竖直方向不反；
4. 暂不改 reward 和球路，先修复 A3 初始几何托底；
5. 如果修复后 `reward_future_dis_ee` 改善但 `reward_contact` 仍然长期为 0，再考虑 staged reward/curriculum，而不是一开始就重写奖励。

## 2026-06-26 A3 击球几何与站立托底联合选型

复测发现，当前 A3 默认姿态即使 `reset_count=0`，零动作下 500-1000 step 也会明显下塌：

```text
current default, base_x=-0.40, base_y=0.35:
  step=1:
    paddle_to_future_dist=0.2273 m
    paddle_touch_point.z=0.9774
  step=500:
    base_rpy ~= [0.1993, 0.5391, 0.1480]
    paddle_touch_point.z=0.4525
  step=1000:
    base_rpy ~= [0.1993, 0.5390, 0.1479]
    paddle_touch_point.z=0.4525
```

因此上一轮训练的根因不只是球拍初始点偏离球路，还包括默认站立姿态在早期会把击球中心高度带崩。继续把 `base_x` 往负方向平移虽然能瞬时贴近 `ball_future_pose`，但会显著增加 reset 或姿态塌陷。

本轮搜索结论：

- 激进几何候选：
  - `base_x=-0.62`
  - `right_shoulder_pitch=0.0`
  - `right_elbow=0.30`
  - `right_wrist_pitch=-0.35`
  - `right_wrist_yaw=-1.20`
  - step 1 `paddle_to_future_dist=0.0228 m`
  - 2000 step `reset_count=18`
  - 结论：几何非常好，但站立失败，不能落地。
- 保持旧右臂、只后移 base：
  - `base_x=-0.55`
  - step 1 `paddle_to_future_dist=0.1165 m`
  - 1000 step `reset_count=8`
  - 结论：base 后移过大也不稳。
- 新联合候选：
  - `candidate 547`
  - `base_x=-0.26`
  - `base_y=0.35`
  - `waist_pitch=-0.04`
  - `hip_pitch=-0.05`
  - `knee=0.50`
  - `ankle_pitch=-0.22`
  - `left_hip_roll/right_hip_roll=+0.16/-0.16`
  - `left_ankle_roll/right_ankle_roll=-0.072/+0.072`

2000 step 验证：

```text
logs/a3_standing_calibration/a3_candidate547_basex026_2000.csv

survival_steps=2000/2000
reset_seen=0
final_root_z=0.843224
max_abs_roll_pitch=0.344542
max_root_z_drift=0.237392
max_foot_slip=0.044406
both_feet_contact_ratio=1.000000
min_paddle_future_dist=0.078336
final_roll=-0.090677
final_pitch=0.324231
```

选择：

- 落地 `candidate 547` 的腿/腰默认姿态；
- A3 reset base 从 `x ~= -0.40` 前移到 `x ~= -0.26` 附近；
- 不改 T1；
- 不改公共 `TTEnv`；
- 不改球路和 reward；
- PD 暂不改，因为中等增强 PD 没有改善姿态，反而增加姿态冲击。

预期影响：

- 初始击球中心不再追求一步贴到 `ball_future_pose`，而是先让默认站姿保持击球中心可用；
- 早期训练更容易在球路附近产生 `reward_contact`；
- reset 范围先缩小，等 `TT_hit_rate/reward_contact` 稳定非零后再扩大。

补充复测：

- `a3_standing_calibration.py` 为了做站立搜索会把球速置 0，因此其中的 `min_paddle_future_dist` 只能辅助判断姿态相对目标点的几何趋势，不能替代真实乒乓球路诊断；
- 真实球路仍以 `a3_pose_calibration.py` 为准。

真实球路复测结果：

```text
new default, base_x=-0.26, base_y=0.35:
  step=1:
    paddle_to_future_dist=0.3514 m
    face_alignment=0.9055
    paddle_touch_point.z=0.9841
  step=1000:
    reset_count=0
    base_rpy ~= [0.0230, 0.3402, -0.0452]
    paddle_touch_point.z=0.6265

new default, base_x=-0.28:
  step=1:
    paddle_to_future_dist=0.3327 m
  step=1000:
    reset_count=15

new default, base_x=-0.30:
  step=1:
    paddle_to_future_dist=0.3142 m
  step=1000:
    reset_count=17

new default, base_x=-0.34:
  step=1:
    paddle_to_future_dist=0.2778 m
  step=1000:
    reset_count=18
```

最终本轮落地边界：

- A3 train/eval reset base 收窄到 `x ~= -0.26`；
- 这是站稳优先的保守起点；
- 当前几何仍不能认为已经完全解决，下一步若 `reward_contact` 仍低，需要做 A3 专用 staged reward/curriculum 或继续搜索“稳定右臂前伸”姿态，而不能再单纯后移 base。

## 2026-06-26 A3 早期学习托底优化计划

最新训练现象：

- A3 训练稳定运行，但 `reward_contact/table_success` 仍然难以起来；
- 站立姿态已经比上一版稳定，但真实默认球路下，稳定姿态 step 1 的 `paddle_to_future_dist` 仍约 `0.35 m`；
- 若继续把 A3 base 后移来追击球点，姿态会明显不稳并频繁 reset；
- 因此本轮不再强行用几何平移解决所有误差，而是先给 A3 早期训练提供更连续、更可达的学习信号。

关键复测：

```text
current stable A3 default, base_x=-0.26, base_y=0.35:
  original/eval-like serve:
    paddle_to_future_dist ~= 0.3514 m
    face_alignment ~= 0.9055

  easier centered serve, vx=-5.0, vy=-0.05, vz=1.45:
    ball_future_pose ~= [-1.6000, -0.0287, 1.0570]
    paddle_touch_point ~= [-1.5448, -0.0560, 0.9841]
    paddle_to_future_dist ~= 0.0955 m
    face_alignment ~= 0.8954
```

分析：

- 球拍模型和击球中心方向基本可用，球拍没有明显拿反；
- 主要问题是 A3 的稳定默认几何与 T1 原始训练球路不完全匹配，早期随机策略很难在不摔倒的同时把球拍移动到接触区；
- 稀疏的 `reward_contact` 和 `reward_table_success` 在早期很可能接近 0，PPO 主要优化“不摔倒/少动/少碰撞”，而不是乒乓球任务；
- 仅扩大 action/noise 会增加探索，但也会显著增加倒地概率；
- 仅缩小随机范围能帮助站稳，但如果球路太远，会进一步减少接触探索。

本轮修改边界：

- 只修改 A3 专用 `a3_tt` 配置；
- 不修改 T1；
- 不修改公共 `TTEnv` 逻辑；
- 不修改球拍 USD/URDF 资产；
- eval 仍保留接近原始 T1 的较宽球路，便于和旧训练对照；
- train 先使用 A3 early-stage friendly serve，让早期产生更稳定的近球拍信号。

拟落地修改：

1. A3 train 球路收窄/放慢：
   - `ball_speed_x_range=(-5.2, -4.8)`
   - `ball_speed_y_range=(-0.10, 0.02)`
   - `ball_speed_z_range=(1.40, 1.60)`
   - `ball_pos_y_range=(-0.03, 0.03)`
2. A3 train contact 区域轻微放宽：
   - `contact_threshold=0.07`
   - 目的不是改变物理碰撞，而是让早期 `reward_contact` 不至于过度稀疏；
   - eval 保持 `contact_threshold=0.05`。
3. A3 reward 增加密集的球-拍距离 shaping：
   - 使用已有 `mdp.reward_paddle_distance_terminal`；
   - 新 reward 只放在 A3，不影响 T1；
   - 目标是在真正接触前也让策略知道“球拍靠近球”是正方向。
4. 提高 A3 `reward_future_dis_ee` 权重：
   - 从 `2.0` 提到 `6.0`；
   - 让球拍靠近 `ball_future_pose` 的信号比上一版更明显。
5. 降低 A3 初始策略噪声：
   - `init_noise_std=0.5 -> 0.35`；
   - 配合已缩小的 reset 范围，降低早期摔倒和大幅乱动。
6. 缩小 A3 reset 初速度 yaw：
   - `yaw=(-0.05, 0.05) -> (-0.02, 0.02)`；
   - 避免初始 yaw 角速度直接破坏当前窄稳定姿态。
7. A3 train 每个 robot episode 的 serve 数减少：
   - `max_serve_per_episode=3`；
   - 如果机器人已经进入明显无效姿态，可以更快 reset 回默认姿态，而不是继续打满 5 个球。

预期训练观察：

- 前 250-500 iter 应优先观察：
  - `Episode_Reward/reward_paddle_ball_dense` 是否有稳定非零值；
  - `Episode_Reward/reward_future_dis_ee` 是否上升；
  - `Episode_Reward/reward_contact` 是否开始偶发非零；
  - `Episode_Termination/base_contact` 或 reset 相关项是否显著下降；
  - `Episode_Reward/reward_table_success` 早期仍可能很低，不作为唯一判断。
- 如果 dense/future reward 能上升但 contact 仍为 0，下一轮应继续微调右臂/球路；
- 如果站立项变差，优先回退 serve 难度或进一步降低 action/noise，不先改 T1-style 稀疏成功奖励。

### 2026-06-26 实时监督反馈与修正计划

用户启动：

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

实时 TensorBoard/event 读取到约第 85 iter：

```text
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

判断：

- 训练进程没有 NaN，也没有直接崩溃；
- 但是新增的 `reward_paddle_ball_dense` 实际几乎为 0，说明它没有给策略提供预期的早期 dense shaping；
- 根因是 `reward_paddle_distance_terminal` 使用“真实球当前位置到球拍击球中心”的距离：
  - 球刚发出时离球拍约数米，reward 极小；
  - 球接近击球区域时，又容易被 `mask_terminal` 或时序窗口过滤；
  - 所以它不适合作为 A3 当前阶段的早期几何引导。

已执行：

- 按用户“有问题可以停止”的授权停止 PID `1765130`；
- TensorBoard 继续保留在 `16006`。

修正方案：

- 新增一个更贴合当前问题的 reward：
  - 输入：`env.ball_future_pose` 和 `env.paddle_touch_point`
  - 含义：预测击球点到球拍击球中心的距离；
  - mask：沿用 `env.mask_invalid`，避免无效球路奖励；
  - 目标：让策略在真正接触前，就收到“把球拍击球中心移到预测击球点”的连续信号。
- A3 配置中将原 `reward_paddle_ball_dense` 从 actual ball distance 切换到 future touch-point distance；
- 保留 `reward_future_dis_ee`，因为它在第 85 iter 已经有非零信号，但原函数使用的是 `paddle_pos`，不是标定后的 `paddle_touch_point`，所以还需要新的 touch-point reward 补齐。

### 2026-06-26 二次实时监督反馈：坐标系修正

用户重新启动：

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

判断：

- 训练无 NaN，但新的 `reward_future_touch_point` 仍然几乎为 0；
- 这不是 PPO 学习慢，而是 reward 计算坐标系错误。

原因：

- `TTEnv.compute_paddle_touch()` 里 `env.paddle_touch_point` 是 world frame；
- `TTEnv.compute_intermediate_values()` 里 `env.ball_future_pose` 是 env-local frame；
- `a3_pose_calibration.py` 打印几何诊断时使用的是：

```python
paddle_env = env.paddle_touch_point - env.scene.env_origins
future_env = env.ball_future_pose
```

- 但新增 reward 第一版直接计算：

```python
env.ball_future_pose - env.paddle_touch_point
```

- 在 4096 并行环境中，除 0 号环境外都会混入 env origin 偏移，reward 因此接近 0。

修正：

- `reward_future_touch_point_target` 内部必须先把 `paddle_touch_point` 转回 env-local：

```python
paddle_touch_point = env.paddle_touch_point - env.scene.env_origins
distance = norm(env.ball_future_pose - paddle_touch_point)
```

- 继续保留 `env.mask_invalid` 过滤无效球路。

### 2026-06-26 第三次实时监督反馈：早期动作扰动过大

坐标系修复后重新启动：

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
Episode_Reward/reward_future_touch_point:
  ~= 0.11，稳定非零

Episode_Reward/reward_future_dis_ee:
  ~= 0.16，稳定非零

Episode_Reward/reward_contact:
  0

Episode_Reward/reward_table_success:
  0

Episode_Reward/termination_penalty:
  ~= -2.0

Train/mean_episode_length:
  ~= 52 steps
```

判断：

- 坐标系修复成功；
- 但 4096 env 初期探索仍会让 A3 很快 reset；
- episode 长度稳定在约 52 step，说明策略没有足够长时间留在可击球窗口；
- 继续跑 500 iter 会主要学习“短 episode 下少受罚”，而不是学习接球。

第三阶段修正方向：

- 降低 A3 初始策略动作幅度：
  - `A3_INITIAL_ACTION_SCALE: 0.15 -> 0.08`
  - `A3_INITIAL_POLICY_NOISE_STD: 0.35 -> 0.20`
- 缩小 reset 初速度：
  - base linear/roll/pitch/yaw velocity 扰动进一步收窄；
- 缩小 reset 关节扰动：
  - locomotion joints 从 `(0.95, 1.05)` 改到 `(0.98, 1.02)`；
  - right-arm manipulation joints 从 `(-0.02, 0.02)` 改到 `(-0.01, 0.01)`；
- 提高 `reward_future_touch_point` 权重：
  - `4.0 -> 8.0`
  - 让早期“击球中心靠近预测击球点”的梯度更强。

预期：

- 前 50 iter 的 `Train/mean_episode_length` 应明显高于 52；
- `termination_penalty` 应低于当前绝对值；
- `reward_future_touch_point` 应保持非零；
- `reward_contact` 可以仍然延迟出现，但不能在 episode 太短的情况下继续训练。

### 2026-06-26 第四次实时监督反馈：低动作正式 run 仍非 timeout reset

用户已启动/接管：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_low_action_500 \
  --headless \
  --predictor
```

监督到第 73 iter：

```text
Train/mean_episode_length:
  last ~= 52.29

Episode_Reward/reward_future_touch_point:
  last ~= 0.2207

Episode_Reward/reward_future_dis_ee:
  last ~= 0.1586

Episode_Reward/reward_contact:
  0

Episode_Reward/reward_table_success:
  0

Episode_Reward/termination_penalty:
  last=-2.0

Episode_Reward/undesired_contacts:
  last ~= -0.0043
```

补充确认：

- `mdp.is_terminated(env)` 返回 `env.reset_buf * ~env.time_out_buf`；
- 当前 `termination_penalty=-2.0` 说明主要是非 timeout reset；
- `TTEnv.check_reset()` 非 timeout 条件只有：
  - `robot_pos.z < 0.50`
  - `robot_pos.x < -3.6`
  - `robot_pos.x > -1.35`
  - `robot_pos.y < -1.1`
  - `robot_pos.y > 1.1`
- 当前 `undesired_contacts` 已接近 0，因此正式 run 的主要问题不再是非足部碰撞，而是根状态越界/掉高导致 robot episode 太短。

在“初始姿态和球路几何可用”的前提下，本轮判断：

- touch-point reward 已有有效信号；
- A3 仍没有足够长的 episode 留在击球窗口内；
- 继续训练会强化“短 episode 下的局部稳定”，难以产生真实 contact。

第四阶段修正方向：

- 不再改球路和初始姿态；
- 进一步降低 early-stage 动作扰动：
  - `A3_INITIAL_ACTION_SCALE: 0.08 -> 0.04`
  - `A3_INITIAL_POLICY_NOISE_STD: 0.20 -> 0.10`
- 增加 A3 腿部默认姿态保持项：
  - 使用已有 `mdp.joint_deviation_l1`
  - 作用于 `.*hip_.*_joint`, `.*knee_joint`, `.*ankle_.*_joint`
  - 目标是减少随机策略早期把腿部支撑姿态打散；
- 轻微增强躯干姿态约束：
  - `flat_orientation_l2: -1.5 -> -3.0`

预期：

- 前 50-100 iter 的 `mean_episode_length` 应显著高于 52；
- `termination_penalty` 不能继续卡在 -2；
- `reward_future_touch_point` 仍应非零；
- 如果这样仍无 contact，再考虑放宽 A3 early-stage 的 reset 边界或做分阶段任务，而不是继续压低 action。

### 2026-06-26 零动作/固定根节点诊断：问题收敛到 A3 控制保持

为了遵守“初始姿态和球路几何可用”的前提，本轮没有继续改球路，而是做两个诊断：

1. A3 默认姿态、当前 easy serve、零动作、自由 root。
2. 同样设置，但使用 `--pin_base` 固定 root，仅用于诊断。

自由 root 诊断关键点：

```text
step 1:
  paddle_to_future_dist ~= 0.0955 m

step 20:
  paddle_to_future_dist ~= 0.0746 m
  paddle_to_ball_dist ~= 1.0559 m

step 40:
  paddle_to_ball_dist ~= 0.3257 m
  paddle_touch_point.z ~= 0.6499
  ball_pos.z ~= 0.9077

step 90:
  paddle_to_ball_dist ~= 0.3883 m
```

固定 root 诊断关键点：

```text
step 1:
  paddle_to_future_dist ~= 0.0955 m

step 30:
  paddle_to_future_dist ~= 0.0920 m
  paddle_to_ball_dist ~= 0.4766 m

step 40:
  paddle_to_ball_dist ~= 0.0844 m
  paddle_touch_point.z ~= 0.9843
  ball_pos.z ~= 0.8969

step 60:
  paddle_to_future_dist ~= 0.0903 m
```

结论：

- 初始一帧几何确实可用；
- 如果固定 root，实际球路已经接近 contact threshold，只差约 1-2 cm；
- 自由 root 下 A3 在来球窗口内明显下沉/前移/俯仰，导致球拍击球中心从可用球路掉开；
- 因此当前 `reward_contact=0` 的主因不应再归结为球路或初始姿态，而是 A3 的控制保持/PD/默认姿态稳定域没有达到 T1 的水平。

与 T1 的边界关系：

- T1 原始流程早期也有明显 `termination_penalty`，但仍能偶发 contact；
- T1 使用 `ImplicitActuatorCfg`，腿部 stiffness 约 200，arms damping 约 10；
- A3 当前使用 `DelayedPDActuatorCfg`，并设置 `min_delay=0,max_delay=3`，腿部 stiffness 120，arms damping 4；
- 这属于 A3 actuator/control 适配差异，不是乒乓球任务逻辑差异。

下一步最小修改原则：

- 不改 T1；
- 不改公共 `TTEnv`；
- 不改球路；
- 不放宽成功标准；
- 只做 A3 asset 级别的控制保持适配，并每次只验证一个假设。

拟先验证的最小 A3 控制修改：

- 将 A3 table-tennis 初期 actuator delay 从 `0-3 step` 改为 `0 step`；
- 修改范围只限 `A3_T2D5_PINGPANG_CFG`，不改裸 `A3_T2D5_CFG`；
- 理由：T1 原流程没有显式 delayed actuator，先让 A3 对齐 T1 的控制响应假设；
- 暂不同时大幅提高 stiffness/damping，避免多个变量叠加。

验证标准：

- 零动作自由 root 诊断中，step 40 附近 `paddle_to_ball_dist` 应明显低于当前 `0.3257 m`；
- `paddle_touch_point.z` 不应从约 `0.98` 掉到 `0.65`；
- 若无改善，再考虑 A3-only PD gain 调整。

验证结果：

- 将 `A3_T2D5_PINGPANG_CFG` 的 actuator delay 临时固定为 0 后，step 40 仍为：
  - `paddle_to_ball_dist ~= 0.3249 m`
  - `paddle_touch_point.z ~= 0.6510`
- 与未改 delay 的 `0.3257 m / 0.6499 m` 基本一致。

结论：

- actuator delay 假设不成立，临时代码改动已撤回；
- 下一步不继续沿 delay 方向修改；
- 需要检查 A3 默认站姿力学平衡和 PD gain。

### 2026-06-26 A3 当前持拍右臂下的下肢/PD 搜索

继续遵守边界：

- 不改 T1；
- 不改公共 `TTEnv`；
- 不改球路；
- 不改核心成功 reward；
- 只搜索 A3 pingpong reset pose 和 A3 pingpong actuator damping。

诊断命令使用 `a3_standing_calibration`，右臂采用 `default_arms`，含义是保留当前 A3 配置里的持拍右臂和球拍姿态，只搜索下肢/腰部站姿：

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

当前最佳候选：

```text
candidate_index=69
name=custom_hp-0.18_kn+0.50_ap-0.26_default_arms_z1.05_waist-0.04_w-0.16_toe+0.00
survival_steps=500/500
reset_seen=0
max_abs_roll_pitch=0.4123
max_root_z_drift=0.2003
max_foot_slip=0.0620
min_paddle_future_dist=0.0782
```

候选含义：

```text
root_z = 1.055
waist_pitch = -0.04
hip_pitch = -0.18
knee = 0.50
ankle_pitch = -0.26
left_hip_roll = -0.16
right_hip_roll = 0.16
left_ankle_roll = 0.072
right_ankle_roll = -0.072
右臂/球拍 = 当前 A3 pingpong default 持拍姿态
waist damping x2
leg damping x3
feet damping x3
```

与当前配置的关键差异：

- 当前 A3 pingpong 下肢是相反的 hip-roll/ankle-roll 方向；
- 当前 A3 pingpong 的 `hip_pitch=-0.05, ankle_pitch=-0.22`，候选更像可承重的屈膝下蹲；
- 单纯 actuator delay 已证明无效，本候选改善来自默认力学平衡和 damping，而不是任务流程。

拟落地的最小代码修改：

- 只在 `A3_T2D5_PINGPANG_CFG` 上覆盖 init_state 和 waist/legs/feet damping；
- 保留裸 `A3_T2D5_CFG`；
- 不改变 A3 右臂持拍姿态、不改变 paddle offset；
- 落地后复跑 moving-ball zero-action 诊断，目标是 step 40 附近 `paddle_to_ball_dist` 明显低于 `0.325 m`。

落地验证结果：

```text
moving-ball zero-action:
  before step40 paddle_to_ball_dist ~= 0.3257 m
  after  step40 paddle_to_ball_dist ~= 0.0714 m

default zero-action standing:
  survival_steps=500/500
  reset_seen=0

short train probe:
  reward_contact max ~= 0.0433
  reward_contact mean ~= 0.00445
```

这说明 A3 已经从“基本碰不到球”进入“可以偶发接触球”的阶段。

新的待处理问题：

```text
A3 candidate69 undesired_contacts mean ~= -42.42
T1 original first 30 iter undesired_contacts mean ~= -3.03
```

下一步边界：

- 不直接删除/降权 `undesired_contacts`，因为 T1 原始流程也依赖这个托底项；
- 先定位 A3 是哪些 body 在发生非足部接触；
- 如果是 A3 合理支撑 body 被误罚，才做 A3-only body pattern 修正；
- 如果是真摔碰/擦桌，则优先调 A3 默认姿态、action/noise 或 reset 范围。

### 2026-06-26 迁移边界重申：以 T1 原始流程为基准

用户提醒：

- T1 使用原始 PACE/LeggedLab table-tennis workflow 是可以训练和评估的；
- A3 替换 T1 后必然有机器人几何、动力学、关节命名、PD/actuator、reset 稳定域差异；
- 但迁移不能变成随意修改任务流程，不能为了让 A3 某个指标变好而把原始乒乓球任务改成另一个任务。

后续硬边界：

1. 不修改 T1 任务、T1 asset、T1 eval/train workflow。
2. 不修改公共 `TTEnv` 行为，除非先证明公共逻辑本身有坐标系/接口 bug，并且确认 T1 不受影响。
3. 不放宽或删除原任务的核心成功标准：
   - `reward_contact`
   - `reward_table_success`
   - `reward_future_landing_dis`
   - `reward_future_pass_net`
4. 不把球路无限简化到脱离乒乓球任务；A3 early-stage easy serve 只能作为课程起点，后续必须逐步回到 T1-like serve range。
5. 不通过删除 reset/termination 来掩盖 A3 摔倒问题；可以做 A3-only early-stage 稳定域调参，但必须记录目的、验证指标和回退条件。
6. A3 专用 reward shaping 只能作为引导项，不能替代最终 `contact/table_success`。
7. 每次只能改一个主要假设：
   - 站立/控制稳定
   - 球拍击球中心对齐
   - reward 稀疏性
   - reset/episode 时长
   - PD/actuator 参数

当前前提：

- A3 初始姿态和当前 easy 球路几何先视为可用；
- `reward_future_touch_point` 已经证明坐标系修复后能稳定非零；
- 当前主要问题暂定为：A3 在 PPO early-stage 随机策略下过早触发非 timeout reset，导致真实 contact 没有机会出现；
- 但是第四阶段“继续降低 action/noise、增加腿部默认姿态保持”的方案暂缓直接落地，需要先确认这是否仍属于 A3 控制适配，而不是偏离原始 T1 workflow。

重新评估顺序：

1. 对比 T1 原始训练 early-stage 的 reset/episode length/contact 出现时机；
2. 明确 A3 当前 `termination_penalty=-2` 对应的 root 高度还是 x/y 越界；
3. 如果是 root 高度/越界，由 A3 控制稳定问题处理；
4. 如果 episode 结束主要来自 ball serve/time out，则优先调 A3 early-stage serve schedule；
5. 在没有上述证据前，不继续增加新 reward 或放宽任务成功条件。

### 2026-06-26 监督训练后的 A3-only 调整计划

已完成的监督训练和诊断说明：

```text
A3 candidate69, 4096 env, supervised run:
  reward_contact last10_mean ~= 0.0238
  reward_table_success = 0
  undesired_contacts last10_mean ~= -49.9
```

当前问题拆分：

1. A3 已经能把球带到球拍/击球点附近，不再是最初的“完全碰不到球”。
2. 球触拍后没有稳定变成向对面桌飞行，说明拍面/右腕/右臂姿态和击球后的稳定性仍不够。
3. A3 的球拍 attachment link 被 contact sensor 当作 robot body 计入 `undesired_contacts`：
   - `pingpang_red_Link`
   - `pingpang_black_Link`
   - `right_hand_pingpang_Link`
   - `pingbang_ball_Link`
4. 这些 link 的接触不同于 torso/pelvis 撞地或撞桌；球拍和击球点 marker 与球接触是任务期望行为。
5. `torso_Link`、`pelvis_link`、普通手臂/手腕的大量接触仍然是真实失稳信号，不能排除。

拟进行的最小代码修改：

```text
只修改 legged_lab/envs/a3_tt/a3_tt_config.py

把 A3 reward.undesired_contacts 的 body_names regex 从：
  排除左右 ankle_roll_Link

改成：
  排除左右 ankle_roll_Link
  排除 A3 球拍/击球点 attachment link

不修改：
  T1 配置
  公共 TTEnv
  reward_contact
  reward_table_success
  ball route
  torso/pelvis termination 或碰撞惩罚
```

验证标准：

1. `reward_contact` 不应下降到 0。
2. `undesired_contacts` 应下降，但不能变成 0；如果 torso/pelvis 仍接触，仍应被罚。
3. `reward_table_success` 若仍为 0，则下一步继续调右腕拍面/右臂姿态或 staged reward，而不是继续删接触惩罚。
4. 与 T1 相关文件无 diff。

验证结果：

```text
接触诊断：
  old_nonfoot_count_total = 45746
  new_undesired_count_total = 35724
  reduction = 10022

短训练 64 env / 30 iter:
  reward_contact mean ~= 0.00274
  reward_table_success = 0
  undesired_contacts mean ~= -42.24
  termination_penalty mean ~= -0.4319
```

结论：

- 球拍/击球点白名单生效，且没有把 torso/pelvis/hand/wrist 的真实失稳接触移除；
- 但训练主问题没有解决，因为 `undesired_contacts` 的大头仍来自 torso/pelvis/hand/wrist；
- 下一步不继续扩大 contact 白名单；
- 下一步应优先处理 A3 右腕/拍面/右臂默认姿态与击球后稳定性，让球触拍后具备向对面桌反弹的速度方向。

### 2026-06-26 A3 右腕/拍面默认姿态调整计划

当前诊断结论：

```text
当前 A3 默认右腕:
  right_wrist_roll  = -0.05
  right_wrist_pitch = -0.33
  right_wrist_yaw   = -1.10

step 40 触球窗口:
  拍面主要法向 x 分量不足；
  球触拍后 ball_vx 不能稳定变为正值；
  reward_table_success 仍为 0。
```

批量 wrist scan 结果：

```text
125 个右腕候选中，43 个候选可让触球后 vx3 > 0。

较优候选 scan101:
  right_wrist_roll  = +0.25
  right_wrist_pitch = -0.75
  right_wrist_yaw   = -1.40
  touch_step = 41
  minD ~= 0.0750
  vx3 ~= +1.237
  vz3 ~= +0.135
  x20 ~= -1.224
  z20 ~= +0.894
  vx20 ~= +0.644
  reset = False
```

拟修改：

```text
只修改 legged_lab/assets/a3/a3.py

在 A3_PINGPONG_READY_JOINT_POS 里覆盖 A3 pingpong 专用右腕：
  right_wrist_roll_joint  -> +0.25
  right_wrist_pitch_joint -> -0.75
  right_wrist_yaw_joint   -> -1.40
```

边界：

- 不修改裸 `A3_T2D5_CFG` 的初始姿态；
- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改球路、成功 reward 或 contact threshold；
- 如果落地后 `reward_contact` 下降到 0，或站立/接触更差，应回退该右腕候选。

验证：

1. `py_compile`；
2. 固定 easy serve 的 zero-action / hold-pose 诊断：
   - 触球后 `ball_vx` 应稳定为正；
   - 不能明显增加 torso/pelvis reset；
3. 64 env / 30 iter 短训练探针：
   - `reward_contact` 不应变 0；
   - `reward_table_success` 若仍为 0，则继续调右臂/拍面或 staged reward，而不是继续删 contact 惩罚。

验证结果：

```text
固定 easy serve:
  touched = 128/128
  first_touch_step = 41
  reset_events = 0
  vx3 mean ~= +1.2360
  vz3 mean ~= +0.1869
  vx20 mean ~= +0.5850
  z20 mean ~= +0.8953

64 env / 30 iter probe:
  reward_contact mean ~= 0.03989
  reward_future_landing_dis mean ~= 0.01778
  reward_table_success = 0
  undesired_contacts mean ~= -45.97
  termination_penalty mean ~= -0.2583
```

阶段结论：

- scan101 右腕姿态应保留，它明显改善了触球和回球方向的前置信号；
- 它还没有解决 `reward_table_success=0`，因为球仍未稳定落到对面桌；
- 剩余主要瓶颈变为右臂/身体稳定性和主动挥拍学习，而不是继续旋转拍面或删除 contact 惩罚。

## 2026-06-26 scan101 后早期探索约束计划

背景：

- scan101 固定球路诊断中，默认姿态已经可以让球稳定触拍，并让触球后的 `ball_vx` 变为朝对面桌方向的正值；
- 64 env / 30 iter 短训中，`reward_contact` 和 `reward_future_landing_dis` 明显高于 candidate69，但 `reward_table_success` 仍为 0；
- 进一步接触来源诊断显示，小随机动作下 first touch 约在 step 39，first undesired contact 约在 step 63，主要来源为 `pelvis_link`、`torso_Link`、`left_hand_Link`、`right_wrist_*_Link` 等。

判断：

- 当前几何不再是“完全够不到球”的状态；
- 问题更像是 A3 初期策略探索过大，触球后身体和双臂迅速进入不稳定接触，导致 episode/reward 被摔倒和身体碰撞主导；
- 这一步不应删除 torso/pelvis/arm 的 undesired contact 惩罚，也不应放宽 table_success，因为这些是保证任务语义的约束；
- 比较安全的 A3-only 调整是先缩小 PPO 初始动作噪声，而不是改 T1、改公共 TTEnv 或重写 reward。

拟修改：

```text
只修改 legged_lab/envs/a3_tt/a3_tt_config.py

A3_INITIAL_POLICY_NOISE_STD:
  0.20 -> 0.15
```

边界：

- 不修改 `A3_INITIAL_ACTION_SCALE = 0.08`；
- 不修改 scan101 右腕默认姿态；
- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改 `reward_contact`、`reward_table_success` 或 ball success 判定；
- 不把 pelvis/torso/arms 加入 allowed contact whitelist。

验证：

1. `py_compile`；
2. 64 env / 30 iter 短训练探针：
   - `reward_contact` 不应回到接近 0；
   - `undesired_contacts` 和 `termination_penalty` 应尽量低于 scan101 noise=0.20 探针；
   - `reward_table_success` 允许仍为 0，因为 30 iter 只用于验证早期稳定性和触球信号；
3. 若触球信号明显下降，则回退该噪声改动；若触球保留但 undesired contact 仍高，再考虑 A3-only 的右臂姿态/右臂正则或 staged reward。

验证结果：

```text
64 env / 30 iter, run_name=a3_scan101_noise015_probe

对比 scan101 noise=0.20:
  reward_contact mean:          0.03989 -> 0.02501
  reward_contact last10:        0.04433 -> 0.02330
  reward_future_landing mean:   0.01778 -> 0.01396
  reward_future_touch_point:    0.39215 -> 0.36600
  undesired_contacts mean:    -45.96783 -> -41.68706
  undesired_contacts last10:  -57.90598 -> -48.25888
  termination_penalty mean:    -0.25833 -> -0.38796
  reward_table_success:         0       -> 0
```

阶段判断：

- `A3_INITIAL_POLICY_NOISE_STD=0.15` 可以略微降低身体/手臂碰撞，但会牺牲一部分触球和落点信号；
- 它不是根因修复，只能作为早期稳定性的保守改进候选；
- 后续重点继续放在 A3 右腕/拍面/右臂默认姿态，让触球后的过网高度和落到对面桌的概率提高。

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
  actual_net_cross_z mean ~= 0.786, min ~= 0.579, max ~= 0.956
```

解释：

- A3 当前默认姿态已经不是完全无法触球，但默认命中率不足；
- 已触到的球多数过网高度偏低，且没有实际触到对面桌；
- 因此下一步应继续扫描/调整右腕拍面角度和右臂默认位姿，而不是优先放宽 `table_success` 或删除稳定性惩罚。

## 2026-06-26 A3 右腕 lift1 默认姿态计划

诊断：

- scan101 在固定 easy serve 下能稳定让 `ball_vx` 变正，但随机训练球路下默认命中率只有 `15/32`；
- wrist-only 扩展扫描显示，仅继续转腕无法直接产生 `table_success`，但存在更适合早期学习的高命中候选；
- base x 前移扫描导致多个候选 reset/不稳定，不作为当前默认修改；
- `lift1 = (right_wrist_roll=0.00, right_wrist_pitch=-1.15, right_wrist_yaw=-1.40)` 在训练随机球路 hold-default 诊断中：
  - hits: `23/32`，高于 scan101 的 `15/32`；
  - resets: `0/32`；
  - hit_z mean: `~0.945`；
  - hit_vz mean: `~1.120`；
  - cross count: `23/32`；
  - `opponent_table_any` 仍为 0，说明它不是完整成功解，但能显著增加早期触球样本。

拟修改：

```text
只修改 legged_lab/assets/a3/a3.py

A3_PINGPONG_READY_JOINT_POS:
  right_wrist_roll_joint:   +0.25 -> +0.00
  right_wrist_pitch_joint:  -0.75 -> -1.15
  right_wrist_yaw_joint:    -1.40 -> -1.40
```

边界：

- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改球路；
- 不修改 `reward_contact`、`reward_future_pass_net`、`reward_table_success`；
- 不移动 base x/y；
- 不放宽碰撞白名单。

验证：

1. `py_compile`；
2. A3 hold-default 随机球路诊断：
   - 默认命中率应高于 scan101；
   - reset 不应增加；
3. 64 env / 30 iter 短训练：
   - `reward_contact` 不应回到 candidate69 的低水平；
   - `reward_future_pass_net` 若仍为 0，下一步再考虑 staged reward 或主动挥拍相关 shaping；
   - `undesired_contacts` 不应明显恶化。

验证结果：

```text
64 env / 30 iter, run_name=a3_lift1_wrist_noise015_probe

对比 scan101 + noise=0.15:
  reward_contact mean:          0.02501 -> 0.06210
  reward_contact last10:        0.02330 -> 0.06219
  reward_future_landing mean:   0.01396 -> 0.02482
  reward_future_touch_point:    0.36600 -> 0.45654
  reward_future_dis_ee:         0.27077 -> 0.31894
  undesired_contacts mean:    -41.68706 -> -44.60379
  termination_penalty mean:    -0.38796 -> -0.51713
  reward_future_pass_net:       0       -> 0
  reward_table_success:         0       -> 0
```

阶段判断：

- lift1 明显增加早期触球和落点相关信号，建议保留；
- 但它也让早期身体/手臂碰撞和 termination 略差；
- 下一步不应继续改球路或放宽成功条件，而是把 A3 初始策略噪声再缩小一档，检查是否能保住 lift1 的触球密度同时降低碰撞。

## 2026-06-26 A3 lift1 + 初始噪声 0.10 计划

拟修改：

```text
只修改 legged_lab/envs/a3_tt/a3_tt_config.py

A3_INITIAL_POLICY_NOISE_STD:
  0.15 -> 0.10
```

边界：

- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改 `A3_INITIAL_ACTION_SCALE = 0.08`；
- 不修改 lift1 右腕姿态；
- 不修改 reward 或球路；
- 不移动 base。

验证：

1. `py_compile`；
2. 64 env / 30 iter 短训：
   - `reward_contact` 应明显高于 scan101/noise=0.15；
   - `undesired_contacts` 和 `termination_penalty` 应低于 lift1/noise=0.15；
   - 若触球密度明显下降，则回退到 noise=0.15。

验证结果：

```text
64 env / 30 iter, run_name=a3_lift1_wrist_noise010_probe

对比 lift1 + noise=0.15:
  reward_contact mean:          0.06210 -> 0.07834
  reward_contact last10:        0.06219 -> 0.08150
  reward_future_landing mean:   0.02482 -> 0.02743
  reward_future_touch_point:    0.45654 -> 0.44659
  reward_future_dis_ee:         0.31894 -> 0.31004
  undesired_contacts mean:    -44.60379 -> -37.15524
  action_rate_l2 mean:         -0.01472 -> -0.00791
  termination_penalty mean:    -0.51713 -> -0.55602
  reward_future_pass_net:       0       -> 0
  reward_table_success:         0       -> 0

对比 scan101 + noise=0.15:
  reward_contact mean:          0.02501 -> 0.07834
  reward_future_landing mean:   0.01396 -> 0.02743
  undesired_contacts mean:    -41.68706 -> -37.15524
  mean_reward:               -381.91359 -> -362.24295
```

阶段结论：

- 当前建议保留 `lift1 + A3_INITIAL_POLICY_NOISE_STD=0.10`；
- 这组配置把问题从“默认姿态下触球稀疏”推进到了“触球更密，但仍不过网/不上对面桌”；
- `reward_future_pass_net` 和 `reward_table_success` 仍为 0，说明下一阶段不应继续只调 wrist，而应考虑：
  - A3 右臂主动挥拍方向/默认右臂姿态；
  - 专门针对过网高度和触球后 `ball_vx/vz` 的 staged shaping；
  - 或在不破坏 T1 的前提下，为 A3 增加早期课程，使策略先学会把球稳定打过网。

## 2026-06-26 A3 hit outcome diagnostics 计划

背景：

- 最近的 `lift1 + noise=0.10` 短训已经让 `reward_contact` 明显上升；
- 但 `reward_future_pass_net` 和 `reward_table_success` 仍为 0；
- 这表示当前主要矛盾不再只是“球拍够不到球”，而是“触球后的球没有稳定朝对面桌方向、过网高度不够或落点不对”；
- 继续直接长训会把训练时间浪费在姿态/拍面几何搜索上。

拟新增：

```text
legged_lab/scripts/a3_hit_outcome_diagnostics.py
```

用途：

- A3-only，只接受 `a3_tt` / `a3_tt_eval`；
- 默认保持 zero-action，即检测当前默认姿态和 PD 托底下的自然触球结果；
- 支持用 `--joint name=value` 临时覆盖 A3 右臂/右腕关节，用于批量扫候选姿态；
- 支持 fixed ball 或复用 A3 训练球路随机范围；
- 输出：
  - hit 数量和命中率；
  - 触球 step；
  - 触球瞬间 `ball_vx/vy/vz`；
  - 预测到网口时的 `z_at_net`；
  - 是否碰自己桌 / 对面桌；
  - reset/termination 数量。

边界：

- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改 A3 默认训练配置；
- 不修改奖励函数；
- 不修改球路；
- 不放宽 `reward_table_success` 或 `reward_future_pass_net`；
- 诊断脚本只作为下一步右臂/拍面默认姿态选择依据。

验证：

1. `py_compile` 新脚本；
2. 用当前 A3 默认姿态跑 64 env 诊断，记录基线；
3. 对右腕/右臂候选姿态做小批量扫描；
4. 只有当候选同时满足：
   - hit 率不低于当前默认；
   - `hit_vx > 0` 比例更高；
   - `z_at_net` 更接近或超过网高安全目标；
   - reset 不明显变差；
   才进入下一步默认姿态修改。

执行结果：

```text
当前默认 lift1:
  first_hit:            40/64
  hit_vx_positive:      35/64
  hit_vx mean:          +0.560
  hit_vz mean:          +0.915
  actual_crossed_net:   40/64
  actual_z_at_net mean: 0.715
  actual_z_at_net max:  0.997
  opponent_table:       0/64
  reset_seen:           0/64

scan101 wrist (roll=+0.25, pitch=-0.75, yaw=-1.40):
  first_hit:            31/64
  hit_vx_positive:      31/64
  hit_vx mean:          +1.325
  hit_vz mean:          +0.424
  actual_z_at_net mean: 0.835
  opponent_table:       0/64
  reset_seen:           0/64

candidate roll=-0.25, pitch=-1.25, yaw=-1.40:
  first_hit:            47/64
  hit_vx_positive:      46/64
  hit_vx mean:          +0.779
  hit_vz mean:          +0.865
  actual_crossed_net:   41/64
  actual_z_at_net mean: 0.814
  actual_z_at_net max:  1.004
  opponent_table:       0/64
  reset_seen:           0/64
```

判断：

- `scan101` 的正向速度最好，但触球率低、上抛不足；
- 当前 `lift1` 的触球率可以，但 `vx` 不够稳定；
- `roll=-0.25, pitch=-1.25, yaw=-1.40` 是当前 wrist-only 扫描下更平衡的默认姿态：
  - 触球率高于当前默认；
  - `hit_vx_positive` 明显高于当前默认；
  - `actual_z_at_net` 均值高于当前默认；
  - reset 没有恶化。

拟修改：

```text
legged_lab/assets/a3/a3.py

A3_PINGPONG_READY_JOINT_POS:
  right_wrist_roll_joint:   +0.00 -> -0.25
  right_wrist_pitch_joint:  -1.15 -> -1.25
  right_wrist_yaw_joint:    -1.40 -> -1.40
```

阶段边界：

- 这一步只改 A3 默认右腕；
- 不改 T1；
- 不改公共 `TTEnv`；
- 不改 reward；
- 不改球路；
- 不放宽成功条件。

阶段结论：

- 右腕默认姿态可以改善 A3 早期样本质量，但仍无法仅靠静态默认姿态获得 `reward_future_pass_net/table_success`；
- 原因是静态拍面能让球反弹，但回球速度仍远低于真正过网落对面的需要；
- 下一阶段应在保持 T1 不变的前提下，增加 A3-only staged shaping，让策略学习主动挥拍，而不是继续只依赖默认姿态。

修改后验证：

```text
py_compile:
  legged_lab/assets/a3/a3.py
  legged_lab/envs/a3_tt/a3_tt_config.py
  legged_lab/mdp/rewards.py
  legged_lab/scripts/a3_hit_outcome_diagnostics.py
  result: pass

default-after-change hit diagnostics:
  first_hit:            47/64
  hit_vx_positive:      46/64
  hit_vz_positive:      41/64
  actual_crossed_net:   41/64
  actual_z_at_net mean: 0.814
  actual_z_at_net max:  1.004
  opponent_table:       0/64
  reset_seen:           0/64

64 env / 30 iter short train:
  run: logs/a3_table_tennis/2026-06-26_21-08-40_a3_rollneg_pitch125_probe
  reward_contact mean:             0.09754
  reward_future_touch_point mean:  0.58099
  reward_future_landing_dis mean:  0.03348
  reward_future_pass_net mean:     0.00000
  reward_table_success mean:       0.00000
  undesired_contacts mean:       -45.38344
  termination_penalty mean:       -0.18657
```

与上一组 `lift1 + noise=0.10` 短训对比：

```text
reward_contact mean:             0.07834 -> 0.09754
reward_future_touch_point mean:  0.44659 -> 0.58099
reward_future_landing_dis mean:  0.02743 -> 0.03348
reward_future_pass_net mean:     0       -> 0
reward_table_success mean:       0       -> 0
undesired_contacts mean:       -37.15524 -> -45.38344
```

判断：

- 默认姿态确实提升触球和击球中心对齐相关信号；
- 但它不会自动产生过网/落对面桌；
- undesired contact 变差，说明后续 staged shaping 需要和稳定性约束一起调，不应只大幅提高击球奖励。

## 2026-06-26 A3 staged hit velocity/net reward 计划

问题：

- A3 当前已经能触拍；
- 但触拍后 `hit_vx` 均值只有约 `0.78 m/s`，最高约 `1.54 m/s`；
- 以 A3 当前击球位置到网口的距离看，这个速度不足以在下落前过网；
- `reward_future_pass_net` 和 `reward_table_success` 都是结果型稀疏信号，早期几乎拿不到；
- 如果只继续加大随机探索，容易增加身体碰撞和摔倒，不一定学到“主动挥拍”。

拟修改：

```text
legged_lab/mdp/rewards.py
  新增 reward_hit_ball_velocity_net_target(...)

legged_lab/envs/a3_tt/a3_tt_config.py
  A3 reward 中新增 reward_hit_ball_velocity_net
```

设计：

- 只在 `env.ball_landing_dis_rew` 为 True 的触拍事件上给奖励；
- 要求 `ball_vx > min_vx`；
- 用当前触拍瞬间的 `ball_pos/ball_linvel` 估算球到 x=0 网口平面的高度；
- 奖励由两部分组成：
  - `vx_score`：鼓励触拍后 `vx` 接近 `vx_target=3.0 m/s`；
  - `net_score`：鼓励估算网口高度接近 `z_target=1.05 m`；
- 使用连续指数型 shaping，不放宽 `reward_future_pass_net` 和 `reward_table_success`；
- 初始权重保守设为 `20.0`，避免压过稳定性惩罚。

边界：

- T1 不使用该 reward term；
- 不修改公共 `TTEnv`；
- 不修改球路；
- 不修改 table success / pass net 的真实判定；
- 不删除或降低稳定性惩罚；
- 若短训中 undesired contacts 明显恶化，则降低权重或增加 staged curriculum，而不是继续加奖励。

验证：

1. `py_compile`；
2. 64 env / 30 iter 短训：
   - `reward_hit_ball_velocity_net` 应有非零信号；
   - `reward_contact` 不应消失；
   - `reward_future_pass_net/table_success` 仍可为 0，但应观察是否开始出现上升趋势；
   - `undesired_contacts` 和 `termination_penalty` 不应显著恶化。

第一次短训观察：

```text
run: logs/a3_table_tennis/2026-06-26_21-13-17_a3_staged_hit_velocity_probe
stopped at iter ~= 13 for reward design correction
reward_hit_ball_velocity_net: mostly 0, peak only about 0.0008
reason: max_t_net was used as a hard mask; current A3 hit_vx is too slow, so most hits exceed max_t_net and receive zero staged reward.
```

修正：

- 保留 `max_t_net=1.4` 作为软时间惩罚参考；
- 移除 `t_net <= max_t_net` 硬 mask；
- 新增 `t_std=0.7`，用 `exp(-(t_net - max_t_net)+ / t_std)` 给慢球连续降权；
- 仍然要求 `ball_vx > min_vx` 和触拍事件 `env.ball_landing_dis_rew`。

第二次短训观察：

```text
run: logs/a3_table_tennis/2026-06-26_21-17-04_a3_staged_hit_velocity_component_probe
max_iterations: 10

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

- component-style staged reward 已经有稳定非零信号；
- 它的量级小于 `reward_future_landing_dis`，目前不会压过原有任务/稳定性项；
- 10 iter 内仍没有 `pass_net/table_success`，这符合预期；
- 后续建议用 250-500 iter 观察该信号是否推动 `hit_vx` 和 `actual_z_at_net` 继续上升，再决定是否调整权重。

## 2026-06-26 A3 staged hit velocity 500-iter run 监控结论

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

监控决策：

- 训练在 iter `187/500` 手动停止；
- 原因不是仿真崩溃，而是任务指标在 180 iter 后已经平台化，继续跑满 500 的收益很低。

最终统计：

```text
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

判断：

- 策略确实学会了压低 `undesired_contacts`，身体碰撞从早期大负值接近 0；
- 但它没有学到有效乒乓球回球：
  - `reward_contact` 没有上升；
  - staged hit velocity reward 没有上升；
  - `reward_future_pass_net` 只有 `1e-6` 量级；
  - `reward_table_success` 始终为 0；
- `termination_penalty` 接近 `-2.0` 且 episode length 降到约 `105`，说明策略可能偏向更短 episode/保守规避，而不是主动击球。

下一步建议：

- 不继续用当前 reward 权重跑长训；
- staged reward 需要更强地绑定“触球后球速/过网潜力”，或者把触球后诊断量直接记录进训练指标；
- 同时需要避免策略通过缩短 episode 或规避动作得到更高总回报。

## 2026-06-26 A3 reward rebalance 计划

背景：

- `a3_staged_hit_velocity_500` 在 iter 187 手动停止；
- 该 run 显示策略主要学会减少 `undesired_contacts`，但任务指标平台化：
  - `reward_contact last50 ~= 0.0399`；
  - `reward_hit_ball_velocity_net last50 ~= 0.0040`；
  - `reward_future_pass_net last50 ~= 0.000002`；
  - `reward_table_success = 0`；
- 方向性 staged reward 比接触奖励小约 10 倍，策略仍可能优先学“碰到/规避”，而不是学“有效回球”。

拟修改：

```text
legged_lab/envs/a3_tt/a3_tt_config.py

reward_contact:
  weight 180.0 -> 120.0

reward_hit_ball_velocity_net:
  weight 20.0 -> 120.0
```

设计意图：

- 降低“只要触球就好”的奖励压力；
- 提高“触球后 vx/vz/net-potential 好”的阶段性奖励；
- 让策略更难通过低风险接触和短 episode 获得相对更好的回报；
- 仍保留真实 `reward_future_pass_net` / `reward_table_success`，不放宽成功标准。

边界：

- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改球路；
- 不修改默认姿态；
- 不降低稳定性惩罚；
- 不修改 `reward_future_pass_net` 和 `reward_table_success` 的判定逻辑。

验证：

1. `py_compile`；
2. 先跑 4096 env / 120 iter probe；
3. 观察最近 50 iter：
   - `reward_hit_ball_velocity_net` 是否高于上一 run 的 `~0.004`；
   - `reward_contact` 不应完全消失；
   - `reward_future_pass_net` 是否从 `1e-6` 量级变大；
   - `termination_penalty` 不应比 `-2.0` 更差；
   - 若 `undesired_contacts` 再次大幅恶化且任务信号不涨，则停止并回退/调低权重。

结果：

```text
run: logs/a3_table_tennis/2026-06-26_21-41-47_a3_reward_rebalance_120
checkpoint: model_119.pt

reward_contact last20:                0.02883222
reward_hit_ball_velocity_net last20:  0.02663544
reward_future_landing_dis last20:     0.02157779
reward_future_pass_net last20:        0.00000144
reward_table_success last20:          0.00000000
undesired_contacts last20:           -2.19831181
termination_penalty last20:          -1.93190809
mean_reward last20:                 -38.05611534
mean_episode_length last20:         109.78849983
```

判断：

- 重平衡后，`reward_hit_ball_velocity_net` 已经从上一 run 的 `~0.004` 提高到 `~0.0266`；
- 但 `reward_future_pass_net` 仍是 `1e-6` 量级，`reward_table_success` 仍为 0；
- `termination_penalty` 仍接近 `-2.0`，episode 仍大量提前结束；
- 说明问题已经不只是“没有 staged reward 信号”，而是策略触球后没有形成稳定过网/落台轨迹，且 episode reset 原因需要进一步定位。

## 2026-06-26 A3 policy outcome diagnostics 计划

背景：

- 当前 `a3_hit_outcome_diagnostics.py` 只支持零动作 rollout，能验证默认姿态几何是否能触球；
- 训练失败时，还需要知道已训练策略触球后的真实球速、过网高度、落点事件，以及 episode 是因为高度、前后位置、左右位置还是 timeout 被 reset；
- 直接改公共 `TTEnv.check_reset()` 或全局日志会影响 T1 边界，不适合作为第一步。

拟修改：

```text
legged_lab/scripts/a3_hit_outcome_diagnostics.py

新增可选参数：
  --load_run
  --checkpoint
  --predictor

新增行为：
  - 默认仍保持零动作诊断；
  - 当指定 checkpoint 时，加载 A3 policy 并使用 policy action rollout；
  - 统计 reset 原因：
      root_z < 0.50
      robot_x < -3.6
      robot_x > -1.35
      robot_y < -1.1
      robot_y > 1.1
      time_out_buf
  - 输出 root pose / reset step / hit outcome 汇总；
```

边界：

- 只修改 A3 专用诊断脚本；
- 不修改 T1；
- 不修改公共 `TTEnv`；
- 不修改 A3 训练配置；
- 不改变任何训练/eval 命令的默认行为。

验证：

1. `py_compile`；
2. 零动作诊断仍可运行；
3. 使用 `a3_reward_rebalance_120/model_119.pt` 跑 policy 诊断；
4. 根据 policy 诊断结果再决定是否继续调拍面/右腕、reset curriculum 或 reward/curriculum。

结果：

```text
run: logs/a3_table_tennis/2026-06-26_21-41-47_a3_reward_rebalance_120/model_119.pt
command mode: policy
num_envs: 128
random_ball: true

first_hit:                 76/128 (0.594)
hit_vx_positive:           63/128 (0.492)
hit_vz_positive:           70/128 (0.547)
actual_crossed_net:        70/128 (0.547)
actual_net_clear:           0/128 (0.000)
opponent_table_after_hit:   0/128 (0.000)
own_table_after_hit:       76/128 (0.594)
reset_seen:                82/128 (0.641)
reset_low_z:                8/128 (0.062)
reset_y_high:               4/128 (0.031)
reset_timeout:             74/128 (0.578)

hit_x mean:                -1.5547
hit_z mean:                 0.9776
hit_vx mean:                0.5620
hit_vz mean:                1.0069
actual_z_at_net mean:       0.8119
```

判断：

- A3 已经不是“完全碰不到球”，而是“能碰球但触球质量错误”；
- 触球后球多数落在己方桌面，实际到网平面时高度约 `0.81 m`，低于 `1.11 m`；
- reset 主要来自 serve/episode timeout，少量来自低高度或 y 方向越界，不是大面积立刻摔倒；
- 单纯继续长训当前 reward，策略会停在低质量触球局部最优。

## 2026-06-26 A3 landing reward compatibility 计划

背景：

- 右腕/右肘/底座小范围扫描没有找到能让零动作或早期策略直接过网的姿态；
- 当前策略诊断显示：
  - `own_table_after_hit` 与 `first_hit` 基本一致；
  - `opponent_table_after_hit` 为 0；
  - 说明策略主要学到了把球挡/打回己方桌面；
- 旧的 `reward_future_landing_dis` 形式是：

```text
reward = threshold - distance(predicted_landing_xy, opponent_target_xy)
```

当 `threshold=3.0` 时，即使预测落点在己方桌面，只要距离对面目标不超过 3m，也可能仍得到正奖励。T1 原始几何下这不是主要问题，因为 T1 初始击球质量更容易越过网；迁移到 A3 后，这个宽松项会奖励“己方落台的低质量触球”，加重局部最优。

拟修改：

```text
legged_lab/mdp/rewards.py

新增：
  reward_future_opponent_landing_target()
    - 仅当 first post-hit 预测落点 x > 0 才给正奖励；
    - 用 exp(-distance / std^2) 奖励靠近对面目标；

  penalty_future_own_landing_after_hit()
    - 当 first post-hit 预测落点 x <= 0 时给 1；
    - 在 A3 配置中用小负权重，防止策略满足于己方落台。

legged_lab/envs/a3_tt/a3_tt_config.py

调整：
  reward_future_landing_dis weight 60.0 -> 0.0
  新增 reward_future_opponent_landing
  新增 penalty_future_own_landing
```

边界：

- 新 reward 函数放在公共 `mdp/rewards.py`，但只在 A3 配置引用；
- 不修改 T1 配置；
- 不修改公共 `TTEnv`；
- 不修改球路；
- 不修改默认姿态；
- 不放宽 `reward_future_pass_net` / `reward_table_success` 成功标准。

验证：

1. `py_compile`；
2. 重新跑 A3 hit outcome diagnostics，确认诊断脚本仍可用；
3. 跑 64 env / 30 iter probe，观察：
   - 新的 opponent landing reward 是否非零；
   - own landing penalty 是否能暴露当前失败模式；
   - `reward_hit_ball_velocity_net` 不应消失；
   - `reward_contact` 不应完全消失；
4. 若短探针没有数值异常，再建议进入 4096 env / 250-500 iter 观察。

结果：

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
mean_episode_length last10:          207.37399902
```

判断：

- 新 reward 没有数值崩溃；
- `penalty_future_own_landing` 成功暴露“预测己方落点”失败模式；
- 但 `reward_future_opponent_landing` 全程为 0，说明“预测落点必须在对面半桌”对早期 A3 仍然过于稀疏；
- 需要在完全成功前增加连续进展项，让预测落点从己方逐步向对面推进，而不是只有跨过 `x=0` 后才给奖励。

## 2026-06-26 A3 landing x-progress curriculum 计划

背景：

- 禁用旧的宽松 `reward_future_landing_dis` 是必要的，因为它会奖励己方落台；
- 但只保留 opponent-side landing reward 又太稀疏；
- A3 当前需要一个中间梯度：只要 post-hit 预测落点 x 方向更靠近对面，就给连续奖励。

拟修改：

```text
legged_lab/mdp/rewards.py

新增：
  reward_future_landing_x_progress()
    - first post-hit 触发；
    - 对 predict_x_land 做 clamp 归一化：
        min_x -> 0
        target_x -> 1
    - 可选叠加 y 方向居中项；
    - 即使还在己方半桌，也只奖励“更向前”，不再按 3m 距离宽松奖励。

legged_lab/envs/a3_tt/a3_tt_config.py

新增：
  reward_future_landing_x_progress weight 80.0
```

边界：

- 仍然只在 A3 配置启用；
- 不恢复旧的 `reward_future_landing_dis`；
- 不降低真实成功项；
- 不修改 T1 和 `TTEnv`。

验证：

1. `py_compile`；
2. 64 env / 30 iter probe；
3. 若 `reward_future_landing_x_progress` 有非零信号且 `reward_hit_ball_velocity_net` 不消失，再进入 4096 env / 250-500 iter。

验证结果 1：

```text
run: logs/a3_table_tennis/2026-06-26_22-11-38_a3_landing_x_progress_probe
num_envs: 64
max_iterations: 30

reward_contact last10:                    0.07295367
reward_future_opponent_landing last10:    0.00000000
reward_future_landing_x_progress last10:  0.00555782
penalty_future_own_landing last10:       -0.06866667
reward_hit_ball_velocity_net last10:      0.04492416
reward_future_pass_net last10:            0.00000004
reward_table_success last10:              0.00000000
undesired_contacts last10:              -50.66830635
termination_penalty last10:              -0.68055558
```

判断：

- x-progress 项有非零信号，但量级偏小；
- `reward_hit_ball_velocity_net` 没有消失，说明连续落点推进项没有立刻压掉触球速度项；
- `reward_future_opponent_landing` 和 `reward_table_success` 仍为 0，说明还处在早期塑形阶段。

追加调参：

```text
reward_future_landing_x_progress:
  min_x: -1.5 -> -3.0
  weight: 80.0 -> 120.0
```

理由：

- 当前 A3 触球后预测落点经常仍在己方，`min_x=-1.5` 会让大部分早期样本 reward 过低；
- 放宽到 `min_x=-3.0` 不是放宽成功标准，只是让“比当前更向前”更早可见；
- 对面落点奖励、过网奖励、桌面成功奖励仍保持原标准。

验证结果 2：

```text
run: logs/a3_table_tennis/2026-06-26_22-14-44_a3_landing_x_progress_tuned_smoke
num_envs: 64
max_iterations: 10

reward_contact mean/last:                    0.06262150 / 0.09672494
reward_future_opponent_landing mean/last:    0.00000000 / 0.00000000
reward_future_landing_x_progress mean/last:  0.05438230 / 0.09806083
penalty_future_own_landing mean/last:       -0.04977778 / -0.08500000
reward_hit_ball_velocity_net mean/last:      0.03170094 / 0.05187657
reward_future_pass_net mean/last:            0.00000004 / 0.00000000
reward_table_success mean/last:              0.00000000 / 0.00000000
undesired_contacts mean/last:              -24.09599653 / -57.74331284
termination_penalty mean/last:              -0.10416667 / -0.50000000
```

判断：

- tuned x-progress 已经成为可见的早期学习信号；
- own-side landing penalty 同时可见，可以继续压制“己方落台”的局部最优；
- 仍未看到真实过网/对面落台成功，下一步需要 4096 env / 250-500 iter 观察是否产生趋势；
- 如果 100-150 iter 后 `reward_future_landing_x_progress` 不升、`penalty_future_own_landing` 不降，或 `undesired_contacts` 长期接近 `-50`，应停止训练并继续调 A3-only reward/默认姿态。

下一步命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_landing_xprogress_500 \
  --headless \
  --predictor
```

重点监控：

- `Episode_Reward/reward_future_landing_x_progress` 应上升；
- `Episode_Reward/penalty_future_own_landing` 应向 0 收敛；
- `Episode_Reward/reward_hit_ball_velocity_net` 不应塌到 0；
- `Episode_Reward/reward_future_pass_net` 应从 `1e-6` 量级往上走；
- `Episode_Reward/reward_table_success` 可以晚一些出现，但不能长期完全没有趋势；
- `Episode_Reward/undesired_contacts` 若长期过大，说明动作/姿态仍破坏站立或球拍接触边界。

## 2026-06-26 A3 x-progress 500 探针中止结论

命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=a3_landing_xprogress_500_tmux1 \
  --headless \
  --predictor
```

运行方式：

- 使用 `tmux` 托管训练，避免交互句柄断开导致训练被杀；
- TensorBoard 保持在 `http://127.0.0.1:16006`。

中止点：

```text
stopped around iter: 113/500
```

监控结果：

```text
reward_contact tail10:                    0.02449041
reward_future_landing_x_progress tail10:  0.03244517
penalty_future_own_landing tail10:       -0.02927472
reward_hit_ball_velocity_net tail10:      0.02287906
reward_future_pass_net tail10:            0.00000040
reward_table_success tail10:              0.00000000
undesired_contacts tail10:               -1.05193623
termination_penalty tail10:              -1.97212617
mean_episode_length tail10:             109.31700134
```

判断：

- A3 已经能把 `undesired_contacts` 压低，说明站立/身体碰撞问题相对改善；
- 但 `reward_contact`、`reward_future_landing_x_progress`、`reward_hit_ball_velocity_net` 都在下降；
- `termination_penalty` 接近 `-2`，`mean_episode_length` 下降，说明策略更像是在走保守/短 episode 路线；
- `reward_future_pass_net` 仍只有 `1e-7` 到 `1e-6` 量级，`reward_table_success` 没有持续出现；
- 不适合直接开 10000 iter 长训，否则大概率会把保守局部最优训练得更稳定。

## 2026-06-26 A3 Phase 2 主动过网塑形计划

目标：

- 保留 T1 原始 workflow；
- 不修改公共 `TTEnv` reset/termination；
- A3 已经能站稳后，将早期学习重心从“少碰撞”转向“主动触球并提高网口高度/前向速度”；
- 不放宽真实 `reward_future_pass_net` 和 `reward_table_success` 成功标准。

拟修改：

```text
legged_lab/mdp/rewards.py

新增：
  reward_hit_net_clearance_progress()
    - 仍然只在 first post-hit 事件上计算；
    - 使用当前球位置/速度估计到 x=0 网口平面的高度；
    - 对 z_at_net 从 min_z 到 target_z 做连续 progress；
    - 同时乘上 forward velocity/time score，避免只向上挑但不过网；
    - 作用是给 A3 早期“打高到能过网”一个更直接的梯度。

legged_lab/envs/a3_tt/a3_tt_config.py

调整：
  A3_INITIAL_ACTION_SCALE: 0.08 -> 0.10
    - 当前稳定性已经改善，需要给右臂/球拍更大动作幅度；
    - 仍低于 T1 默认 action_scale=0.25，保持保守。

  A3_TRAIN_MAX_SERVE_PER_EPISODE: 3 -> 5
    - 与 T1/base 默认一致；
    - 让策略在一次 episode 内看到更多发球，减少过早 episode 切断。

  reward_contact: 120 -> 150
    - 与 T1 接触奖励量级一致；
    - 避免策略通过少触球降低风险。

  reward_hit_ball_velocity_net: 120 -> 180
    - 加强触球后前向速度/网口潜力。

  reward_hit_net_clearance_progress: weight 160
    - 新增 A3 早期过网高度塑形。

  penalty_future_own_landing: -40 -> -25
    - 仍惩罚己方落台；
    - 但降低早期“多数触球都不完美”时的接触回避倾向。

  reward_future_opponent_landing: 120 -> 160
  reward_future_pass_net: 100 -> 150
  reward_table_success: 100 -> 150
    - 稀疏成功一旦出现，需要比保守稳定项更有吸引力。
```

边界：

- 不修改 `legged_lab/envs/t1_tt/t1_tt_config.py`；
- 不修改公共 `TTEnv`；
- 不修改球路；
- 不修改 A3 URDF/USD 资产；
- 不修改真实成功判断，只增加 A3 早期塑形。

验证计划：

1. `py_compile`；
2. 64 env / 20-30 iter smoke，确认 reward 不 NaN、训练能启动；
3. 4096 env / 150-250 iter probe；
4. 若 `reward_hit_net_clearance_progress` 和 `reward_future_pass_net` 出现上升趋势，同时 `undesired_contacts` 不重新爆炸，再考虑 2000-10000 iter 长训。

Phase 2 smoke 结果：

```text
run: logs/a3_table_tennis/2026-06-26_22-51-23_a3_phase2_netclear_smoke
run: logs/a3_table_tennis/2026-06-26_22-53-54_a3_phase2_netclear_soft_smoke
```

结论：

- `A3_INITIAL_ACTION_SCALE=0.10` 对 A3 全身太激进，短训中 `undesired_contacts` 可到 `-100` 以上；
- 将 net-clearance 改成 soft height score 后，日志中仍基本为 0，说明该 reward 形式没有提供有效早期梯度；
- 已撤回激进权重：
  - `A3_INITIAL_ACTION_SCALE` 回到 `0.08`；
  - `A3_TRAIN_MAX_SERVE_PER_EPISODE` 回到 `3`；
  - `reward_contact/reward_hit_ball_velocity_net/reward_future_pass_net/reward_table_success` 回到上一版稳定权重；
  - `reward_hit_net_clearance_progress` 保留为可选函数，但在 A3 配置中权重为 `0.0`，不影响训练。

## 2026-06-26 A3 右臂/拍面批量几何搜索计划

背景：

- 奖励堆权重会诱导 A3 全身动作变大，身体接触惩罚爆炸；
- 更合理的下一步是先判断当前右臂/拍面默认几何是否已经接近最优。

已新增诊断工具：

```text
legged_lab/scripts/a3_right_arm_pose_grid_search.py
```

功能：

- 同一个 Isaac 进程内，每个 env 分配一个候选右臂/手腕默认姿态；
- 零动作保持默认姿态；
- 统一发同一条球；
- 统计 first-hit 后的 `hit_vx/hit_vz/actual_z_at_net/own_table/opponent_table`；
- 输出 top candidates 和 CSV。

粗网格结果：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python \
  -m legged_lab.scripts.a3_right_arm_pose_grid_search \
  --task=a3_tt \
  --max_steps=180 \
  --top_k=15 \
  --csv logs/a3_table_tennis/a3_right_arm_grid_20260626_2300.csv \
  --headless
```

统计：

```text
candidates: 108
hit: 78
crossed x=0: 77
actual_z_at_net > 0.9125: 36
actual_z_at_net > 1.0: 2
actual_z_at_net > 1.11: 0
opponent_table_after_hit: 0
own_table_after_hit: 78
```

Top candidate：

```text
id=052
re=0.410
rwr=-0.250
rwp=-1.250
rwy=-1.400
hit_vx=1.012
hit_vz=1.327
actual_z_at_net=0.984
actual_crossed_net=True
opponent_table_after_hit=False
own_table_after_hit=True
```

判断：

- Top candidate 基本就是当前 A3 默认右臂/手腕姿态；
- 当前默认姿态不是随便猜的，它在粗网格里已经是比较好的被动击球几何；
- 但所有被动候选都先落己方桌，说明仅靠静态姿态无法完成任务；
- 下一步应该让策略学右臂主动挥拍，同时继续限制腿/腰/左臂动作，防止全身不稳定。

## 2026-06-26 A3 per-joint action scale 计划

目标：

- T1 原始 workflow 不变；
- A3 初期训练中让右臂/手腕有更大动作空间；
- 腿、脚、腰、左臂保持小动作范围，避免为了击球破坏站立；
- 比全局 `action_scale=0.10` 更细、更安全。

拟修改：

```text
legged_lab/envs/base/tt_env.py

兼容扩展：
  - 如果 cfg.robot.action_scale 是 float/int：
      保持旧行为；
  - 如果 cfg.robot.action_scale 是 list/tuple：
      转成 shape=(1, num_actions) 的 tensor；
      与 clipped_actions 广播相乘；
      长度不等于 num_actions 时直接报错。

legged_lab/envs/a3_tt/a3_tt_config.py

新增 A3_ACTION_SCALE_BY_JOINT，按 A3_T2D5_JOINT_NAMES 顺序：
  - waist: 0.04
  - head: 0.00
  - left arm: 0.02
  - right shoulder: 0.12
  - right elbow: 0.14
  - right wrist: 0.16
  - hips/knees: 0.035
  - ankles: 0.03
```

边界：

- 不修改 T1 配置；
- 不改变 observation/action 维度；
- 不改变 actor 网络结构；
- 不改变 reset/termination；
- 只改变 A3 的 action scale 数值。

验证：

1. `py_compile`；
2. 64 env / 15-20 iter smoke；
3. 若 `undesired_contacts` 不爆炸，再跑 4096 env / 150-250 iter；
4. 如果任务信号上升且稳定性可控，再进入长训。

per-joint 250 probe 结果：

```text
run: logs/a3_table_tennis/2026-06-26_23-05-54_a3_perjoint_scale_250
checkpoint: model_249.pt
```

TensorBoard：

```text
reward_contact tail10:                    0.04043716
reward_future_landing_x_progress tail10:  0.04737265
penalty_future_own_landing tail10:       -0.04160717
reward_hit_ball_velocity_net tail10:      0.03555303
reward_future_pass_net tail10:            0.00000056
reward_table_success tail10:              0.00000000
undesired_contacts tail10:               -0.05651013
termination_penalty tail10:              -1.99896545
mean_episode_length tail10:             100.37199936
```

policy outcome diagnostics：

```text
first_hit:                 93/128 (0.727)
hit_vx_positive:           83/128 (0.648)
hit_vz_positive:           86/128 (0.672)
actual_crossed_net:        88/128 (0.688)
opponent_table_after_hit:   0/128 (0.000)
own_table_after_hit:       93/128 (0.727)
reset_timeout:             79/128 (0.617)
hit_vx mean:                0.6792
hit_vz mean:                0.8804
actual_z_at_net mean:       0.8351
```

判断：

- per-joint action scale 有正向作用：触球率、过 x=0 比上一版更高；
- 但仍然全部先落己方桌；
- 当前 `penalty_future_own_landing` 只在 first-hit 预测落点上惩罚，实际 own-table 事件没有直接作为 A3 reward；
- 下一步应加入实际 own-table 失败事件的 A3-only 惩罚，避免策略满足于“打到己方桌后再过 x=0”。

## 2026-06-26 A3 actual own-table penalty 计划

拟修改：

```text
legged_lab/envs/base/tt_env.py

兼容扩展：
  - 在 compute_intermediate_values 中保存 self.has_touch_own_table_just_now；
  - reset 时清零；
  - T1 不引用该字段，原始 reward/workflow 不变。

legged_lab/mdp/rewards.py

新增：
  penalty_own_table_after_paddle_hit()
    - 当 has_touch_paddle 且 has_touch_own_table_just_now 时返回 1；
    - 若已经碰到对面桌，不再惩罚；
    - 这是实际事件惩罚，不依赖 predicted landing。

legged_lab/envs/a3_tt/a3_tt_config.py

新增 A3 reward term：
  penalty_actual_own_table_after_hit weight -80.0
```

边界：

- 不改 T1 配置；
- 不改变 reset；
- 不改变球路；
- 不改变真实成功标准；
- 只让 A3 明确知道“己方落台”是失败。

验证：

1. `py_compile`；
2. 64 env smoke；
3. 4096 env / 150-250 iter probe；
4. 如果 own-table 比例下降、`reward_future_pass_net` 上升，再考虑长训。

实际验证结果：

```text
run: logs/a3_table_tennis/2026-06-26_23-34-03_a3_actual_own_penalty_250
checkpoint: model_249.pt
```

TensorBoard tail10：

```text
reward_contact:                    0.036505
reward_future_landing_x_progress:  0.043151
penalty_future_own_landing:       -0.038092
penalty_actual_own_table_after_hit:-0.072452
reward_hit_ball_velocity_net:      0.032193
reward_future_pass_net:            0.000001
reward_table_success:              0.000000
undesired_contacts:               -0.057960
termination_penalty:              -1.999027
```

policy outcome diagnostics：

```text
first_hit:                 92/128 (0.719)
hit_vx_positive:           80/128 (0.625)
hit_vz_positive:           86/128 (0.672)
actual_crossed_net:        82/128 (0.641)
actual_net_clear:           0/128 (0.000)
opponent_table_after_hit:   0/128 (0.000)
own_table_after_hit:       92/128 (0.719)
hit_vx mean:                0.6379
hit_vz mean:                0.9581
actual_z_at_net mean:       0.8456
actual_z_at_net max:        1.0200
```

与 `a3_perjoint_scale_250` 对比：

- 稳定性基本保持，`undesired_contacts` 尾部仍能回到稳定区；
- actual own-table 惩罚使曲线有轻微改善，但物理诊断中 `own_table_after_hit` 仍几乎等于触拍数；
- 主要问题不是“完全不碰球”，而是触拍后 `vx` 不够大、网处高度不够，球仍先落己方桌或低于网高；
- 现阶段不建议直接 10000 长训，否则大概率强化“稳定触球但不过网”的局部最优。

## 2026-06-26 A3 stage-2 active swing 计划

目标：

- 不改 T1；
- 不改球路；
- 不改成功标准；
- 只在 A3 上提高右臂/右腕主动挥拍能力，并把触拍后奖励从“能碰到球”进一步推向“更快、更向上、更接近直接过网”。

拟修改：

```text
legged_lab/envs/a3_tt/a3_tt_config.py

A3_ACTION_SCALE_BY_JOINT：
  - 保持 torso/head/left arm/legs/ankles 不变；
  - right shoulder: 0.12 -> 0.16
  - right elbow:    0.14 -> 0.20
  - right wrist:    0.16 -> 0.22

A3 reward：
  - reward_contact: 120 -> 100，降低“只碰到球”局部最优；
  - reward_hit_ball_velocity_net: 120 -> 180；
  - reward_hit_ball_velocity_net 参数更偏 vx/vz：
      vx_target 3.0
      vz_target 2.0
      z_target 1.10
      max_t_net 1.2
      vx_weight 0.55
      vz_weight 0.30
      z_weight 0.10
      t_weight 0.05
  - reward_hit_net_clearance_progress: 0 -> 60，作为过网高度辅助项；
  - 保留 actual own-table 惩罚，继续防止己方落台。
```

边界：

- 如果 smoke 中 `undesired_contacts` 明显比 per-joint baseline 更差，立即回退右臂 action scale 或降低 stage-2 权重；
- 如果触拍率明显下降，恢复 `reward_contact=120` 或减小 own-table 惩罚；
- 如果触拍率保持、`hit_vx` / `actual_z_at_net` 上升，再跑 4096 env / 250 probe；
- 只有当 `actual_net_clear` 或 `opponent_table_after_hit` 出现非零苗头，才进入 1000+ 或 10000 长训。

stage-2 aggressive smoke 结果：

```text
run: logs/a3_table_tennis/2026-06-26_23-58-44_a3_stage2_active_swing_smoke
num_envs: 64
max_iterations: 20

reward_future_landing_x_progress tail10:  0.086986
penalty_actual_own_table_after_hit tail10:-0.276444
reward_hit_ball_velocity_net tail10:      0.080088
reward_hit_net_clearance_progress tail10: 0.000000
reward_future_pass_net tail10:            0.000000
reward_table_success tail10:              0.000000
undesired_contacts tail10:              -80.502790
```

判断：

- `x-progress` 和 `hit_velocity` 增强，说明右臂主动挥拍方向有效；
- `undesired_contacts` 比 baseline smoke 的约 `-62` 更差，触发边界条件；
- 先降到中间档，而不是直接跑 250。

stage-2 moderated 修改：

```text
right shoulder: 0.16 -> 0.14
right elbow:    0.20 -> 0.18
right wrist:    0.22 -> 0.20
reward_contact: 100 -> 110
reward_hit_ball_velocity_net: 180 -> 160
reward_hit_net_clearance_progress: 60 -> 30
```

stage-2 moderated smoke 结果：

```text
run: logs/a3_table_tennis/2026-06-27_00-01-38_a3_stage2_moderated_smoke
num_envs: 64
max_iterations: 20

reward_contact tail10:                    0.047673
reward_future_landing_x_progress tail10:  0.077569
penalty_future_own_landing tail10:       -0.068000
penalty_actual_own_table_after_hit tail10:-0.277667
reward_hit_ball_velocity_net tail10:      0.061982
reward_hit_net_clearance_progress tail10: 0.000000
reward_future_pass_net tail10:            0.000000
reward_table_success tail10:              0.000000
undesired_contacts tail10:              -71.865002
termination_penalty tail10:              -0.079167
```

判断：

- 比 aggressive 稳定，但仍比 baseline smoke 的 `undesired_contacts ~= -62` 更激进；
- `x-progress` 和 `hit_velocity` 明显高于 baseline，说明值得做 4096 env / 250 probe；
- 早停边界：
  - 60-80 iter 后 `undesired_contacts` 仍明显高于 per-joint/actual-own baseline 同步阶段；
  - `reward_contact` 或 `reward_hit_ball_velocity_net` 坍塌；
  - 出现异常 termination/NaN；
  - 满足以上任一条件则停止并回退右臂 action scale 或 reward 权重。

stage-2 moderated 250 probe 结果：

```text
run: logs/a3_table_tennis/2026-06-27_00-04-14_a3_stage2_moderated_250
checkpoint: model_249.pt

reward_contact tail10:                    0.036530
reward_future_landing_x_progress tail10:  0.046880
penalty_future_own_landing tail10:       -0.040983
penalty_actual_own_table_after_hit tail10:-0.087401
reward_hit_ball_velocity_net tail10:      0.044786
reward_hit_net_clearance_progress tail10: 0.000000
reward_future_pass_net tail10:            0.000003
reward_table_success tail10:              0.000000
undesired_contacts tail10:               -0.056910
termination_penalty tail10:              -1.995988
```

policy outcome diagnostics：

```text
first_hit:                 98/128 (0.766)
hit_vx_positive:           87/128 (0.680)
hit_vz_positive:           90/128 (0.703)
actual_crossed_net:        92/128 (0.719)
actual_net_clear:           0/128 (0.000)
opponent_table_after_hit:   0/128 (0.000)
own_table_after_hit:       98/128 (0.766)
hit_vx mean:                0.6313
hit_vy mean:                0.5092
hit_vz mean:                0.8450
actual_z_at_net mean:       0.8346
actual_z_at_net max:        1.0230
```

判断：

- stage-2 moderated 提高触拍率和过 x=0 比例，但没有提高真实 `hit_vx` 或 `actual_z_at_net`；
- 横向速度 `hit_vy mean=0.5092` 偏大，说明拍面/击球方向仍存在侧向分量；
- 继续长训风险高：策略可能强化“更稳定地碰球/过 x=0，但仍先落己方桌”的局部最优；
- 下一步回到默认右腕/右肘/拍面几何搜索，优先找 `hit_vx` 更高、`abs(hit_vy)` 更低、`actual_z_at_net` 更高的静态候选，再决定是否替换默认姿态。

下一步 A3 right-arm pose grid：

```text
固定 base pose / ball route；
扩大 right_elbow、right_wrist_roll、right_wrist_pitch、right_wrist_yaw 搜索；
CSV 后处理时使用 forward-oriented score：
  - 奖励 hit_vx 正且更大；
  - 惩罚 abs(hit_vy)；
  - 奖励 actual_z_at_net 接近或高于 1.0；
  - 保留 actual_net_clear/opponent_table_after_hit 为硬正向信号；
  - own_table_after_hit 作为负信号，但不单独作为排序依据。
```

right-arm grid / zero-action 对照结果：

```text
grid csv: logs/a3_table_tennis/a3_right_arm_grid_forward_20260627_0026.csv

best deterministic forward candidate:
  id=307
  re=0.55, rwr=-0.30, rwp=-1.40, rwy=-1.40
  hit_vx=1.4007, hit_vy=0.0900, hit_vz=0.6556, actual_z_at_net=0.9350

id=307 random-ball zero-action:
  first_hit: 88/128
  hit_vx mean: 0.7222
  hit_vy mean: 0.3466
  hit_vz mean: 0.8015
  actual_z_at_net mean: 0.8117

configured default random-ball zero-action:
  first_hit: 92/128
  hit_vx mean: 0.7428
  hit_vy mean: 0.2489
  hit_vz mean: 0.9016
  actual_z_at_net mean: 0.8542
```

判断：

- 确定性网格中的单点姿态不一定能泛化到随机球路；
- 当前默认姿态在随机球路下反而更稳，不应直接替换；
- A3 的主要瓶颈不是静态默认拍面，而是主动挥拍速度不足；
- T1 的 TT `action_scale=0.25`，A3 当前右肩/右肘仍偏保守，同时 A3 wrist pitch/yaw 只有约 6Nm 额定扭矩，不适合作为主要发力关节。

## 2026-06-27 A3 stage-3 shoulder/elbow swing 计划

目标：

- 仍不改 T1；
- 仍不改球路和成功标准；
- 不替换默认静态姿态；
- 只改变 A3 per-joint action scale 的右臂分配，让肩/肘发力、腕部少动。

拟修改：

```text
A3_ACTION_SCALE_BY_JOINT：
  right_shoulder_pitch/roll/yaw: 0.14 -> 0.22
  right_elbow:                  0.18 -> 0.25
  right_wrist_roll:             0.20 -> 0.12
  right_wrist_pitch/yaw:        0.20 -> 0.08

reward 暂不继续加大，沿用 stage-2 moderated：
  reward_contact=110
  reward_hit_ball_velocity_net=160
  reward_hit_net_clearance_progress=30
```

验证边界：

- 64 env / 20 iter smoke；
- 若 early `undesired_contacts` 明显超过 stage-2 moderated 且没有任务信号提升，则回退；
- 若 hit-velocity/x-progress 不高于 stage-2 moderated，也不进入 250；
- 若 smoke 通过，再跑 4096 env / 250 probe，并用物理诊断比较 `hit_vx`、`hit_vy`、`actual_z_at_net`。

stage-3 smoke 结果：

```text
run: logs/a3_table_tennis/2026-06-27_00-32-01_a3_stage3_shoulder_elbow_smoke
num_envs: 64
max_iterations: 20

reward_contact tail10:                    0.066327
reward_future_landing_x_progress tail10:  0.078452
penalty_future_own_landing tail10:       -0.071667
penalty_actual_own_table_after_hit tail10:-0.271667
reward_hit_ball_velocity_net tail10:      0.051363
reward_hit_net_clearance_progress tail10: 0.000000
reward_future_pass_net tail10:            0.000000
reward_table_success tail10:              0.000000
undesired_contacts tail10:              -79.077016
termination_penalty tail10:              -0.050000
```

判断：

- `reward_hit_ball_velocity_net` 低于 stage-2 moderated 的 `0.061982`；
- `undesired_contacts` 高于 stage-2 moderated 的 `-71.865002`；
- 不进入 250 probe；
- 回退到 stage-2 moderated action scale，作为当前较稳的 A3 训练配置。

## 2026-06-27 A3 long training 计划

用户判断：

- 250/500 iter 不足以代表最终是否能学会；
- T1 原始流程可以直接长训学到，A3 也应先做一次同尺度长训验证；
- 挥拍 curriculum 不应在没有长训验证前被视为必须项。

执行方案：

```text
task: a3_tt
num_envs: 4096
max_iterations: 10000
run_name: a3_stage2_moderated_long_10000
resume: no
predictor: yes
logger: tensorboard
```

观察重点：

- `Episode_Reward/reward_contact`
- `Episode_Reward/reward_hit_ball_velocity_net`
- `Episode_Reward/reward_hit_net_clearance_progress`
- `Episode_Reward/reward_future_pass_net`
- `Episode_Reward/reward_table_success`
- `Episode_Reward/penalty_actual_own_table_after_hit`
- `Episode_Reward/undesired_contacts`
- checkpoint 后用 `a3_hit_outcome_diagnostics` 检查 `actual_net_clear`、`opponent_table_after_hit`、`own_table_after_hit`。

判断边界：

- 如果 1000/2000/3000 iter 后 `reward_future_pass_net` 和 `reward_table_success` 仍长期为 0，再考虑 A3 专项主动挥拍 curriculum；
- 如果长训中出现稳定性崩溃或 NaN，应停止并回看动作尺度/PD/奖励。

## 2026-06-27 A3 stage-4 stable-return 调整计划

长训后现象：

- `reward_contact` 在约 2065 iter 达峰后快速下跌；
- `reward_future_pass_net` / `reward_table_success` 在约 2400 iter 附近出现峰值，随后衰减；
- `Policy/mean_noise_std` 从 `0.10` 增长到 `1.18` 左右，峰值区间与任务信号断崖高度重合；
- 确定性诊断下，`model_2500.pt` 和 `model_9999.pt` 在随机球路仍能高比例触球/过网，但几乎总是先落己方桌，且最终模型击球后低姿态 reset。

原因判断：

- TensorBoard 训练曲线使用 PPO 采样动作，受动作 std 影响；诊断脚本使用 inference policy，主要反映网络均值动作；
- A3 的击球窗口比 T1 更窄，高探索噪声会把已学到的触球策略冲散；
- 当前 dense shaping 对“碰到球、抬高、过网趋势”奖励偏强，但对“对面桌真实落点”和“击球后不摔”约束不够强；
- `own_table_after_hit` 是主失败模式，当前权重不足以压过 contact / hit velocity / x-progress 等正奖励。

边界：

- 不改 T1；
- 保留 `a3_tt` 作为 stage-2 moderated 长训基线；
- 新增 `a3_tt_stable` / `a3_tt_stable_eval` 作为 A3-only 试验任务，避免覆盖已有结果；
- train workflow 仅增加可选参数，默认行为保持不变。

拟修改：

```text
新增 task:
  a3_tt_stable
  a3_tt_stable_eval

A3 stable reward:
  reward_contact: 110 -> 120
  reward_future_landing_x_progress: 120 -> 60
  reward_hit_ball_velocity_net: 160 -> 80
  reward_hit_net_clearance_progress: 30 -> 20
  reward_future_opponent_landing: 120 -> 250
  reward_future_pass_net: 100 -> 120
  reward_table_success: 100 -> 350
  penalty_actual_own_table_after_hit: -80 -> -300
  add reward_actual_opponent_table_target: +250
  add penalty_hit_low_base_reset: -150

A3 stable PPO:
  entropy_coef: 0.006 -> 0.0005
  learning_rate: 1e-3 -> 5e-4
  desired_kl: 0.01 -> 0.006
```

训练入口增强：

```text
--load_optimizer / --no_load_optimizer
--reset_policy_noise_std VALUE
```

用途：

- 从 `model_2500.pt` 接续时可不加载旧 optimizer；
- 可把 policy std 从约 `1.18` 重置到 `0.15`，验证断崖是否由过高探索噪声触发。

验证：

```text
1. py_compile
2. 64 env / 20 iter smoke
3. 若 smoke 正常，再用 4096 env / 500 iter 从 model_2500.pt 接续 probe
4. 诊断对比:
   - first_hit 不应显著低于 2500 baseline
   - own_table_after_hit 应下降
   - opponent_table_after_hit 应上升
   - reset_low_z 应下降
```

## 2026-06-27 A3 stage-4b reward rollback 计划

stage-4 stable 4096-env probe 结果：

- `Policy/mean_noise_std` 被成功压在 `0.15` 附近；
- 但 `reward_future_pass_net`、`reward_table_success`、`reward_actual_opponent_table_target` 在 2500-2573 iter 全程几乎为 0；
- `termination_penalty` 稳定为 `-2.0`；
- 判断：低探索噪声方向正确，但 stable reward 的强惩罚/强终局目标让大批量训练失去早期任务信号。

stage-4b 原则：

- 不改 T1；
- 不改原始 `a3_tt`；
- 不覆盖失败对照 `a3_tt_stable`；
- 新增 `a3_tt_stage4b` / `a3_tt_stage4b_eval`；
- 保留低 entropy / 低 std 接续；
- 回调 reward：先保住“挥拍、触球、过网”课程，再逐步加“己方落台/击球后摔倒”约束。

拟定 reward：

```text
reward_contact: 110 -> 125
reward_future_landing_x_progress: 120 -> 110
reward_hit_ball_velocity_net: 160 -> 190
reward_hit_net_clearance_progress: 30 -> 70
reward_future_pass_net: 100 -> 160
reward_future_opponent_landing: 120 -> 160
reward_table_success: 100 -> 180
reward_actual_opponent_table_target: 0 -> 80

penalty_future_own_landing: -40 -> -30
penalty_actual_own_table_after_hit: -80 -> -40
penalty_hit_low_base_reset: 0 -> -20
```

拟定 PPO：

```text
entropy_coef=0.0005
learning_rate=5e-4
desired_kl=0.006
```

验证顺序：

```text
1. py_compile
2. 64 env / 20 iter smoke
3. 从 model_2500.pt 接续，4096 env / 100 iter probe
4. 若 100 iter 内 pass_net/table_success 仍为 0，则停止并继续调 reward；若出现信号，再扩到 500 iter
```

## 2026-06-27 A3 stage-4c short-horizon post-hit shaping 计划

stage-4b probe 结果：

- `reset_policy_noise_std=0.15`：到 2549 iter 时 `reward_future_pass_net`、`reward_table_success`、`reward_actual_opponent_table_target` 仍为 0，`termination_penalty` 接近 `-2.0`；
- `reset_policy_noise_std=0.05`：到 2529 iter 时 `Policy/mean_noise_std` 稳定在约 `0.051`，但 `reward_future_pass_net` 峰值仅约 `9.7e-7`，对桌信号仍为 0；
- 同一 `model_2500.pt` 的确定性诊断结果为 `first_hit=251/256`、`actual_net_clear=230/256`、`opponent_table_after_hit=27/256`、`own_table_after_hit=251/256`。

原因判断：

- A3 当前不是完全碰不到球，而是确定性均值策略能触球并过网，但球几乎总是先落己方桌；
- PPO 采样动作会打散 A3 的窄击球窗口，即使 std 降到 `0.05`，一帧式 first-hit shaping 仍太稀疏；
- 继续单纯增加 own-table 或稳定惩罚，会让早期有效击球梯度更少。

拟修改：

```text
新增 A3-only task:
  a3_tt_stage4c
  a3_tt_stage4c_eval

新增 A3-only 使用的 reward:
  reward_post_hit_net_progress

reward_post_hit_net_progress 触发条件:
  has_touch_paddle=True
  尚未落己方桌/对方桌
  球仍向 +x 方向运动

reward_post_hit_net_progress 评分:
  前向速度 vx 接近目标
  x 从 A3 侧朝网口推进
  预测/当前过网高度接近目标
  y 方向偏差不过大
  已经实际越过网口且高度足够时给少量 bonus

stage4c PPO:
  继续低 entropy / 低 std 接续
  entropy_coef 进一步降到 0.0001
  learning_rate 降到 3e-4
  desired_kl 降到 0.004
```

边界：

- 不修改 T1 配置；
- 不覆盖 `a3_tt` / `a3_tt_stage4b`；
- 不改变默认 train 行为；
- stage4c 先 smoke，再从 `model_2500.pt` 以 `--reset_policy_noise_std 0.03-0.05` 短探针验证。

stage4c 1024-env / 50-iter 结果：

- TensorBoard 任务信号被救回：`reward_future_pass_net recent10≈0.077`，`reward_table_success recent10≈0.038`；
- 但 `model_2549.pt` 确定性诊断显著退化：
  - `first_hit=113/256`，低于 `model_2500.pt` 的 `251/256`；
  - `actual_net_clear=3/256`，低于 `model_2500.pt` 的 `230/256`；
  - `hit_x mean=-0.855`，明显晚于原始 `model_2500.pt` 的约 `-1.334`；
  - `hit_vz mean=0.082`，明显低于原始 `model_2500.pt` 的约 `2.855`。

判断：

- stage4c 的 post-hit `x_score` 给了“球已经靠近网口”的正反馈，导致 PPO 倾向于更晚触球；
- 这会让 TensorBoard 的短期 sampled reward 看起来更好，但确定性策略变差；
- 下一版需要限制 late-hit 奖励，并把 post-hit shaping 从 `x_progress` 转向 `vz / net height / predicted landing`。

## 2026-06-27 A3 stage-4d conservative post-hit shaping 计划

拟修改：

```text
新增 A3-only task:
  a3_tt_stage4d
  a3_tt_stage4d_eval

reward_post_hit_net_progress 增加参数:
  max_reward_x
  vz_target
  vz_weight

stage4d post-hit reward:
  weight: 35 -> 18
  max_reward_x: -1.05
  x_weight: 0.20 -> 0.00
  vz_weight: 0.00 -> 0.20
  z_weight: 0.20 -> 0.30
  landing_weight: 0.25
  y_weight: 0.10

stage4d PPO:
  entropy_coef=0.00005
  learning_rate=1e-4
  desired_kl=0.002
  num_learning_epochs=3
```

验证：

```text
1. py_compile
2. 64 env smoke
3. 128 env resume smoke from model_2500.pt
4. 1024 env / 50 iter probe
5. 必须用物理诊断确认 model_25xx 不再比 model_2500 明显退化，才允许扩大并发或长训
```

stage4d 1024-env / 50-iter 结果：

- TensorBoard 任务信号明显恢复：
  - `reward_future_pass_net recent10≈0.216`
  - `reward_table_success recent10≈0.060`
  - `reward_future_opponent_landing recent10≈0.089`
- `model_2549.pt` 物理诊断：
  - `first_hit=226/256`
  - `actual_net_clear=198/256`
  - `opponent_table_after_hit=31/256`
  - `own_table_after_hit=226/256`
  - `reset_low_z=2/256`

判断：

- stage4d 成功避免 stage4c 的 late-hit 崩坏，并显著改善击球后稳定性；
- 但 almost all hits 仍先落己方桌，说明下一步应该小幅加强 first-landing / own-table 约束；
- 不允许直接长训，必须先做 stage4e 小探针。

## 2026-06-27 A3 stage-4e own-table ramp 计划

拟修改：

```text
新增 A3-only task:
  a3_tt_stage4e
  a3_tt_stage4e_eval

基于 stage4d:
  reward_future_opponent_landing: 160 -> 220
  penalty_future_own_landing: -30 -> -60
  penalty_actual_own_table_after_hit: -40 -> -80
  reward_table_success: 180 -> 220
  reward_actual_opponent_table_target: 80 -> 120
  penalty_hit_low_base_reset: -20 -> -40

保持:
  stage4d conservative post-hit reward
  stage4d PPO low lr / low entropy / low KL
```

验证边界：

- 如果 `first_hit` 或 `actual_net_clear` 明显低于 stage4d，则回退；
- 如果 `own_table_after_hit` 不下降但 `opponent_table_after_hit` 不升，说明需要几何/球路/击球姿态再校正，而不是继续加罚；
- 如果 `undesired_contacts` 或 reset 重新上升，说明稳定惩罚过强或动作空间仍不匹配。

## 2026-06-27 own-table 诊断口径修正

发现：

- `a3_hit_outcome_diagnostics.py` 当前的 `own_table_after_hit` 统计使用了 `env.has_touch_own_table_prev`；
- 该 flag 可能在球被机器人击打前就因为来球先弹己方桌而变为 True；
- 因此旧诊断里的 `own_table_after_hit≈first_hit` 可能不是“击球后又落己方桌”，而是把合法/预期的来球己方弹跳也算进去了。

拟修改：

```text
own_table_after_hit:
  改为只统计 first_hit_seen 之后的 has_touch_own_table_just_now
  并要求击球后先离开 own-table contact 区域，再次进入才计为 post-hit own-table

新增/保留区分:
  own_table_before_hit
  opponent_table_after_hit
  own_table_after_hit
```

边界：

- 只改 A3 诊断脚本；
- 不改训练环境、不改 reward、不改 T1；
- 重新诊断 stage4d model_2549 和 stage4e model_2598 后，再决定是否继续加 own-table 惩罚。

修正后复查：

```text
stage4d model_2549:
  first_hit=226/256
  actual_net_clear=198/256
  opponent_table_after_hit=31/256
  own_table_after_hit=220/256
  reset_low_z=2/256

stage4e-from-stage4d model_2598:
  first_hit=238/256
  actual_net_clear=218/256
  opponent_table_after_hit=22/256
  own_table_after_hit=214/256
  reset_low_z=60/256
```

判断：

- stage4e 加重 own-table / opponent-table 权重后，没有提升真实 opponent table 命中，反而显著增加低姿态 reset；
- 继续单纯加大 own-table 惩罚不是好方向；
- 需要在 stage4d 稳定基线基础上，引入“击球后弹道落点目标”的 dense shaping，直接约束预计落点到对面桌范围。

## 2026-06-27 A3 stage-4f ballistic landing target 计划

拟修改：

```text
新增 A3-only reward:
  reward_post_hit_ballistic_landing_target

触发条件:
  has_touch_paddle=True
  尚未发生首次桌面接触
  球向 +x 运动

评分:
  用当前 ball_pos / ball_linvel 估计无阻力落到桌面高度 z=0.78 的 x/y
  奖励落点靠近 opponent table target: x≈1.15, y≈0.0
  对过短、过远、横向偏离给低分

新增 A3-only task:
  a3_tt_stage4f
  a3_tt_stage4f_eval

基于 stage4d:
  保留 stage4d conservative post-hit reward
  不继承 stage4e 的强 own-table ramp
  增加 ballistic landing target reward，初始 weight=60
  PPO 继续使用 stage4d 的低 lr / 低 entropy / 低 KL
```

验证：

```text
1. py_compile
2. stage4f 128-env smoke from stage4d model_2549
3. 若 table_success / opponent_landing 有信号且 reset 不恶化，再跑 1024-env 50 iter
4. 诊断标准：opponent_table_after_hit 必须高于 stage4d 的 31/256，reset_low_z 不应高于 stage4d 太多
```

stage4f 1024-env / 50-iter 诊断：

```text
run: 2026-06-27_22-50-08_a3_stage4f_from4d2549_std003_1024_50
checkpoint: model_2598.pt

first_hit=239/256
actual_net_clear=226/256
opponent_table_after_hit=29/256
own_table_after_hit=225/256
reset_low_z=46/256
hit_vx mean=4.8751
hit_vy mean=1.9219
hit_vz mean=3.3230
actual_z_at_net mean=1.5643
```

判断：

- stage4f 的 TensorBoard 曲线更好，`reward_table_success` 末段约 0.07-0.08；
- 但独立随机球诊断没有超过 stage4d 的真实 opponent table 命中，且低身体 reset 明显增多；
- 说明 ballistic landing dense reward 有方向价值，但 weight=60 过强或过早，容易诱导 A3 通过压低身体/大幅动作换取球路奖励；
- 下一步不直接长训 stage4f，而是做更保守的 stage4g。

## 2026-06-27 A3 stage-4g conservative hit/pass-net rollback 计划

拟修改：

```text
新增 A3-only reward:
  penalty_post_hit_low_base

作用:
  只在已经触拍后生效
  当 robot base z 低于温和阈值时给连续惩罚
  用于提前抑制 stage4f 出现的低身体 reset，而不是只等 reset 后惩罚

新增 A3-only task:
  a3_tt_stage4g
  a3_tt_stage4g_eval

基于 stage4d:
  保留 stage4d conservative post-hit reward
  保留低 lr / 低 entropy / 低 KL PPO
  略增强 hit velocity / net-clear / future-pass-net 主线
  ballistic landing target 从 60 降到 25，作为辅助而不是主目标
  own-table / actual opponent table 权重保持温和，避免 stage4e 的强惩罚副作用
  新增 post-hit low-base penalty，weight 初始小，避免一开始就压死探索
```

验证边界：

- 先从 stage4d `model_2549.pt` resume，`--no_load_optimizer --reset_policy_noise_std 0.03`；
- 先跑 128-env smoke，再跑 1024-env / 50 iter；
- 合格条件不是 TensorBoard 单项变好，而是独立诊断同时满足：
  - `opponent_table_after_hit` 高于 stage4d 的 31/256；
  - `reset_low_z` 明显低于 stage4f 的 46/256；
  - `first_hit` 和 `actual_net_clear` 不明显低于 stage4d/stage4f；
- 如果 stage4g 仍不能提升真实对面落台，应回到几何/击球姿态或球路 curriculum，而不是继续堆 reward。

stage4g 1024-env / 50-iter 诊断：

```text
run: 2026-06-27_23-01-38_a3_stage4g_from4d2549_std003_1024_50
checkpoint: model_2598.pt

first_hit=240/256
actual_net_clear=221/256
opponent_table_after_hit=32/256
own_table_after_hit=223/256
reset_low_z=47/256
reset_y_high=27/256
hit_vx mean=4.8951
hit_vy mean=1.8617
hit_vz mean=3.1833
actual_z_at_net mean=1.5330
```

判断：

- stage4g 只比 stage4d 的 `opponent_table_after_hit=31/256` 略高 1 个环境，提升不具备稳定意义；
- 低身体 reset 仍接近 stage4f，说明单靠小的 low-base penalty 不足以恢复稳定；
- 关键球路问题变清楚：球已能高速过网，但过网高度过高、横向速度过大，导致大量球不落对面有效桌面；
- 下一步不继续增大 ballistic landing reward，而是新增“击球后球路过高/横向过快”惩罚，并继续从 stage4d 稳定模型重新 resume。

## 2026-06-27 A3 stage-4h post-hit arc/lateral constraint 计划

拟修改：

```text
新增 A3-only reward:
  penalty_post_hit_trajectory_excess

作用:
  只在触拍后、首次桌面接触前生效
  估计球到网平面时的 z_at_net
  惩罚 z_at_net 明显高于目标上界
  惩罚 |ball_vy| 明显过大

新增 A3-only task:
  a3_tt_stage4h
  a3_tt_stage4h_eval

基于 stage4g:
  降低 hit velocity 中 vz 分量的诱导
  加强 z_at_net 靠近合理过网高度，而不是越高越好
  ballistic landing target 继续降权，只做辅助
  post-hit low-base penalty 适度增加，但仍不使用 stage4e 那种强 own-table ramp
```

验证边界：

- 从 stage4d `model_2549.pt` 重新 resume，不从已经低身体化的 stage4f/g resume；
- 先 128-env smoke，再 1024-env / 50 iter；
- 合格条件：
  - `actual_z_at_net mean` 应从 stage4g 的 1.533 明显下降；
  - `hit_vy mean` 应从 stage4g 的 1.862 下降；
  - `reset_low_z` 应明显低于 stage4f/g 的 46-47/256；
  - `opponent_table_after_hit` 不低于 stage4d 的 31/256，理想情况下明显超过。

stage4h 1024-env / 50-iter 诊断：

```text
run: 2026-06-27_23-09-58_a3_stage4h_from4d2549_std003_1024_50
checkpoint: model_2598.pt

first_hit=234/256
actual_net_clear=220/256
opponent_table_after_hit=24/256
own_table_after_hit=217/256
reset_low_z=41/256
reset_y_high=22/256
hit_vx mean=5.0561
hit_vy mean=1.9997
hit_vz mean=3.2941
actual_z_at_net mean=1.5454
```

判断：

- stage4h 的轨迹惩罚在 TensorBoard 上有稳定信号，但没有把真实球路拉回合理区间；
- 与 stage4g 相比，低身体 reset 略降，但对面落台从 32/256 降到 24/256；
- `actual_z_at_net` 和 `hit_vy` 仍然偏高，说明当前策略更倾向于用大挥拍/高弧线获得 hit/pass-net 奖励；
- 当前阶段不建议继续在同一个随机球路上堆 reward 权重；
- 下一步应做 A3 专用 curriculum 或几何回查：
  1. 固定/窄化球路，先训练中心来球的低弧线对面落台；
  2. 可视化/诊断击球瞬间拍面法向、球拍中心、球-拍接触点，确认是否拍面角度天然给球过多上旋/上抛；
  3. 在稳定中心球能达到较高 opponent-table 后，再逐步放宽 `ball_speed_y_range` / `ball_pos_y_range`；
  4. 暂不建议从 stage4f/g/h 的 `model_2598.pt` 继续长训，优先使用 stage4d `model_2549.pt` 作为稳定回退点。

## 2026-06-29 A3 trained-policy WebRTC visualization fix 计划

现象：

- 使用 `legged_lab.scripts.play --task=a3_tt_stage4d --livestream 2` 加载已训练 checkpoint 时，WebRTC 客户端黑屏；
- Isaac 日志显示场景、Robot/Table/Ball 已初始化，WebRTC 曾连接成功，因此优先判断为 viewport camera 没有被设置到有效视角，而不是模型加载或 A3 asset 失败；
- `preview.py` 已显式调用 `env.sim.set_camera_view([-3.0, 6.0, 1.0], [-3.0, 0.0, 1.0])`，但 `play.py` 没有类似逻辑。

拟修改：

- 仅修改 `legged_lab/scripts/play.py`；
- 增加 `--camera_eye x y z` 与 `--camera_target x y z` 可选参数；
- 在非 headless 或启用 livestream 时，创建环境后设置一次 viewport camera；
- 默认相机位置复用 `preview.py` 的桌球可视化视角；
- 不修改训练、评估、奖励函数、A3/T1 task 配置和 checkpoint 加载流程。

验证：

- `py_compile legged_lab/scripts/play.py`；
- 使用原 play 命令重新打开 WebRTC，确认不再黑屏；
- 如默认视角仍不理想，可通过命令行调整 `--camera_eye` / `--camera_target`，不需要再改代码。

## 2026-06-29 A3 play WebRTC step-loop render/update 补充计划

复测现象：

- 初始化阶段可以看到 Isaac Sim 界面；
- 进入策略 play / env.step 循环后 WebRTC 变黑；
- 最新 Isaac 日志未显示仿真崩溃或 asset 加载失败，WebRTC 是连接后又断开；
- 项目中已有 `tt_webrtc_probe.py`，该脚本专门通过固定 camera、每步 `env.sim.render()`、每步 `simulation_app.update()`、适当 sleep 来避免 WebRTC 在仿真循环开始后黑屏。

补充判断：

- 第一次 camera patch 只解决“相机没指向场景”的问题；
- 当前更像是 play loop 跑得太快，Kit/WebRTC 没有稳定刷新远程 viewport；
- 这属于可视化脚本调度问题，不应修改 A3/T1 环境物理、奖励、PD、reset 或训练 workflow。

拟修改：

- 继续仅修改 `legged_lab/scripts/play.py`；
- 增加 WebRTC/GUI 可视化循环参数：
  - `--keep_camera_interval`：默认每步重新固定 camera，设为 0 可关闭；
  - `--visualize_sleep`：默认在 livestream 下使用短 sleep，给 WebRTC 编码/传输留时间；
- 在 play 循环每次 `env.step(actions)` 后，对 livestream/GUI 模式执行：
  - 按间隔重新设置 camera；
  - 显式 `env.sim.render()`；
  - 显式 `simulation_app.update()`；
  - 可选 sleep；
- 默认 camera 改用 `tt_webrtc_probe.py` 已使用的 TT/A3 WebRTC 视角：
  - eye `[-3.2, -2.0, 1.6]`
  - target `[-1.8, 0.35, 0.85]`
- 这些逻辑只在非 headless 或 livestream 时生效，不影响训练命令。

验证：

- `py_compile legged_lab/scripts/play.py`；
- 用原 WebRTC play 命令复测；
- 如果仍黑屏，再用 `tt_webrtc_probe.py` 做 zero-action 对照，确认是 play policy loop 还是底层 WebRTC/renderer 问题。

## 2026-06-29 A3 play/probe 差异收敛计划

复测结论：

- `tt_webrtc_probe.py --task=a3_tt_stage4d --mode=zero --livestream 2` 可以正常看到画面；
- `play.py --task=a3_tt_stage4d --load_run ... --checkpoint ... --predictor --livestream 2` 仍黑屏；
- 因此 Isaac/WebRTC/A3 asset 本身可用，问题收敛到 `play.py` 特有路径。

关键差异：

- `play.py` 会在可视化前导出 predictor/JIT/ONNX，日志中固定出现 ONNX version converter traceback；
- `tt_webrtc_probe.py` 不导出模型；
- `tt_webrtc_probe.py` 设置 camera 时会加上 `env.scene.env_origins[0]`，而当前 `play.py` 直接使用输入 camera 坐标；
- `tt_webrtc_probe.py` 有明确的有限步数调试入口，当前 `play.py` 只能一直运行，不方便短跑定位。

拟修改：

- `play.py` 增加 `--skip_export`，可视化时跳过 predictor/JIT/ONNX 导出；
- `play.py` 增加 `--max_play_steps`，用于短跑验证；
- `play.py` 的 camera 设置逻辑改为与 `tt_webrtc_probe.py` 一致：将 eye/target 视为 env-local 坐标并加上 `env.scene.env_origins[0]`；
- 保持默认不跳过导出，避免改变原有 play 导出行为；WebRTC 复测命令中显式传入 `--skip_export`。

验证：

- `py_compile legged_lab/scripts/play.py`；
- WebRTC 复测时优先使用 `--skip_export`；
- 如果 `--skip_export` 后仍黑屏，再进入 policy action/mesh NaN 的下一层诊断。

## 2026-06-29 A3 play action/predictor isolation 计划

复测结论：

- `play.py --skip_export` 后仍黑屏；
- 最新日志不再出现 ONNX traceback，但 WebRTC 仍断开；
- `play.py` 进程没有崩溃，说明不是 Python fatal error，而是策略 play 循环下视频流没有有效画面；
- `tt_webrtc_probe.py` 能看，说明 zero-action 环境 step + render/update 这条路径本身可用。

下一层差异：

- `play.py` 使用训练好的 policy action，`tt_webrtc_probe.py` 默认使用 zero action；
- `play.py --predictor` 每步调用 `_record_ball_positions()` / `_maybe_predict_and_update_env()`，probe 不调用；
- `tt_webrtc_probe.py` 在进入 step loop 前有 warmup render，play 当前没有。

拟修改：

- `play.py` 增加 `--action_mode policy|zero|random|sine`：
  - 默认 `policy`，保持原 play 行为；
  - `zero` 用于“加载 checkpoint 但动作置零”的 WebRTC 对照；
- `play.py` 增加 `--disable_predictor_update`：
  - 加载 predictor runner 但不在每步更新预测 marker；
  - 用于隔离 BallPred/BallFuture marker 或 predictor update 是否影响渲染；
- `play.py` 增加 `--warmup_render_steps`：
  - livestream 下默认 warmup 若干帧，和 `tt_webrtc_probe.py` 对齐；
- 不修改训练、奖励、环境 reset、PD 或 checkpoint 内容。

验证矩阵：

1. `--action_mode zero --disable_predictor_update --skip_export`：
   - 若可见，说明 runner 加载不是问题，继续查 policy action；
2. `--action_mode policy --disable_predictor_update --skip_export`：
   - 若黑屏，说明策略动作导致渲染/仿真异常；
3. `--action_mode policy --skip_export`：
   - 若只有这个黑屏，说明 predictor marker/update 路径有问题。

## 2026-06-29 A3 play env-headless / runner-load isolation 计划

新增观察：

- zero-action、禁用 predictor update、跳过 export 后，`play.py` 仍在 50-100 step 左右黑屏；
- `tt_webrtc_probe.py` 能正常显示；
- `tt_webrtc_probe.py` 创建环境时固定 `headless=False`；
- `play.py` 当前使用 `env_class(env_cfg, args_cli.headless)`，而 Isaac livestream 启动参数会进入 no-window/WebRTC 模式，环境内部可能仍被当成 headless 运行；
- 如果环境内部 `headless=True`，`TTEnv.step()` 内部不会走 `self.sim.render()`，与 probe 行为不一致。

拟修改：

- `play.py` 在 `--livestream > 0` 时，创建环境时强制 `env_headless=False`，使其与 `tt_webrtc_probe.py` 的渲染路径对齐；
- 打印 `args_headless`、`env_headless`、`livestream`，方便终端确认；
- 增加 `--no_load_runner`：
  - 完全跳过 runner/checkpoint/policy 加载；
  - 只允许 `--action_mode` 为 `zero/random/sine`；
  - 用于判断黑屏是否由 runner/checkpoint 加载本身引入。

验证：

1. `--no_load_runner --action_mode zero`：
   - 若可见，说明 play 环境配置已与 probe 对齐，下一步查 runner；
2. 加载 runner 但 `--action_mode zero`：
   - 若黑屏，说明 runner/checkpoint 加载影响 WebRTC；
3. 加载 runner 且 `--action_mode policy`：
   - 用于最终可视化策略。

## 2026-06-29 A3 WebRTC policy-probe 计划

复测结论：

- `play.py` 在多轮收敛后仍会黑屏，包括 zero-action、禁用 predictor update、跳过 export 的情况；
- 终端持续输出 `[Play] Success...`，说明仿真循环仍在运行，主要问题是 `play.py` 的 WebRTC 视频路径；
- `tt_webrtc_probe.py` 已确认可见，说明最可靠路线是把 policy 加载能力移植到这个已知可见的 probe，而不是继续扩大 `play.py` 的改动范围。

拟修改：

- 仅扩展 `legged_lab/scripts/tt_webrtc_probe.py`；
- `--mode` 增加 `policy`；
- 增加 `--load_run`、`--checkpoint`、`--predictor`、`--disable_predictor_update`；
- `policy` 模式下：
  - 使用 task 的 agent cfg 定位 `logs/<experiment_name>/<load_run>/<checkpoint>`；
  - 使用 `OnPolicyRunner` 或 `OnPolicyPredictorRegressionRunner` 加载 checkpoint；
  - 每步使用 `policy(obs)` 输出动作；
  - predictor 模式可选更新 env 中的 ball prediction；
  - 继续复用 probe 原有 camera/render/update/sleep 逻辑；
- 不修改训练、奖励、PD、reset、A3/T1 task 配置。

验证：

- `py_compile legged_lab/scripts/tt_webrtc_probe.py`；
- 先用 probe 的 `mode=zero` 确认原功能不变；
- 再用 `mode=policy --load_run ... --checkpoint ... --predictor` 查看训练模型可视化。

## 2026-06-29 A3 policy visualization standing failure 观察

WebRTC policy probe 已经可见：

```bash
python -m legged_lab.scripts.tt_webrtc_probe \
  --task=a3_tt_stage4d \
  --mode=policy \
  --load_run 2026-06-27_21-48-11_a3_stage4d_resume2500_std003_1024_50 \
  --checkpoint model_2549.pt \
  --predictor \
  --livestream 2
```

用户观察：

- 策略加载后画面可见；
- A3 几乎无法稳定站住，会很快摔倒；
- 终端仍有 hit 统计输出，说明策略不是完全不动，而是在尝试任务动作时破坏了身体稳定。

判断：

- 这已经不是 WebRTC/render 问题，而是当前 A3 policy 的真实行为问题；
- stage4d `model_2549.pt` 只能称为“当前 reward probe 中相对较好的回退点”，不是可部署的稳定打球策略；
- 之前 `reset_low_z` 诊断是短 rollout / 批量统计，不能替代单环境长时间可视化；
- A3 当前训练仍存在“为触球/过网牺牲站稳”的失败模式。

下一步验证：

1. `mode=zero + freeze_ball`：确认默认姿态和 PD 本身是否能站；
2. `mode=policy + freeze_ball + disable_predictor_update`：确认没有来球时 policy 是否仍会摔；
3. 对比 `model_2500.pt` 与 `model_2549.pt`：确认 stage4d 继续训练是否损坏了站稳；
4. 若 policy 无球也摔，需要重新引入 A3 standing/ready-pose curriculum 或更强的 base-stability 约束，再接乒乓球任务。

## 2026-06-29 A3 下肢 ready stance 专项计划

最新 WebRTC 观察：

- 上身和球拍方向暂时不作为本轮主要问题；
- A3 初始下肢看起来双腿偏并拢，站姿不符合稳定击球姿态；
- stage4d policy 可视化时会很快摔倒，因此继续单纯调 reward/球路收益有限；
- 当前 `A3_PINGPONG_READY_JOINT_POS` 的髋 roll / 踝 roll 符号与 `A3_STABLE_STANDING_JOINT_POS` 相反，可能把双腿向内收窄，这是本轮优先验证点。

边界：

- 不覆盖 `a3_tt`、`a3_tt_stage4d`、T1 任务和已有 checkpoint workflow；
- 不改上肢、球拍 offset、球路、reward 权重；
- 新增 A3-only 任务用于测试下肢 ready stance，确认稳定后再决定是否作为后续长训入口。

拟修改：

- 新增 `a3_tt_stage5_ready` / `a3_tt_stage5_ready_eval`；
- 复用 stage4d 的 reward 和 agent；
- 下肢默认姿态改为：
  - 髋 roll 使用稳定站姿方向，并略微加大外展；
  - 膝关节略加屈曲；
  - 踝 pitch/roll 配合脚底接地；
  - 根节点高度小幅下调，避免屈膝后脚悬空；
- 初期 reset 随机范围收窄：
  - base pose 保持在当前击球几何附近；
  - base velocity 置零；
  - locomotion joints 只允许极小扰动；
  - right-arm 扰动保持极小或为零，避免把已可接受的上肢姿态打散。

验证顺序：

1. `a3_tt_stage5_ready + mode=zero + freeze_ball`：只看默认姿态和 PD 是否能站；
2. `a3_tt_stage5_ready + mode=zero`：看来球/重置是否破坏站姿；
3. `a3_tt_stage5_ready + mode=policy + freeze_ball`：确认旧 stage4d policy 在新下肢默认下是否仍摔；
4. 如果 zero-action 能站但旧 policy 摔，下一步从 `a3_tt_stage5_ready` 重新短训 500-1000 iter，而不是继续沿用旧 checkpoint。

## 2026-06-29 A3 stage5b 早期任务 curriculum 计划

`a3_tt_stage5_ready` 1024-env 短训结果：

- 运行到 151 iter 后手动停止；
- `undesired_contacts` 从早期最差约 `-55` 收敛到约 `-0.1`；
- `penalty_hit_low_base_reset` 始终很小；
- `reward_contact` 有稳定但很低的信号；
- `reward_future_pass_net`、`reward_table_success`、`reward_actual_opponent_table_target` 到 151 iter 仍为 0；
- 说明下肢姿态解决了一部分站稳/非脚接触问题，但策略收敛到“稳定保守、任务打不开”的局部解。

边界：

- 保留 `a3_tt_stage5_ready`，不覆盖该诊断任务；
- 不修改 T1、不修改全局 TTEnv reset 逻辑；
- 不修改球拍 offset 和上肢默认姿态；
- 新增 A3-only `stage5b` 作为下一轮训练入口。

拟修改：

- 新增 `a3_tt_stage5b` / `a3_tt_stage5b_eval`；
- 继承 stage5 ready 下肢姿态；
- 适度提高右臂 action scale，腿部 action scale 保持不变，避免重新引入摔倒；
- 提高初始策略噪声，给右臂/腕部更多探索；
- 加强早期击球后轨迹 shaping：
  - `reward_contact` 稍增；
  - `reward_hit_ball_velocity_net` 更宽松地奖励朝对面飞；
  - `reward_hit_net_clearance_progress` 和 `reward_post_hit_net_progress` 提高；
  - `reward_future_pass_net` 加强但降低高度要求；
  - 暂时降低 own-table 相关惩罚，避免初期一碰球就被强惩罚压回保守策略；
- 保留 low-base 和 undesired-contact 托底。

验证：

- `py_compile` A3 config、env registry、diagnostic script；
- `tt_webrtc_probe --task=a3_tt_stage5b --mode=zero --freeze_ball --trial_steps=2 --headless` smoke test；
- 下一轮训练先 200-300 iter 观察：
  - `undesired_contacts` 是否保持低；
  - `reward_hit_ball_velocity_net` 是否高于 stage5_ready；
  - `reward_future_pass_net` 是否脱离 0；
  - `table_success` 是否出现非零。

## 2026-06-29 A3 curriculum 理解修正

用户指出：

- 不是把训练拆成“一个阶段只训练一个能力”的专用单项任务；
- PPO 表面上同步优化所有 reward，但由于奖励难易和密度不同，实际能力形成有隐含先后；
- 对 A3 乒乓球任务，站稳是所有后续能力的前提；
- 但站稳阶段也不能完全去掉球、击球点、球拍靠近等任务信息，否则可能训练出和真实击球任务不兼容的保守站立策略。

修正后的原则：

- 保持完整任务语境，不做纯 standing-only 训练作为最终入口；
- 所有关键 reward 可以同时存在，但需要有主次和坡度：
  - 始终保留站稳、姿态、非脚接触、低 base reset 等托底；
  - 早期保留击球点靠近、球拍接近、接触等稠密信号；
  - 过网、对面桌成功等稀疏高阶目标早期权重要低或更宽松；
  - 当稳定性和接触率上来后，再逐步增强击球方向、过网和落点；
- 避免两个极端：
  - 只训站稳，导致策略不会为了击球移动/伸臂；
  - 一开始强压完整成功任务，导致站稳还没学会就被稀疏 reward 噪声牵引。

对当前 A3 结果的解释：

- `stage5_ready` 更像验证默认下肢姿态和稳定性，而不是最终训练阶段；
- `stage5b` 证明保留任务信号后可以稳定碰球，但回球高度/速度不足；
- 下一步应在完整任务语境中做软 curriculum：
  - 保留下肢 ready stance；
  - 保持站稳托底；
  - 保留接触和靠近击球点；
  - 调整拍面/右腕与击球方向 reward，使“碰球”自然过渡到“向前向上击球”。

进一步明确：

- curriculum 的核心不是“某个阶段只训练一个能力”，而是 reward 主次随能力进展变化；
- 早期机器人容易摔倒时：
  - 摔倒、低 base、非脚接触、姿态不稳等惩罚需要更重；
  - 乒乓球任务信号仍保留，但不能压过稳定性；
- 当摔倒率明显下降后：
  - 逐步提高向球路/击球点移动的奖励；
  - 保持稳定性托底，防止策略为了追球重新牺牲站稳；
- 当能稳定靠近球路后：
  - 提高球拍接触、击球时机和击球中心相关奖励；
- 当能稳定击球后：
  - 再提高 `vx>0`、`vz`、过网高度和落点奖励；
- 最理想的切换依据不是固定 iter，而是能力指标：
  - low-base reset / undesired contact / body boundary reset；
  - paddle-contact rate；
  - hit-vx-positive / hit-vz-positive；
  - net-clear rate；
  - opponent-table rate。

## 2026-06-29 A3 stage5c 软课程任务计划

用户进一步明确：

- 训练不是拆成“只站稳、只追球、只击球”的孤立专用阶段；
- 所有 reward 表面上同步参与 PPO 优化，但不同 reward 的难度和稀疏度会让能力形成有先后；
- A3 当前最大的前提问题仍是稳定性，如果机器人早期频繁摔倒，追球和击球 reward 很难稳定生效；
- 因此需要在完整乒乓球任务中做软 curriculum：
  - 早期摔倒、低 base、非脚接触、姿态不稳惩罚更重；
  - 球路、球拍接近、触球和出球 reward 保留，但初期不压过稳定性；
  - 稳定性变好后逐步提高向球路/击球点移动的 reward；
  - 能靠近球路后逐步提高触球、向前向上击球、过网和落点 reward。

当前可用机制：

- 代码已有 `mdp.modify_reward_weight_linear`；
- 它可以按全局仿真 action step 线性修改 reward 权重；
- 先使用固定 step schedule，风险最低，不改 runner、不改 TTEnv；
- 后续如果 TensorBoard 指标稳定，可以再扩展成按能力指标触发的 curriculum。

拟新增任务：

- `a3_tt_stage5c`
- `a3_tt_stage5c_eval`

边界：

- 不修改 T1；
- 不修改 `a3_tt`、`a3_tt_stage5_ready`、`a3_tt_stage5b`；
- 不修改全局 reset / reward 函数实现；
- 不修改球拍模型、球拍 offset、A3 USD/URDF 资产；
- 只新增 A3-only config、task registry 和诊断脚本白名单。

stage5c 初始 reward 主次：

- 强化稳定性托底：
  - 更重 `termination_penalty`；
  - 更重 `undesired_contacts`；
  - 更重 `flat_orientation_l2`、脚姿态、脚过近惩罚；
  - 更重 `penalty_hit_low_base_reset`；
- 保留完整任务信号：
  - 保留 `reward_future_touch_point`、`reward_future_dis_ee`；
  - 保留 `reward_contact`，但初始低于 stage5b；
  - 保留击球后轨迹 reward，但初始更宽松、更低权重；
  - 保留过网和落点 reward，但初始较低；
- 暂时不让 own-table 惩罚过强，避免策略因为早期一碰球就被压回保守解。

stage5c curriculum 权重趋势：

- 早期到中期：
  - 稳定惩罚逐步回到正常强度；
  - `reward_future_touch_point`、`reward_future_dis_ee`、`reward_contact` 增强；
- 中期到后期：
  - `reward_hit_ball_velocity_net`、`reward_hit_net_clearance_progress`、`reward_post_hit_net_progress` 增强；
  - `reward_future_pass_net`、`reward_table_success`、`reward_actual_opponent_table_target` 增强；
  - own-table 惩罚逐步恢复到 stage5b 水平。

预期观察指标：

- 前 250-500 iter：
  - `undesired_contacts`、`termination_penalty`、`penalty_hit_low_base_reset` 不能发散；
  - `reward_future_dis_ee` / `reward_future_touch_point` 应保持非零；
- 500-1500 iter：
  - `reward_contact` 应逐步提高；
  - `reward_hit_ball_velocity_net` 和 `reward_post_hit_net_progress` 应明显高于 stage5_ready；
- 1500 iter 以后：
  - `hit_vx_positive` / `hit_vz_positive` 应提高；
  - `actual_net_clear` 应从 0 开始出现；
  - `opponent_table_after_hit` / `table_success` 是后期目标。

## 2026-06-29 A3 目标修正：T1 是基线，不是上限

当前共识：

- T1 的训练流程和最终 checkpoint 只能作为迁移参考；
- A3 不应完全照抄 T1，也不应把 T1 的最终效果作为目标上限；
- A3 的机器人结构、自由度、右腕、球拍安装方式、身体高度和动力学都不同，因此需要 A3 专用优化；
- 最终目标是利用 A3 的结构能力，做出比当前 T1 checkpoint 更好的乒乓球策略。

T1 对 A3 的作用：

- 提供原始 PACE 流程的正确性参考；
- 提供 reward 项、预测器、reset、PD target 控制链路的参考；
- 提供训练节奏参考：
  - 早期先稳定；
  - 中期打开触球；
  - 后期提高过网和落点；
- 提供最低可接受基线：
  - A3 至少要达到稳定触球；
  - 后续应超过 T1 的 success、过网质量和落点稳定性。

A3 不应机械对照 T1 的地方：

- 不直接复用 T1 的默认姿态；
- 不直接复用 T1 的 action scale 和 policy noise；
- 不直接复用 T1 的 reset 随机范围；
- 不直接复用 T1 的 reward 权重比例；
- 不把 T1 的最终 checkpoint 当作最优训练结果，因为 T1 曲线本身在 4000-5500 iter 更强，9999 iter 已有回落。

A3 的优化目标分层：

1. 稳定性超过 T1：
   - 更少低 base reset；
   - 更少非脚接触；
   - 击球前后都能保持站姿；
2. 触球率达到或超过 T1：
   - 先追求稳定 `hit`；
   - 击球中心和球路几何要更一致；
3. 回球质量超过 T1：
   - 击球后 `vx` 更稳定为正；
   - `z_at_net` 能稳定高于球网；
   - 减少 own-table-after-hit；
4. 落点成功率超过 T1：
   - 逐步提高 `table_success`；
   - 落点集中在对面桌有效区域；
5. 策略鲁棒性超过 T1：
   - 更宽球路；
   - 更大 y 向变化；
   - 更稳定 predictor 输入扰动和 reset 随机化。

后续评估原则：

- 不能只看最终 checkpoint；
- 要保存并评估多个中间 checkpoint；
- 对 A3 应同时看：
  - train reward；
  - hit rate；
  - success rate；
  - low-base / termination；
  - first-hit 位置；
  - hit 后 `vx/vz`；
  - net clear；
  - opponent table landing；
- T1 曲线用于判断训练节奏，但 A3 目标要高于 T1。

## 2026-06-30 A3 stage5f：能力触发 curriculum 与解除早期门控计划

stage5d/stage5e 结果复盘：

- `stage5d` 最终 `TT_hit_rate` 从早期约 `0.67` 掉到约 `0.003`，`TT_success_rate=0`；
- `stage5e` 最终 `TT_hit_rate` 约 `0.038`，优于 stage5d，但 `TT_success_rate` 仍为 0；
- 两者最终 `mean_episode_length` 都约 `75`，说明没有学会长时间稳定站立；
- stage5e 的 `reward_standing_stability` 很早饱和，但 WebRTC 可视化仍显示站不稳，说明该 reward 对“真稳定”和“僵住/假稳定”的区分不足；
- stage5d/stage5e 的 ball curriculum 都按 step 推进，策略能力不足时仍会自动加难，导致 hit rate 断崖式下降。

本阶段新增 A3-only stage：

```text
a3_tt_stage5f
a3_tt_stage5f_eval
```

核心改动：

1. 解除早期任务奖励的稳定门控
   - `reward_future_touch_point`
   - `reward_future_dis_ee`
   - `reward_contact`
   - `reward_future_landing_x_progress`

   这些项用于探索“靠近球路/触球/初步击球”，不再用 stability gate 压弱。

2. 只 gate 高阶任务奖励
   - `reward_hit_ball_velocity_net`
   - `reward_hit_net_clearance_progress`
   - `reward_future_pass_net`
   - `reward_post_hit_net_progress`
   - `reward_table_success`
   - `reward_actual_opponent_table_target`

   高阶项使用较高 `gate_floor=0.35/0.25`，避免早期完全失去梯度，但不稳定状态下不能拿满高质量回球奖励。

3. 改写站稳奖励参数
   - 降低或移除 `low_velocity_score` 对 dense standing 的影响；
   - 避免鼓励“低速度僵住”；
   - 更强调 base 高度、身体竖直、双脚支撑、干净接触、脚距；
   - `penalty_unstable_hit` 从强惩罚改为弱约束，防止“刚碰到球就被过度惩罚”。

stage5f 初始稳定权重：

```text
height_weight = 0.32
upright_weight = 0.30
support_weight = 0.23
velocity_weight = 0.00
clean_weight = 0.10
feet_width_weight = 0.05
```

4. 能力触发 ball curriculum

不再仅按 `global_step` 推进球路。新增 curriculum 会维护一个窗口内的在线指标：

```text
serve_count
hit_rate
success_rate
mean_episode_length
reset_rate
stage
```

升阶段条件示例：

```text
window_serves >= min_window_serves
mean_episode_length >= threshold
hit_rate >= threshold
reset_rate <= threshold
```

如果升阶段后 hit/stability 明显掉下去，则回退一档。

5. 小幅放大 A3 下肢 action scale

当前下肢 action scale 偏保守，容易叠加稳定/能耗/动作平滑惩罚后变成“少动僵住”。stage5f 只在新 stage 中小幅放大下肢 action scale，让策略能调整重心和身体位置。

边界：

- 不修改 T1；
- 不修改 stage5d/stage5e 作为失败对照；
- 不直接改变全局 reset 逻辑；
- 训练后必须优先通过 WebRTC 检查是否仍僵住/摔倒，而不是只看 hit rate。

实现确认（2026-06-30 17:41）：

- 已新增 `modify_ball_ranges_by_ability()`；
- 已新增 `a3_tt_stage5f` / `a3_tt_stage5f_eval`；
- 已将 stage5f 的早期探索项改回 ungated reward；
- 已将高阶过网/落台项保留 soft gate；
- 已将 stage5f dense standing reward 的 `velocity_weight` 调为 `0.00`；
- 已小幅放大 stage5f 下肢 action scale；
- 语法检查和最小 smoke test 通过。

验证命令：

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

生成目录：

```text
logs/a3_table_tennis/2026-06-30_17-41-57_a3_stage5f_abilitycurr_smoke_64_1
```

## 2026-06-29 A3 stage5c 长训前优化计划

基于 T1 曲线和 A3 stage5b 诊断，长训前做小范围优化：

- 保留 `stage5c` 作为训练入口，不新增新的主任务名；
- 不修改 T1、不修改 `stage5b`、不修改全局 reward 函数；
- 主要修正 stage5c 的击球质量目标：
  - 提高 `reward_hit_ball_velocity_net` 中向上速度和过网高度占比；
  - 提高 `reward_hit_net_clearance_progress` 的目标高度；
  - 提高 `reward_post_hit_net_progress` 的 `z` 和 `vz` 权重；
  - `reward_future_pass_net` 的目标高度从 table+0.25 调到 table+0.30；
  - 后期 `reward_table_success` 和对面落点 reward 稍高于 stage5b；
  - own-table 惩罚后期稍恢复，避免一直奖励低质量回球；
- 保持早期稳定性托底不变：
  - `termination_penalty`
  - `undesired_contacts`
  - `penalty_hit_low_base_reset`
  - 脚姿态和脚过近惩罚。

预期：

- 前 500 iter 不要求 success 明显出现；
- 1000-2500 iter 应看到 contact 和 post-hit trajectory 打开；
- 3000-6000 iter 是重点观察区间；
- 不只评估最终 checkpoint，需要重点评估 2500/3000/4000/5000/6000。
