# A3 与 T1 在 PACE 乒乓任务中的差异分析（2026-06-23）

## 分析目标

本文件记录将当前 T1 乒乓任务迁移到 A3T2.5 时，A3 与 T1 在模型、控制、观测、奖励、击球建模和训练流程上的差异，并判断哪些差异需要单独优化。

结论先行：A3 不能被视为“换一个 USD 路径”的简单替换。更稳妥的路线是新增 A3 专用 task 和 asset cfg，复用乒乓球环境、球路预测、训练 runner 和大部分任务流程，但对机器人相关配置、奖励正则、击球点、初始姿态、PD/动作尺度做 A3 专门适配。

## 分析依据

### 已阅读的项目文件

- `legged_lab/envs/__init__.py`
- `legged_lab/envs/t1_tt/t1_tt_config.py`
- `legged_lab/envs/base/tt_env.py`
- `legged_lab/envs/base/tt_config.py`
- `legged_lab/assets/booster/booster.py`
- `legged_lab/assets/table_tennis/ball.py`
- `legged_lab/assets/table_tennis/table.py`
- `legged_lab/mdp/rewards.py`
- `legged_lab/scripts/train.py`
- `legged_lab/scripts/eval.py`

### 已阅读的 A3 资料

- `/mnt/ssd/zxl/a3_t2d5 .zip`
- `a3_t2d5/README.md`
- `a3_t2d5/urdf/model.urdf`
- `a3_t2d5/meshes/*.STL`

### A3 URDF 解析结果

- robot name：`A3T2.5`
- links：39
- joints：38
- revolute joints：31
- fixed joints：7
- 解析总质量：约 `57.904204 kg`

A3 README 说明：腰部 `waist_pitch_joint`、`waist_roll_joint` 以及脚踝 roll 相关结构实际为并联结构，但 URDF 已做串联等效，训练时可按串联处理，部署时再做并联解算。

## 总体差异概览

| 维度 | 当前 T1 | A3T2.5 | 影响 |
| --- | --- | --- | --- |
| 资产格式 | 已有 USD | 只有 URDF + STL | 需要 URDF 导入或生成 USD |
| 动作关节数 | 21 | 31 revolute joints | 策略输入/输出维度改变，不能复用 T1 checkpoint |
| 关节命名 | `Left_Hip_Pitch` 等大驼峰 | `left_hip_pitch_joint` 等小写 snake case | reward、reset、action/obs 关节列表都要映射 |
| root/body 命名 | `Trunk`、`.*_foot_link` | `pelvis_link`、`torso_Link`、`*_ankle_roll_Link` | 接触、终止、feet cfg、base mass randomization 要改 |
| 击球 body | 当前硬编码 body index 15 | 候选 `right_hand_Link` | 必须参数化 paddle body 和 offset |
| 机器人尺寸/质量 | T1 配置固定 | A3 约 57.9 kg，尺寸/臂长不同 | 初始高度、PD、动作尺度、future target 高度需调 |
| 球拍模型 | T1_TT USD 可能已包含球拍/末端碰撞 | A3 URDF 原始包未见球拍 | 可能需要附加 paddle collision/visual |
| 奖励正则 | T1 body/joint regex | A3 body/joint regex | 需要 A3 专用 reward cfg |
| 训练 checkpoint | 已有 T1 checkpoint | 无 A3 checkpoint | A3 需要重新训练 |

## 资产层差异

### T1

T1 当前通过 `BOOSTER_T1_TT_P2_CFG` 引用：

```text
legged_lab/assets/booster/T1_TT/T1_TT.usd
```

这意味着 T1 的物理属性、body hierarchy、碰撞体、可能的球拍末端等已经被封装在 USD 中。

### A3

A3 只有：

```text
a3_t2d5/urdf/model.urdf
a3_t2d5/meshes/*.STL
```

没有现成 USD。迁移时需要选择：

1. 使用 `sim_utils.UrdfFileCfg` 在运行时导入 URDF；
2. 或先用 IsaacLab `scripts/tools/convert_urdf.py` 生成 USD，再在 `ArticulationCfg` 中引用 USD。

### 是否需要单独优化

需要。

A3 的 collision 质量会明显影响训练稳定性和球/地面接触表现。初版可以使用 URDF 导入默认凸包碰撞，但中后期建议检查：

- fixed joints 是否被合并；
- 手部和脚部碰撞是否合理；
- STL mesh 是否过细导致导入慢或碰撞异常；
- 是否需要为右手额外添加球拍 collision/visual。

## 运动学和自由度差异

### T1

当前 T1 任务使用 21 个动作关节：

- 双臂：8
- 腰部：1
- 双腿：12

关节命名例如：

```text
Left_Shoulder_Pitch
Right_Elbow_Yaw
Waist
Left_Hip_Pitch
Left_Ankle_Roll
```

### A3

A3 URDF 有 31 个 revolute joints：

- 腰部：3，`waist_yaw_joint`、`waist_roll_joint`、`waist_pitch_joint`
- 头部：2
- 双臂：14，含肩、肘、腕
- 双腿：12

### 影响

如果 A3 使用全部 31 个 revolute joints：

- action dim：从 21 变为 31；
- 单帧 actor obs 长度从约 `3 * 21 + 18 = 81` 变为约 `3 * 31 + 18 = 111`；
- actor/critic 网络第一层输入维度和 actor 输出维度都改变；
- T1 checkpoint 无法直接加载到 A3 policy。

### 是否需要单独优化

需要。

建议分两阶段：

1. 首版全关节接入，用于验证流程完整性和 A3 动作链路；
2. 如果训练不稳，再做受控关节子集优化，例如固定头部、弱化或固定部分左手腕，仅保留右臂击球和双腿移动所需自由度。

## 命名和配置映射差异

### T1 当前依赖

`t1_tt_config.py` 中大量使用 T1 命名：

- body：`Trunk`
- feet：`.*_foot_link`
- ankle：`.*_Ankle_Pitch`、`.*_Ankle_Roll`
- hip：`.*_Hip_Yaw`、`.*_Hip_Roll`
- arm：`Left_Shoulder_.*`、`Right_Elbow_.*`
- waist：`Waist`

### A3 对应关系

建议初版映射：

| 语义 | T1 | A3 |
| --- | --- | --- |
| root/base | `Trunk` | `pelvis_link` 或 `torso_Link` |
| height scanner body | `Trunk` | `torso_Link` |
| 终止接触 body | `Trunk` | `torso_Link`、`pelvis_link` |
| feet body | `.*_foot_link` | `left_ankle_roll_Link`、`right_ankle_roll_Link` |
| paddle anchor | body index 15 | `right_hand_Link` |
| waist joint | `Waist` | `waist_.*_joint` |
| hip joints | `.*_Hip_.*` | `.*hip_.*_joint` |
| knee joints | `.*_Knee_.*` | `.*knee_joint` |
| ankle joints | `.*_Ankle_.*` | `.*ankle_.*_joint` |
| arm joints | `Right_Shoulder_.*` | `right_shoulder_.*_joint` |
| elbow joints | `Right_Elbow_.*` | `right_elbow_joint` |

### 是否需要单独优化

必须做专门适配，但这更偏“正确性修改”，不是训练调参。

如果不做命名映射，A3 task 在 reward manager、event manager、joint/body resolve 阶段就会失败。

## 击球建模差异

### 当前 T1 逻辑

当前 `TTEnv.compute_paddle_touch()` 不是完全依赖真实球拍碰撞来判断击球，而是：

1. 固定取 `paddle_index = 15`；
2. 从该 body 的世界位置和四元数出发；
3. 加局部偏移 `[0.0, -0.345, 0.0]`；
4. 得到 `paddle_touch_point`；
5. 用球心到该点的距离计算 `ball_contact` 和击球奖励。

这说明击球点是任务层的虚拟末端点。真实碰撞仍可能影响球的物理运动，但奖励触发依赖这个虚拟点。

### A3 问题

A3 的 body index 不同，不能使用 `15`。A3 原始资料中也未见球拍 mesh，因此：

- 必须把 paddle body 改为配置项；
- A3 应配置 `right_hand_Link` 作为候选 anchor；
- paddle local offset 需要重新标定；
- 如球无法产生合理反弹，需要给右手添加 paddle collision/visual。

### 是否需要单独优化

需要，而且是 A3 迁移的核心优化项之一。

推荐顺序：

1. 首版用 `right_hand_Link + local_offset` 复现虚拟击球奖励；
2. 可视化检查 `paddle_touch_point` 是否落在 A3 右手实际可击球区域；
3. 训练前期先保证 reward 能被触发；
4. 后续添加真实 paddle collision，使球的物理反弹和虚拟击球点一致。

## 初始姿态和站位差异

### T1

T1 当前初始 base：

```python
pos=(-1.6, 0.0, 0.72)
```

腿部默认姿态类似：

```text
hip pitch = -0.20
knee pitch = 0.42
ankle pitch = -0.23
```

右臂已有 T1_TT 专用击球姿态。

### A3

A3 质量和尺寸不同，root 是 `pelvis_link`，腿长和手臂链路也不同。不能直接照搬 T1 的 base height 和右臂角度。

### 是否需要单独优化

需要。

初版需要调：

- base 初始高度，防止脚穿地或悬空；
- 双腿站立角；
- 右臂预备击球姿态；
- 左臂平衡姿态；
- 站位 reset range，避免 A3 距球桌太近或太远。

建议先用 1 个 env 可视化/打印检查站立姿态，再开始大规模训练。

## 动作尺度、PD 和 actuator 差异

### T1

当前 T1_TT_P2 使用 `DelayedPDActuatorCfg`，不同组的 stiffness/damping/effort 都是 T1 专门设定。

### A3

A3 URDF limit 中 effort/velocity 明显不同，例如：

- hip joints effort 约 `220`
- knee joints effort 约 `320`
- ankle pitch effort 约 `118.2`
- ankle roll effort 约 `54.75`
- shoulder effort 约 `60`
- elbow/wrist部分更低

### 是否需要单独优化

需要。

首版建议用 URDF effort/velocity 做上限，PD 参数保守设置。后续需要重点调：

- `action_scale`
- stiffness/damping
- effort limit scale
- delayed actuator delay
- ankle/waist 的稳定性参数

A3 的腰部为串联等效的并联结构，训练可按串联，但部署时动作映射必须另行处理。

## 奖励和正则项差异

### 可复用部分

乒乓球任务本身可复用：

- 发球速度范围；
- 球路预测；
- 击球奖励思想；
- 过网/落点/成功判定；
- predictor runner；
- table/ball asset。

### 需要 A3 专用的部分

与机器人形态相关的 reward 必须改：

- feet contact / fly / feet slide / feet force；
- feet orientation；
- feet too near；
- undesired contacts；
- joint deviation；
- energy ankle；
- paddle-head-too-near；
- torso/waist regularization。

### 是否需要单独优化

需要。

初版可以按 T1 reward 权重拷贝一份 A3 版本并替换命名，但训练稳定性大概率需要重新调 reward 权重。尤其是：

- A3 更重，energy 权重可能需要重新缩放；
- 足部接触 body 不同，feet force threshold 可能需要重新标定；
- A3 多腰/腕自由度，需要更多姿态正则，避免策略用不自然动作刷击球奖励；
- 右手腕自由度增加后，击球末端稳定性可能更难，需要增加右臂/腕部平滑和偏差惩罚。

## 观测、策略和 checkpoint 差异

### T1

已有可用 checkpoint：

```text
logs/t1_table_tennis/2026-06-22_21-09-35/model_9999.pt
```

### A3

A3 如果动作维度为 31，则：

- actor 输出维度不同；
- actor obs 维度不同；
- critic obs 维度也不同；
- checkpoint 网络权重形状不匹配。

### 是否需要单独优化

需要重新训练。

不建议把“加载 T1 checkpoint”作为 A3 初版目标。可选的高级路线是后续做迁移学习，例如只迁移部分 MLP 层或做 T1->A3 行为蒸馏，但这不是最小可行接入的必要步骤。

## predictor 是否需要修改

当前 predictor 学的是球的未来位置，输入主要来自球位置历史。球桌、球和球路物理没有因为机器人从 T1 换成 A3 而改变。

### 是否需要单独优化

首版不需要改 predictor 结构。

需要注意的是：

- A3 策略观测维度改变，不影响 predictor 网络本身；
- A3 训练时 predictor 权重仍需在 A3 新 run 中保存；
- 如果发球范围或击球策略分布改变很大，可后续重新调 predictor 训练轮数和数据窗口。

## 训练流程差异

### 可复用流程

以下流程可复用：

- `legged_lab.scripts.train`
- `legged_lab.scripts.eval`
- `OnPolicyPredictorRegressionRunner`
- `TTEnv` 主循环；
- ball/table/aerodynamics/predictor 逻辑。

### 需要独立流程参数

A3 建议新 experiment name：

```text
a3_table_tennis
```

并单独保存 checkpoint。不要和 T1 的 `t1_table_tennis` 混用。

建议先跑 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=1 \
  --headless \
  --predictor
```

通过后再增加 `num_envs`。

## 必须做的 A3 适配项

1. 新增 A3 asset cfg。
2. 新增 A3 task：`a3_tt` / `a3_tt_eval`。
3. 参数化 `TTEnv` 中的 paddle body/index/offset，默认保持 T1 当前行为。
4. 参数化 `future_body_height` 和 `future_paddle_y_offset`，默认保持 T1 当前行为。
5. 新增 A3 action/observation joint list。
6. 新增 A3 reward cfg，替换 T1 body/joint regex。
7. 设置 A3 feet body、terminate body、base mass randomization body。
8. 设置 A3 初始姿态和 actuator。
9. 使用 A3 独立 experiment name，重新训练。

## 球拍安装和击球点标定

T1 不是在训练代码中动态安装球拍，而是使用乒乓专用资产：

```text
legged_lab/assets/booster/T1_TT/T1_TT.usd
```

T1 task 选择的是 `BOOSTER_T1_TT_P2_CFG`，该配置加载 `T1_TT.usd`。USD 中 `right_hand_link` 下存在一个用于标定击球中心的 marker：

```text
/T1_TT/right_hand_link/marker_ball
local translate = (0, -0.35, 0)
```

基础 `RobotCfg` 中的默认 T1 击球点配置为：

```text
paddle_body_index = 15
paddle_local_offset = (0.0, -0.345, 0.0)
```

这与 T1 USD 中的 `marker_ball` 基本一致。`TTEnv.compute_paddle_touch()` 用机器人某个 body 的世界位姿加上 `paddle_local_offset` 得到 `paddle_touch_point`，再以该点和球之间的距离计算虚拟接触奖励。

因此，球拍迁移至少包含两层含义：

1. 资产层：球拍视觉/碰撞/marker 是否已经固定安装到机器人手部。
2. 任务层：`paddle_body_name` 或 `paddle_body_index` 以及 `paddle_local_offset` 是否准确指向球拍有效击球中心。

当前 A3 只有机器人 URDF/STL，没有球拍模型、球拍碰撞、球拍安装位姿或 marker。A3 首版临时使用：

```text
paddle_body_name = "right_hand_Link"
paddle_local_offset = (0.12, 0.0, 0.0)
```

该值只是迁移初值，并非由 A3 球拍安装文档标定得到。后续如果要稳定训练，应补充以下任一类信息：

- 已经安装球拍的完整 A3 URDF/USD；
- 一个球拍 URDF/USD，并包含 parent 为 `right_hand_Link` 的 fixed joint；
- 从 `right_hand_Link` 到球拍中心或有效击球中心的局部坐标 `xyz/rpy`；
- 球拍尺寸、质量、碰撞形状和安装方向。

如果只给单独球拍 URDF，但没有相对 A3 右手的 fixed joint 或安装位姿，无法唯一确定球拍应安装在 A3 手上的哪个位置和朝向，只能猜测，训练效果会不稳定。

### 原 T1 的具体机制

原 T1 table tennis task 使用的机器人配置链路如下：

```text
t1_tt_config.py -> BOOSTER_T1_TT_P2_CFG -> booster/T1_TT/T1_TT.usd
```

也就是说，原任务不是加载普通 `booster/t1/t1.usd`，而是加载乒乓专用的 `T1_TT.usd`。通过 USD API 检查可知：

```text
普通 T1:
  /T1/right_hand_link
  /T1/right_hand_link/visuals
  /T1/right_hand_link/collisions

乒乓 T1:
  /T1_TT/right_hand_link
  /T1_TT/right_hand_link/marker_ball
  /T1_TT/right_hand_link/visuals
  /T1_TT/right_hand_link/collisions
```

其中 `marker_ball` 的局部位姿为：

```text
parent = /T1_TT/right_hand_link
local translate = (0, -0.35, 0)
local orient = identity
```

代码默认参数：

```text
paddle_body_index = 15
paddle_local_offset = (0.0, -0.345, 0.0)
```

因此，T1 的击球中心不是训练时通过物理接触自动发现的，而是一个已经在 T1 右手坐标系中标定好的虚拟击球点。`TTEnv.compute_paddle_touch()` 每步计算：

```text
paddle_touch_point = right_hand_link_world_pos + rotate(right_hand_link_world_quat, paddle_local_offset)
```

然后用球心到 `paddle_touch_point` 的距离计算虚拟接触奖励。当前 `contact_threshold` 是 0.05m 量级，因此该 offset 需要厘米级准确；偏差过大会导致 `reward_contact` 几乎无法触发。

### 对 A3 的含义

A3 当前资产只有机器人本体，没有 T1 那样的 `marker_ball`，也没有带球拍的专用 A3 USD/URDF。因此 A3 首版配置：

```text
paddle_body_name = "right_hand_Link"
paddle_local_offset = (0.12, 0.0, 0.0)
```

只是一个迁移初值。它没有来自 A3 手部坐标系或球拍安装结构的依据，所以 A3 训练中 `reward_contact` 接近 0 是合理风险。

如果要让 A3 接近 T1 的做法，应为 A3 建立同样的标定结构：

1. 资产中加入 `right_hand_Link` 下的 `paddle_center` 或 `marker_ball`。
2. 或者加入固定到 `right_hand_Link` 的 `paddle_link`，并把 `paddle_body_name` 指向该 link。
3. 用可视化或 CAD/实测确定 `right_hand_Link -> paddle_center` 的 `xyz/rpy`。
4. 首轮训练仍可沿用虚拟接触点；真实球拍碰撞可作为第二阶段增强。

### A3 当前训练未学到乒乓任务的主要原因判断

从 A3 baseline 1000 iter 指标看：

```text
Train/TT_hit_rate ~= 0.001
Train/TT_success_rate = 0
Episode_Reward/reward_contact ~= 0
Episode_Reward/reward_table_success = 0
```

结合 T1 的实现方式，当前最可能的主要根因是 A3 没有完成球拍安装位姿和击球中心标定。T1 的击球中心已经通过 `right_hand_link/marker_ball` 和 `paddle_local_offset=(0,-0.345,0)` 对齐，而 A3 目前只是临时使用 `right_hand_Link + (0.12,0,0)`。在 `contact_threshold=0.05m` 的虚拟接触判定下，击球中心偏差几厘米就会明显影响 reward，偏差十厘米级基本会导致策略很难获得有效击球奖励。

该问题应视为 A3 table tennis 迁移的第一优先级。站位、reset range、初始右臂姿态、PD/action_scale 也会影响学习，但它们应在击球中心可视化正确之后再系统调参；否则继续长训可能只是在错误几何上探索。

建议解决顺序：

1. 复用 T1 或简化球拍几何，先做 A3 右手上的可视化球拍/marker。
2. 标定 `right_hand_Link -> paddle_center` 的局部 `xyz/rpy`。
3. 在 A3 配置中更新 `paddle_local_offset`；如果使用独立 `paddle_link`，则改为 `paddle_body_name="paddle_link"`，offset 可设为该 link 到击球中心的局部偏移。
4. 使用非 headless 小环境检查球、手、球拍中心、预测未来击球点是否在同一可达区域。
5. 再跑 500-1000 iter 对比 `reward_contact` 和 `TT_hit_rate`。若仍低，再调 A3 reset/右臂初始姿态/action_scale。

### 新增 A3 乒乓拍 URDF 资料分析

用户补充：

```text
/mnt/ssd/zxl/095dabe4-ebcc-4666-80d9-ab49f41fbaa8.zip
```

该包为 A3T2.5 乒乓专用 URDF，包含球拍相关 link/mesh，不再只是机器人裸手模型。关键文件：

```text
0000014503_A3T2.5-URDF-std-pingpang-0519/urdf/URDF-JOINT-LINK.urdf
0000014503_A3T2.5-URDF-std-pingpang-0519/meshes/right_hand_pingpang_Link.STL
0000014503_A3T2.5-URDF-std-pingpang-0519/meshes/pingpang_red_Link.STL
0000014503_A3T2.5-URDF-std-pingpang-0519/meshes/pingpang_black_Link.STL
0000014503_A3T2.5-URDF-std-pingpang-0519/meshes/pingbang_ball_Link.STL
```

解析结果：

```text
links: 43
joints: 42
revolute joints: 31
fixed joints: 11
total mass: 58.27723163 kg
```

revolute joint 顺序与此前裸 A3 的 31 个动作关节一致，因此策略动作维度仍为 31。新增/替换的右手乒乓链路为：

```text
right_wrist_yaw_Link
  fixed right_hand_pingpang_joint, xyz=(0,0,0), rpy=(0,0,0)
    -> right_hand_pingpang_Link
       fixed pingpang_red_joint, xyz=(0.21021, 0.032078, 0.032036), rpy=(0,0,0)
         -> pingpang_red_Link
       fixed pingpang_black_joint, xyz=(0.21021, 0.032078, 0.032036), rpy=(0,0,0)
         -> pingpang_black_Link
       fixed pingbang_ball_joint, xyz=(0.210211399202899, 0.0320784994676765, 0.0320358706296689), rpy=(0,0,0)
         -> pingbang_ball_Link
```

`pingpang_red_Link`/`pingpang_black_Link` 是拍面视觉/碰撞，`pingbang_ball_Link` 是供应方给出的击球点 marker。其 fixed joint origin 可直接作为 A3 版虚拟击球点标定：

```text
paddle_body_name = "right_hand_pingpang_Link"
paddle_local_offset = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
```

也可以在确认 IsaacLab 保留零质量 fixed link 后使用：

```text
paddle_body_name = "pingbang_ball_Link"
paddle_local_offset = (0.0, 0.0, 0.0)
```

更稳妥的首版做法是使用 `right_hand_pingpang_Link + offset`，因为 `pingbang_ball_Link` 的质量为 0，导入/转换时是否稳定保留为 articulation body 需要额外验证。

该新 URDF 基本解决了此前 A3 训练 `reward_contact` 接近 0 的主要疑点：原先 A3 使用裸手 `right_hand_Link + (0.12,0,0)` 作为猜测击球点，现在可以切换为供应方标定的乒乓拍链路和 marker。

### 2026-06-24 A3 乒乓 URDF 训练结果复盘

用户使用 A3 乒乓专用 URDF 训练：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=1000 \
  --run_name=a3_pingpang_baseline_1000 \
  --headless \
  --predictor
```

TensorBoard/event 文件结果：

```text
Train/mean_reward: last -31.673779, tail100 -31.612850
Train/mean_episode_length: last 60.290001, tail100 60.033900
Train/TT_hit_rate: last 0.000944584, max 0.009558824
Train/TT_success_rate: 0
Episode_Reward/reward_contact: last 0, tail100 0.000039286
Episode_Reward/reward_table_success: 0
Episode_Reward/termination_penalty: tail100 -2
Episode_Reward/reward_future_dis_ee: tail100 0.022451
Loss/predictor_mse: last 0.062777
```

与旧的裸 A3 结果相比，乒乓 URDF 并没有显著提升 `reward_contact` 或 `TT_hit_rate`。说明“是否有球拍/击球点 marker”已经不是唯一瓶颈。

运行时几何检查显示，当前配置中的击球点标定是生效的：

```text
paddle_body_name = right_hand_pingpang_Link
paddle_local_offset = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
pingbang_ball_Link world position == computed paddle_touch_point
```

reset 后的初始几何关系：

```text
ball_pos_w              [ 1.35, -0.02466, 1.03]
ball_linvel_w           [-5.127443, -0.446022, 1.610857]
ball_future_pose_env    [-1.6, -0.295825, 1.077451]
robot_future_pos_env    [-1.7,  0.304175, 0.9]
paddle_touch_point_w    [-1.543757, 0.259369, 0.603646]
initial paddle-ball distance ~= 2.918755
```

零动作 rollout 约 120 step 的最近距离仍约为：

```text
min paddle-ball distance ~= 0.298689 m
```

而当前虚拟接触奖励的有效半径是厘米级，`contact_threshold=0.05m`，并且还会扣除球半径。因此 A3 的默认姿态下，球拍中心虽然已标定正确，但轨迹没有进入有效接触奖励区。乒乓球任务奖励对击球点距离非常敏感；如果策略早期拿不到 contact reward，后续 table success reward 也不会出现。

当前结论：

1. A3 乒乓 URDF 的 marker/offset 配置是正确生效的。
2. 训练未学习到乒乓任务的根因更可能是 A3 初始站姿、右臂初始姿态、球路/reset 分布和 T1 奖励课程不匹配。
3. T1 奖励不能直接视为 A3 的最终奖励；A3 需要 A3 专用的 staged reward/curriculum。
4. `num_envs=64` 只是小规模实验，会放大稀疏奖励问题；但当前零动作几何和 TensorBoard 指标显示，瓶颈不是单纯增加训练步数即可解决。

建议下一步优化顺序：

1. 先做 A3 击球姿态标定：调整 A3 `init_state.joint_pos` 的右臂/手腕默认角度，让 reset 后的 `paddle_touch_point` 靠近 `ball_future_pose_env`，目标是把初始或自然摆臂下的最近距离从约 0.30m 降到 0.10m 以内。
2. 收窄 A3 初期 ball reset 分布：固定或缩小球的 y、高度、速度范围，先训练单一可达球路，等能稳定接触后再恢复随机化。
3. 增加 A3-only dense shaping：提高击球前 `paddle_touch_point -> ball_future_pose` 的距离奖励权重，或启用/新增更强的 `paddle_ball_distance`/`future_ee_target` 奖励；早期可以放宽 contact threshold，后续 curriculum 再收紧。
4. 降低早期右臂相关正则约束：A3 的右臂/腕部 DoF 和 T1 不同，T1 的 joint deviation、action rate、energy 等权重可能让策略不愿大幅移动右臂。
5. 增加 debug logging：记录每个 episode 的最小 `paddle-ball distance`、最小 `paddle-future distance`、termination reason，避免只从 reward_contact=0 反推原因。
6. 稳定性单独处理：`termination_penalty` 后期仍饱和为 -2，说明 episode 经常以失败终止；需要同时检查 base reset、站姿高度、越界条件和 A3 PD/action scale。

因此，奖励函数确实需要 A3 专门设计或至少 A3 专用调权，但不建议第一步大改公共 `TTEnv`。应优先在 `a3_tt_config.py` 中做 A3-only 的姿态、reset、reward weight/curriculum 配置；必要时再给 `TTEnv` 增加可配置 debug 指标和 A3-only dense reward 开关，保证 T1 workflow 不变。

### T1 训练初期为什么更容易学到

T1 不是从普通站姿、任意右手位置直接硬搜乒乓任务。代码中 T1 任务使用的是：

```text
legged_lab/assets/booster/T1_TT/T1_TT.usd
BOOSTER_T1_TT_P2_CFG
```

`BOOSTER_T1_TT_P2_CFG` 已经是一个专门的乒乓姿态配置，右臂默认值为：

```text
Right_Shoulder_Pitch = 0.0
Right_Shoulder_Roll  = -0.1
Right_Elbow_Pitch    = 0.2
Right_Elbow_Yaw      = 0.5
```

同一文件里还存在 `BOOSTER_T1_TT_CFG`、`BOOSTER_T1_TT_P_CFG`、`BOOSTER_T1_TT_P2_CFG` 三个版本，说明原 T1 工作本身已经经历过球拍/右臂预姿态的迭代，不是纯粹依赖随机探索。

T1 正式训练 run：

```text
logs/t1_table_tennis/2026-06-22_21-09-35
```

早期指标：

```text
Train/TT_hit_rate:
  step 0   = 0.000000
  step 99  = 0.024454
  step 499 = 0.049025
  step 999 = 0.055047

Train/TT_success_rate:
  step 0   = 0.000000
  step 99  = 0.000224
  step 499 = 0.001461
  step 999 = 0.001522

Episode_Reward/reward_contact:
  step 0    = 0.000000
  step 99   = 0.003349
  step 249  = 0.031232
  step 999  = 0.035231
  step 1999 = 0.249502
```

这说明 T1 早期也不是一开始就会打球，但它很快能产生少量接触样本，后续通过 `reward_contact -> reward_future_landing_dis/reward_future_pass_net -> reward_table_success` 逐步放大成功行为。A3 当前在 1000 iter 后仍接近 `TT_hit_rate ~= 0.001`、`success = 0`，说明 A3 尚未进入 T1 当时的“少量有效接触样本”阶段。

因此 A3 需要先复刻 T1 已经隐含完成的前置工作：

1. 建立 A3 专用右臂/球拍预姿态，而不是只替换 URDF。
2. 让 `paddle_touch_point` 在 reset 后接近常见球路的预测击球点。
3. 初期用更窄球路和更强 dense shaping 产生第一批 contact 样本。
4. 待 `TT_hit_rate` 达到数个百分点后，再逐步恢复 T1 的随机球路和原始奖励权重。

### T1 站立和 reset 是如何处理的

T1 任务使用的不是普通机器人裸模型，而是已经准备好的乒乓专用 USD：

```text
legged_lab/assets/booster/T1_TT/T1_TT.usd
BOOSTER_T1_TT_P2_CFG
```

关键配置：

```text
root init pos = (-1.6, 0.0, 0.72)

leg default pose:
  .*_Hip_Pitch   = -0.20
  .*_Hip_Roll    =  0.0
  .*_Hip_Yaw     =  0.0
  .*_Knee_Pitch  =  0.42
  .*_Ankle_Pitch = -0.23
  .*_Ankle_Roll  =  0.0

right arm table-tennis pre-pose:
  Right_Shoulder_Pitch = 0.0
  Right_Shoulder_Roll  = -0.1
  Right_Elbow_Pitch    = 0.2
  Right_Elbow_Yaw      = 0.5
```

T1 的 actuator 配置是手工调过的 delayed PD：

```text
legs:
  hip/waist effort ~= 30-45
  knee effort      ~= 60
  stiffness        = 100
  damping          = 3

feet:
  ankle pitch effort = 20
  ankle roll effort  = 15
  stiffness          = 30
  damping            = 3

arms:
  effort     = 18
  stiffness  = 30
  damping    = 3
```

T1 reset 也会扰动默认姿态：

```text
locomotion joints: default_joint_pos * uniform(0.5, 1.5)
right arm joints:  default_joint_pos + uniform(-0.5, 0.5)
base pose:         default root pose + sampled x/y/yaw offset
```

但 T1 的默认站姿、脚底碰撞、质量惯量、PD 和球拍预姿态已经彼此匹配，所以加扰动后仍能比较稳定地进入任务。

A3 当前的风险是：A3 的腿部默认值直接沿用了 T1 的数值模式：

```text
hip pitch   = -0.20
knee        =  0.42
ankle pitch = -0.23
```

但 A3 的身体比例、关节轴方向、脚底碰撞、质量分布和 base 高度都不同；这些相同数字不保证对应 A3 的稳定站姿。因此如果 A3 在 Isaac Sim 中 reset 后直接倒下，优先应检查 A3 站立配置，而不是先调右臂击球姿态。

处理顺序应改为：

1. 先用 `--pin_base` 标定右臂/球拍几何，避免倒下影响观察。
2. 单独建立 A3 standing calibration：固定 base/无球或静止球，找到 A3 自由站立稳定的 root height、腿部默认关节、脚底接触和 PD。
3. 再把稳定站姿与右臂乒乓预姿态合并。
4. 最后进入短训验证。

### A3 右臂+球拍预姿态的可视化标定流程

目标不是让 A3 reset 后自动击球，而是先找到一个合理的“准备击球姿态”，使 reset 后的 `paddle_touch_point` 接近常见 `ball_future_pose`。建议首轮目标：

```text
norm(paddle_touch_point - ball_future_pose) < 0.10m ~ 0.15m
```

当前 A3 诊断约为 `0.30m` 量级，因此需要先调右臂/手腕默认姿态。

使用 Isaac Sim WebRTC Streaming Client 时，IsaacLab 启动参数应使用：

```bash
--livestream 2
```

在当前 IsaacLab `AppLauncher` 中，`--livestream 2` 会自动把 host 端切为 headless，但仍提供 WebRTC 画面。不要再依赖本机 GUI 窗口。

建议操作逻辑：

1. 启动一个 A3 单环境可视化标定脚本或 play/eval 脚本，使用 `--num_envs=1 --livestream 2`。
2. 在 WebRTC Streaming Client 里观察 A3、球、球桌和右手球拍姿态。
3. 同时在终端打印每次 reset 后的：

```text
ball_future_pose
paddle_touch_point
paddle_to_future_distance
right arm joint positions
```

4. 第一阶段固定球路和 base reset，避免随机性干扰姿态判断。
5. 调整右臂关节默认值：

```text
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

6. 每次只改小范围，观察球拍中心是否更接近 `ball_future_pose`，并检查球拍面是否大致朝向来球方向。
7. 选出距离小、姿态自然、站立稳定的一组关节值，写入 A3 专用 `init_state.joint_pos`。
8. 再用短训 200-500 iter 验证 `reward_contact` 和 `TT_hit_rate` 是否明显高于当前 baseline。

当前环境已经创建了 `ball_future` 和 `ball_pred` 可视化对象，但 `compute_intermediate_values()` 里 `update_ball_future_visual()` 调用暂时被注释。因此若要在 WebRTC 里直接看见未来击球点，建议后续新增一个 A3-only 标定脚本或调试开关，在不影响 T1 的前提下显示：

```text
绿色球: ball_future_pose
蓝色球: learned ball_prediction
红色/黄色球: paddle_touch_point 或 pingbang_ball_Link
```

这样人工标定会比只看终端数值更直观。

### `ball_future_pose` 和 `ball_prediction` 的区别

当前流程可以理解为：

```text
真实球在仿真里运动
  -> 环境根据球的位置/速度/反弹/重力/空气阻力计算 ball_future_pose
  -> predictor MLP 用最近 5 帧球 3D 位置学习回归 ball_future_pose
  -> actor 策略观测里使用 MLP 输出的 ball_prediction
  -> critic 和部分 reward 使用环境计算的 ball_future_pose
```

二者区别：

```text
ball_future_pose:
  环境用物理公式算出的未来击球参考点。
  是监督目标/训练真值，依赖仿真里的完整球状态。
  critic 可以看到；reward_future_ee_target 等奖励会使用它。

ball_prediction:
  MLP predictor 根据最近 5 帧球位置预测出的未来击球点。
  是 actor 实际看到的未来点估计。
  它学习模仿 ball_future_pose，但可能有误差。
```

这样设计相当于 asymmetric actor-critic：

```text
训练时 critic/reward 可以使用更强的仿真真值信号；
actor 只使用更接近真实部署条件的感知预测结果。
```

因此 A3 姿态标定时应以 `ball_future_pose` 作为几何目标，因为它是当前环境定义下的“应去击球的位置”。训练策略时，actor 实际依赖的是 `ball_prediction`，所以 predictor 误差也会影响学习，但 A3 当前主要瓶颈是球拍初始工作区离 `ball_future_pose` 太远，早期连 contact 样本都很少。

### A3 右臂预姿态调整 SOP

目标：

```text
在固定简单球路下，让 reset 后的 A3 paddle_touch_point 接近 ball_future_pose。
首轮目标距离 < 0.15m，理想目标 < 0.10m。
```

推荐采用“可视化 + 数值闭环”的方式，而不是只凭肉眼在 Isaac Sim 里拖动模型。

#### 阶段 0：准备调试环境

1. 使用 `zxl-pace` 环境。
2. 单环境运行，避免多个 env 干扰观察。
3. 使用 WebRTC：

```bash
--livestream 2
```

4. 固定或收窄球路：

```text
ball_pos_y_range      -> 先固定为 0 附近
ball_speed_x_range    -> 先固定或窄范围
ball_speed_y_range    -> 先固定为 0 附近
ball_speed_z_range    -> 先固定或窄范围
reset_base pose_range -> 先固定/窄范围
right arm reset noise -> 先关闭或缩小
```

#### 阶段 1：显示需要看的点

需要在 WebRTC 中看到：

```text
红色球: 当前真实球
绿色球: ball_future_pose
黄色球: paddle_touch_point / pingbang_ball_Link
```

同时终端打印：

```text
right arm joint values
paddle_touch_point
ball_future_pose
paddle_to_future_distance
paddle_ball_min_distance
```

当前公共环境里 `ball_future` 可视化对象已存在，但更新调用被注释；建议新增 A3-only 标定脚本/调试开关显示这些点，避免影响 T1。

#### 阶段 2：调右臂关节

优先调这些 A3 关节：

```text
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

调参逻辑：

1. 先调 shoulder，让整只右臂把球拍送到身体右前方。
2. 再调 elbow，让球拍中心接近目标高度和前后位置。
3. 最后调 wrist，让球拍面朝向来球方向，并微调击球中心。
4. 每次只改一个或两个关节，小步调整，例如 `0.05 rad` 或 `0.1 rad`。
5. 每次调整后记录 `distance = norm(paddle_touch_point - ball_future_pose)`。
6. 如果姿态穿模、贴身体、接近关节极限或站立不稳，即使距离小也不采用。

#### 阶段 3：选择候选姿态

候选姿态需要同时满足：

```text
paddle_to_future_distance < 0.15m
球拍不贴身体、不穿模
右臂姿态自然，离关节极限有余量
球拍面大致朝向来球
A3 站立稳定
```

选出 2-3 组候选后，分别做短 rollout，比较：

```text
zero-action min paddle-ball distance
random small-action min paddle-ball distance
termination 是否变多
```

#### 阶段 4：写入 A3 默认姿态

把最佳候选写入 A3 专用配置，不改 T1：

```text
legged_lab/assets/a3/a3.py
  A3_T2D5_CFG.init_state.joint_pos
```

或者更稳妥地新增 A3 pingpong 专用 cfg，在 `A3_T2D5_PINGPANG_CFG` 上覆盖右臂默认姿态，避免影响裸 A3。

#### 阶段 5：短训验证

先不要直接长训。建议：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --max_iterations=500 \
  --run_name=a3_pose_candidate_001 \
  --headless \
  --predictor
```

看指标：

```text
Episode_Reward/reward_contact 是否明显非零
Train/TT_hit_rate 是否从 ~0.001 提升到至少 0.02 量级
Episode_Reward/termination_penalty 是否没有明显恶化
reward_future_dis_ee 是否提高
```

若 500 iter 后 contact 仍接近 0，则继续调姿态或加强 A3-only dense reward；若 hit rate 有明显提升，再进入 curriculum/reward 权重调整。

## 建议做的 A3 优化项

## 2026-06-25 A3 站立姿态自动筛选结果

新增 A3-only 站立姿态筛选脚本后，按用户参考图生成了“下蹲、两脚分开、右臂持拍在身前”的候选姿态，并补充测试了更接近 URDF 零位的直腿/轻微弯腿姿态。

测试命令示例：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.a3_standing_calibration \
  --task=a3_tt_eval \
  --max_candidates=128 \
  --trial_steps=250 \
  --print_interval=50 \
  --headless
```

关键结果：

```text
第一轮 ready-arm 下蹲候选:
  96 candidates, 250 steps
  0/96 survived 250 steps
  best: mild_z0.90_w+0.00_toe+0.05, survived 114/250 steps

第二轮加入 zero/soft/light leg:
  128 candidates, 250 steps
  0/128 survived 250 steps
  best: zero_leg_z0.96, survived 122/250 steps

第三轮解耦手臂，加入 default/zero/ready arms:
  128 candidates, 250 steps
  0/128 survived 250 steps
  best score: zero_leg_default_arms_z0.96, survived 92/250 steps under strict roll/pitch threshold

单独复测 candidate_index=37:
  threshold 放宽到 max_abs_roll_pitch=0.80, max_root_z_drift=0.40
  survived 95/500 steps, reset_seen=True
```

当前结论：

- 还没有找到可以直接写入 A3 训练配置的稳定自由基站姿。
- 参考图那种持拍下蹲姿态不是唯一问题；即使 default arms + zero legs，也会在短时间内明显倾倒或触发 reset。
- 因此当前阶段不应继续猜右臂击球姿态，也不应直接训练乒乓球任务。
- 更可能需要先检查 A3 的站立物理链路：root 高度、脚底 collision、质心/惯量、PD stiffness/damping/effort、脚底摩擦、URDF 转 USD 后的碰撞质量。

后续建议优先级：

1. 单独做 A3 站立物理诊断，不带球拍动作目标，先确认 zero/near-zero pose 是否能静态站立。
2. 可视化检查脚底 collision 是否真实贴地、是否存在脚底凸包/视觉网格不匹配。
3. 对 A3 leg/feet actuator 做单独 PD 扫描，尤其是 ankle stiffness/damping 和 hip/knee effort。
4. 在自由基稳定前，继续用 `--pin_base` 标定球拍几何；但不要把 pin-base 下看起来合理的姿态直接用于训练。

相关结果 CSV：

```text
logs/a3_standing_calibration/2026-06-25_19-41-56_results.csv
logs/a3_standing_calibration/2026-06-25_19-43-24_results.csv
logs/a3_standing_calibration/2026-06-25_19-44-44_results.csv
logs/a3_standing_calibration/2026-06-25_19-45-31_results.csv
```

## 2026-06-25 A3 电机参数表补充分析

用户补充 A3 电机/减速器参数截图。该表包含各关节电机型号、gear ratio、额定/峰值扭矩、额定/峰值转速，以及仿真中建议使用的高速端等效惯量换算方式。

关键公式：

```text
J_output = J_motor * N^2
rpm_to_rad_s = rpm * 2 * pi / 60
```

按截图计算得到的串联关节等效 armature：

```text
PFP-110-75: 0.12034028684
PFP-93-65:  0.06646569891
PFP-78-58:  0.01208336871
PFP-59-60:  0.00496735130
PFP-41-48:  0.00081008933
```

并联等效补充：

```text
ankle_pitch: 0.06444060531
ankle_roll:  0.02012630058
waist_pitch: 0.08820859156
waist_roll:  0.01462087613
```

和当前 `a3.py` 对比：

- 当前 torque/velocity limit 大部分已经参考了该表：
  - hip / waist_yaw 约 `220 Nm`, `12.04 rad/s`
  - knee 约 `320 Nm`, `14.66 rad/s`
  - ankle_pitch 约 `118.2 Nm`, `10.8 rad/s`
  - ankle_roll 约 `54.75 Nm`, `19.37 rad/s`
- 当前 `armature=0.01` 是统一占位值，明显没有按表中 `J_output` 写入。
- PFP-59-60 相关的 upper J3/J4/J5 峰值扭矩应为 `36 Nm`，当前 shoulder_yaw / elbow / wrist_roll 使用的 `24 Nm` 偏保守。
- PFP-41-48 额定转速为 `150 rpm = 15.708 rad/s`，当前 wrist_pitch / wrist_yaw / head 的 `12.78 rad/s` 偏保守。

判断：

- 初始姿态不能摔倒，这是 A3 乒乓任务成立的前提。
- 当前站立筛选失败不应归咎于 reward 或 predictor。
- 在继续姿态搜索前，应先把 A3 actuator armature 和明显偏保守的限幅修到截图参数，否则站立物理链路和真实 A3 不一致。
- 修正 armature 后仍需继续检查脚底 collision、root 高度、质心/惯量和 PD 增益；armature 修正只是必要条件，不是充分条件。

修正 actuator 后复测：

```text
128 candidates, 250 steps
0 candidates survived 250 steps
best: candidate_index=52, light_crouch_default_arms_z1.02
survival_steps=116/250
```

该结果说明电机参数修正确实改善了候选评分，但 A3 仍未达到“初始姿态自由基稳定站立”的要求。后续重点应转到脚底碰撞、root 高度/脚底贴地、link inertia/CoM 和 PD 增益，而不是继续直接训练乒乓球策略。

## 2026-06-25 A3 root 高度/脚底接触/PD 诊断结果

新增 A3-only 诊断脚本后，先把问题拆成两个层面：

```text
pinned root:
  固定机器人 root，只看脚底是否贴地、是否悬空或穿地、左右脚支撑力和脚底滑移。

free base:
  释放机器人 root，看同一组 root_z 和 PD 下是否能自由基短时间稳定站立。
```

核心结果：

```text
pinned current pose:
  root_z=0.90: contact_ratio=0.780, foot_slip=0.6623
  root_z=0.96: contact_ratio=0.980, foot_slip=0.1547
  root_z=1.02: contact_ratio=1.000, foot_slip=0.0399
  root_z=1.08: contact_ratio=0.000, foot_slip=0.0027

free current pose:
  root_z=0.90: 92/250 steps
  root_z=1.02: 101/250 steps

free current pose + leg/feet damping x2:
  root_z=1.02: 114/250 steps

free current pose + leg/feet stiffness x1.5 + damping x2:
  root_z=1.02: 94/250 steps

free light_crouch pose + leg/feet damping x2:
  root_z=1.02: 140/250 steps, reset_seen=False, max_abs_roll_pitch=0.5517
```

判断：

- A3 当前初始 root 高度 `0.90` 不适合作为训练起点，脚底接触滑移非常大。
- `root_z=1.02` 是目前最合理的脚底贴地区间；`1.08` 已经出现悬空。
- 站不稳不是单纯因为 actuator 太弱；增加 stiffness 反而更差，damping-only 更有效。
- 最佳候选 `light_crouch + root_z=1.02 + leg/feet damping x2` 仍然只是接近稳定，不能直接当作最终训练姿态。
- 这解释了为什么 A3 乒乓训练几乎没有学到击球：策略刚开始还没有机会稳定站在可击球空间附近，reward 和 predictor 的信号被站立失败/姿态漂移淹没。

因此，下一步不是继续扩大训练迭代，而是：

1. 用 WebRTC 可视化 `root_z=1.02` 和 `light_crouch` 候选，确认脚底 collision 是否真实贴地。
2. 观察自由基倾倒方向，是前后倒、左右倒，还是脚底碰撞/摩擦造成滑移。
3. 小范围微调 leg/ankle/waist damping 和 crouch 姿态。
4. 确认能稳定站立后，再把 A3 专用 reset pose 写入 A3 pingpong 配置。

### 继续确认后的更新

进一步加入 reset 原因诊断后，发现之前失败不是诊断阈值误判，而是机器人确实会在 table frame 下漂移或倒下：

```text
light_crouch + default arms:
  reset around 100-140 steps
  y drifts to about 1.10+, or x reaches about -1.34

ready_light_crouch:
  reset around 110-125 steps
  roll/pitch grows to about 1.3 rad

base_y=0.0:
  still fails
  robot_pos.z drops to about 0.47-0.50
```

这说明问题不是站位 y 边界本身，而是自由基站立失稳。随后加入 zero-arm 对照后，站立稳定性明显改善：

```text
zero_light_crouch + root_z=1.04 + leg/feet damping x2:
  500/500 steps, no env reset
  max_abs_roll_pitch=0.5584

zero_light_crouch + root_z=1.05 + waist damping x2 + leg/feet damping x3:
  1000/1000 steps, no env reset
  max_abs_roll_pitch=0.5412
  max_root_z_drift=0.1796
  both_feet_contact_ratio=0.981
```

当前最重要的判断更新：

- A3 站不稳不只是 root 高度或脚底 collision 问题，上肢/球拍非对称姿态也是重要扰动源。
- 首个可作为基线的稳定候选曾是：

```text
root_z = 1.05
legs = light_crouch: hip_pitch=-0.12, knee=0.25, ankle_pitch=-0.14
arms = zero arms
waist damping scale = 2.0
leg damping scale = 3.0
feet damping scale = 3.0
```

- 这个候选不是最终乒乓击球姿态，只是“先能站住”的 reset 基线。
- 右臂持拍姿态应该在该稳定基线上逐步引入，例如先只抬右臂少量角度，再观察 500-1000 step 是否仍能保持稳定。

进一步搜索 zero-arm 宽站姿后，找到了更稳的站立基线候选：

```text
candidate_index = 205
root_z = 1.055
hip_pitch = -0.18
knee = 0.38
ankle_pitch = -0.20
left_hip_roll = 0.10
right_hip_roll = -0.10
left_hip_yaw = 0.00
right_hip_yaw = 0.00
left_ankle_roll = -0.045
right_ankle_roll = 0.045
arms = zero arms
waist damping scale = 2.0
leg damping scale = 3.0
feet damping scale = 3.0
```

单独 1000 step 复测结果：

```text
survival=1000/1000
reset_seen=False
max_abs_roll_pitch=0.4789
max_root_z_drift=0.1886
max_foot_slip=0.0951
both_feet_contact_ratio=0.991
```

和前一版 `zero_light_crouch` 基线相比：

```text
roll/pitch: 0.5412 -> 0.4789
foot_slip:  0.1976 -> 0.0951
```

因此当前 A3 站立基线应优先使用 `candidate_index=205`。它仍然不是最终持拍姿态，但已经足够作为“先能站住”的 A3 reset pose 候选。后续应先 WebRTC 可视化确认，再写入 A3 pingpong 专用配置。

1. 可视化标定右手 paddle offset。
2. 添加真实 paddle collision/visual，并让物理碰撞和虚拟击球点一致。
3. 调整 base 初始高度和 reset range。
4. 调整 `action_scale`、PD、delay 和 effort scale。
5. 根据 A3 训练曲线重调 energy、feet、joint deviation、right arm/wrist regularization 权重。
6. 根据训练稳定性决定是否固定头部或部分腕关节。
7. 检查 URDF 转 USD 后的 collision 质量，必要时生成专用 USD。
8. 后续如果需要部署，再处理串联等效关节到真实并联机构的动作映射。

## 不建议首版做的事项

1. 不建议直接替换 T1 asset 路径来复用 `t1_tt_eval`。
2. 不建议直接加载 T1 checkpoint 到 A3。
3. 不建议在未验证 paddle offset 前做长时间大规模训练。
4. 不建议一开始就大改 `TTEnv` 主流程；公共代码只做配置化扩展。
5. 不建议把 predictor 作为首轮主要改动对象。

## 结论

A3 接入应视为“同一乒乓任务框架下的新机器人任务”，不是 T1 的模型文件替换。球、球桌、空气动力、球路预测和 runner 流程可以复用；机器人资产、动作维度、关节/身体命名、奖励正则、击球点、初始姿态和 actuator 参数必须做 A3 专用适配。

首轮目标应是让 `a3_tt` 完成环境构建和短 rollout；第二轮目标是站姿、右手击球点和物理碰撞可视化正确；第三轮再进入 reward/PD/action scale 的训练优化。
## 2026-06-29 T1 强化学习流程复查与 A3 对比

结论先行：

- 原始 T1 代码并没有使用显式 staged curriculum；
- T1 使用的是完整乒乓球任务长训：
  - 完整稳定性惩罚；
  - 完整任务 reward；
  - actor 使用 MLP predictor 给出的 `ball_prediction`；
  - critic/reward 使用环境物理计算的 `ball_future_pose`；
  - policy action 作为默认关节姿态附近的偏移，再交给 Isaac articulation PD controller；
- T1 之所以可行，关键不在“有隐藏的挥拍先验阶段”，而在于 T1 的默认姿态、球拍安装几何、PD 参数、action scale 和 reset 随机范围一起构成了一个可学习的初始分布。

T1 代码路径：

- 任务配置：`legged_lab/envs/t1_tt/t1_tt_config.py`
- 资产/默认姿态/PD：`legged_lab/assets/booster/booster.py`
- action 到 PD target：`legged_lab/envs/base/tt_env.py`
- reset 事件：`legged_lab/mdp/events.py`
- predictor runner：`rsl_rl/rsl_rl/runners/on_policy_predictor_regression_runner.py`

T1 reward 结构：

- 稳定性托底：
  - `lin_vel_z_l2`
  - `ang_vel_xy_l2`
  - `ang_vel_z_l2`
  - `energy`
  - `action_rate_l2`
  - `undesired_contacts`
  - `fly`
  - `flat_orientation_l2`
  - `termination_penalty`
  - `hit_unstable_support`
  - `feet_orientation_*`
  - `feet_slide`
  - `feet_force`
  - `feet_too_near`
  - `feet_stumble`
  - `dof_pos_limits`
  - `joint_deviation_*`
- 任务信号：
  - `reward_contact`
  - `reward_future_dis_ee`
  - `reward_future_dis_ro`
  - `reward_future_vel_base`
  - `reward_future_landing_dis`
  - `reward_future_pass_net`
  - `reward_table_success`

T1 没有显式使用：

- 独立 standing-only 阶段；
- reward weight linear curriculum；
- 专门的 hit velocity reward；
- 专门的 own-table after-hit penalty；
- 专门的 post-hit net progress reward。

T1 的关键隐含条件：

- `BOOSTER_T1_TT_P2_CFG` 已经是乒乓球 ready 资产：
  - root 初始位置为 `(-1.6, 0.0, 0.72)`；
  - 下肢默认轻微屈膝：hip pitch `-0.20`，knee `0.42`，ankle pitch `-0.23`；
  - 右臂默认姿态已经适配球拍：
    - `Right_Shoulder_Pitch = 0.0`
    - `Right_Shoulder_Roll = -0.1`
    - `Right_Elbow_Pitch = 0.2`
    - `Right_Elbow_Yaw = 0.5`
- T1 的 policy action 不是直接力矩，而是：
  - `processed_actions = action * action_scale + default_joint_pos`
  - 然后调用 `set_joint_position_target`
  - 实际由 Isaac articulation actuator PD 跟踪；
- T1 默认 `action_scale = 0.25`，初始 policy noise 较大，但由于默认姿态和 PD 可承受，探索不会一开始就完全破坏任务；
- T1 训练 reset 范围较大：
  - base pose：`x=(-0.6, 0.02)`，`y=(-0.6, 0.8)`，`yaw=(-0.3, 0.3)`；
  - base velocity：约 `(-0.2, 0.2)`；
  - locomotion joints 按默认姿态 `0.5-1.5` scale；
  - manipulation joints 在默认姿态附近 `(-0.5, 0.5)` offset；
- 这说明 T1 的默认姿态/PD/球拍几何足够稳定，能够支撑较大的探索。

A3 与 T1 的关键差异：

- A3 初始迁移时没有 T1 这种已经验证过的 table-tennis ready 默认姿态；
- A3 下肢更高、更重、更复杂，root 高度、髋 roll/踝 roll、膝屈曲不匹配时会直接摔倒；
- A3 球拍安装和击球中心虽然可以复用模型，但局部 offset、右腕姿态、拍面朝向必须重新标定；
- A3 如果一开始默认几何让球路和击球中心不重合，`reward_contact` 和后续 `reward_table_success` 会非常稀疏；
- A3 如果 action scale / policy noise 像 T1 一样大，早期更容易把站姿破坏掉；
- A3 如果稳定惩罚太弱，策略可能用摔倒/低 base 去偶然碰球；
- A3 如果 own-table 或过网成功目标太早太强，策略又可能因为早期几乎拿不到高阶 reward 而退回保守局部解。

对 stage5c 的解释：

- stage5c 不是对 T1 方法的否定；
- stage5c 是为了让 A3 的训练分布逐步接近 T1 已经天然具备的条件：
  - 稳定默认姿态；
  - policy 在默认姿态附近探索；
  - 有足够密度的球路/击球点接近 reward；
  - 触球后再逐步强化过网和落点；
- 如果后续 A3 的默认姿态、球拍朝向和 PD 调到足够接近 T1 的可学习条件，理论上也可以减少甚至取消显式 reward curriculum，回到更接近 T1 的单阶段长训。

## 2026-06-29 T1 训练曲线复查：2026-06-22_21-09-35

数据来源：

- TensorBoard event：
  - `logs/t1_table_tennis/2026-06-22_21-09-35/events.out.tfevents.1782133775.FdseRobot-02.259724.0`
- checkpoint：
  - `model_0.pt` 到 `model_9999.pt`
- eval CSV：
  - `logs/t1_table_tennis/2026-06-22_21-09-35/eval_result/eval_results.csv`

训练基本信息：

- 训练长度：10000 iter；
- 保存间隔：250 iter；
- `num_envs = 4096`；
- `num_steps_per_env = 24`；
- predictor 训练到 iter 20；
- 从 2026-06-22 21:09 运行到 2026-06-23 05:02；
- 主配置：
  - `action_scale = 0.25`；
  - ball speed x `(-6.5, -5.0)`；
  - ball speed y `(-0.8, 0.4)`；
  - ball speed z `(1.5, 2.0)`；
  - train reset base pose x `(-0.6, 0.02)`，y `(-0.6, 0.8)`，yaw `(-0.3, 0.3)`；
  - locomotion joint reset scale `(0.5, 1.5)`；
  - manipulation joint reset offset `(-0.5, 0.5)`。

关键曲线阶段：

| 阶段 | 现象 |
| --- | --- |
| 0-500 iter | 平均 reward 很差，success 近乎 0；但 episode length 已从约 18 提到约 270，说明先在学稳定和不早死 |
| 500-1000 iter | 仍几乎没有成功；`reward_contact` 约 0.03，`reward_table_success` 约 0.002，说明任务还没打开 |
| 1000-2000 iter | 接触和成功开始抬头；`reward_contact` 到 2000 附近约 0.25，`reward_table_success` 到约 0.11 |
| 2000-3000 iter | 乒乓球任务明显打开；`reward_contact` 均值约 0.42，`reward_table_success` 均值约 0.19 |
| 3000-5000 iter | 最佳学习区间；`reward_contact` 均值约 0.54，`reward_table_success` 均值约 0.35，`reward_future_pass_net` 均值约 0.45 |
| 5000-7500 iter | 仍较好，但开始缓慢回落 |
| 7500-10000 iter | 成功相关 reward、episode length 明显下降；最终模型可用，但训练曲线不是最优点 |

滚动 200 iter 最佳点：

- `Train/mean_reward` 最大约 `12.58`，出现在约 iter `5228`；
- `reward_contact` 最大滚动均值约 `0.562`，出现在约 iter `4600`；
- `reward_table_success` 最大滚动均值约 `0.400`，出现在约 iter `5159`；
- `reward_future_pass_net` 最大滚动均值约 `0.475`，出现在约 iter `5156`；
- `reward_future_landing_dis` 最大滚动均值约 `0.828`，出现在约 iter `5159`；
- `mean_episode_length` 最大滚动均值约 `345`，出现在约 iter `4254`。

最终 iter 9999 指标：

- `Train/mean_reward = -4.58`；
- `Train/mean_episode_length = 155.0`；
- `Train/TT_hit_rate = 0.466`；
- `Train/TT_success_rate = 0.270`；
- `Episode_Reward/reward_contact = 0.236`；
- `Episode_Reward/reward_future_pass_net = 0.199`；
- `Episode_Reward/reward_table_success = 0.173`；
- `Episode_Reward/termination_penalty = -0.904`。

判断：

- T1 的学习不是“一开始就会打球”；
- T1 也是先学稳定和延长 episode，再逐步打开接触和过网/落点；
- 真正任务打开大约在 1500-2500 iter 之后；
- 最强区间在 4000-5500 iter；
- 9999 checkpoint 不是曲线上的最优点，后期可能因为持续探索、较大 reset 分布或策略漂移导致回落；
- 但 T1 仍然可用，说明其默认姿态/PD/球拍几何足够好，允许单阶段长训穿过早期稀疏区。

eval CSV 结果：

- 总记录：4268 个 serve；
- label 统计：
  - `hit = 3647`；
  - `success = 509`；
  - `missed = 112`；
- 如果 `hit + success` 都视为触球，eval 触球率约 `97.4%`；
- success label 单独占比约 `11.9%`；
- eval 脚本的 label 口径和训练 `Train/TT_success_rate` 口径不同，不能直接一一对齐。

对 A3 的启发：

- A3 不需要被要求一开始就学会完整打球；
- 但 A3 必须像 T1 一样，先具备足够可靠的默认几何和 PD 托底，使训练能跨过 0-1500 iter 的稀疏区；
- A3 当前的问题是早期稳定性和击球几何没有 T1 那么自然，因此 `stage5c` 的软 curriculum 是为了模拟 T1 隐含具备的学习条件，而不是另起一套完全不同的方法；
- 对 A3 长训的评估不应该只看最终 checkpoint，也应该保存并评估 2000、3000、4000、5000、6000 等中间 checkpoint，防止错过最佳区间。
