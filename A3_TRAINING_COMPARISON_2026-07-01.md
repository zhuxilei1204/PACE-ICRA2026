# A3 训练对照日志（2026-07-01）

## 对照目标

比较 A3 在同一几何修正、同一宽球路、同一严格 contact 判定下，仅加入“拍面法向、击球速度、击球窗口”优化是否能改善训练效果。

## 实验 A：stage5g_wide

- GPU：0
- tmux：`a3_stage5g_wide_4096`
- task：`a3_tt_stage5g_wide`
- run：`logs/a3_table_tennis/2026-07-01_00-53-29_a3_stage5g_wide_a3adapted_10000_4096`
- command：

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

配置要点：

- A3 FK 几何 offset：`x=0.224`, `y=-0.397`
- A3 适配宽球路：`vx=(-5.90,-4.95)`, `vy=(-0.42,0.18)`, `vz=(1.42,1.85)`, `pos_y=(-0.09,0.09)`
- `contact_threshold=0.04`
- 关闭 ball curriculum
- 不包含新增拍面法向/击球窗口/挥拍速度 reward

当前状态：

- 已进入长训；
- 早期 hit 未虚高；
- 需要重点观察 250/500/1000 iter 的 `TT_hit_rate`, `TT_success_rate`, `mean_episode_length`, `undesired_contacts`, `reward_contact`, `reward_future_pass_net`。

快照（约 iter 273）：

- `TT_hit_rate=0.0125`
- `TT_success_rate=0`
- `mean_episode_length=73.48`
- `undesired_contacts=-0.217`
- `termination_penalty=-3.20`
- `reward_contact=0.00116`
- `reward_future_pass_net=2.12e-05`

## 实验 B：stage5h_hitquality

- GPU：1
- tmux：`a3_stage5h_hitquality_4096`
- task：`a3_tt_stage5h_hitquality`
- run：`logs/a3_table_tennis/2026-07-01_01-08-13_a3_stage5h_hitquality_a3adapted_10000_4096`
- command：

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

与实验 A 完全相同：

- A3 FK 几何 offset；
- A3 适配宽球路；
- `contact_threshold=0.04`；
- 关闭 ball curriculum；
- `num_envs=4096`, `max_iterations=10000`。

新增差异：

- 拍面法向：诊断确认 A3 球拍正面为本地 `-Z`，奖励窗口内法向朝向来球 `-ball_linvel`；
- 击球窗口：用 `ball_future_t` 和 `ball_future_pose` 奖励窗口内接近预测击球点；
- 挥拍速度：窗口内奖励球拍朝对面桌方向的速度；
- 出球质量：加强并收紧 `reward_hit_ball_velocity_net`, `reward_hit_net_clearance_progress`, `reward_future_pass_net`, `reward_post_hit_net_progress`。

新增 TensorBoard reward：

- `Episode_Reward/reward_strike_window_touch_point`
- `Episode_Reward/reward_paddle_normal_alignment`
- `Episode_Reward/reward_paddle_swing_velocity`

当前状态：

- 64-env smoke：通过；
- 4096-env smoke：通过；
- GPU1 长训已启动并进入 PPO；
- 前几轮新增 reward 均有非零值。

快照（约 iter 24）：

- `TT_hit_rate=0`
- `TT_success_rate=0`
- `mean_episode_length=106.06`
- `undesired_contacts=-19.45`
- `termination_penalty=-2.65`
- `reward_contact=0.000386`
- `reward_future_pass_net=3.78e-07`
- `reward_strike_window_touch_point=0.349`
- `reward_paddle_normal_alignment=0.158`
- `reward_paddle_swing_velocity=0.0486`

## 对比判断

重点比较：

- 是否更快获得真实 `reward_contact`，而不是只抬高 dense shaping；
- `TT_hit_rate` 是否提高但不过度虚高；
- `reward_hit_ball_velocity_net` / `reward_future_pass_net` 是否早于实验 A 打开；
- `TT_success_rate` / `reward_table_success` 是否最终优于实验 A；
- `undesired_contacts` 和 `termination_penalty` 是否因挥拍/法向 reward 变差。

若实验 B 的接触和出球质量提升，但稳定性变差，下一步应降低 `reward_paddle_swing_velocity` 或提高稳定 gate；若稳定性相当且 success 更好，可保留 hitquality 方向继续长训或微调落点目标。

## 2026-07-01 中途状态快照

状态：

- 两个任务均未完成，tmux session 仍在运行；
- `stage5g_wide`：约 `9413/10000` iter，最新 checkpoint `model_9250.pt`，预计还需约 36 分钟；
- `stage5h_hitquality`：约 `7327/10000` iter，最新 checkpoint `model_7250.pt`，预计还需约 3 小时 24 分钟；
- TensorBoard 已运行在 `0.0.0.0:16006`。

当前 TensorBoard 指标：

| run | TT_hit_rate | TT_success_rate | mean_episode_length | reward_contact | reward_future_pass_net | reward_table_success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage5g_wide | 0.00077 | 0 | 98.74 | 0.000046 | 0 | 0 |
| stage5h_hitquality | 0.03733 | 0 | 77.57 | 0.00104 | 0.00000003 | 0 |

中途判断：

- `stage5h_hitquality` 的 hit/contact 明显高于 `stage5g_wide` 当前值，但两个 run 的 `success_rate` 仍为 0；
- `stage5h_hitquality` 的新增 shaping reward 正常生效：`reward_strike_window_touch_point≈0.402`，`reward_paddle_normal_alignment≈0.353`，`reward_paddle_swing_velocity≈0.168`；
- 当前还不能说哪个最终更好，关键要等两者完成后看 `success_rate`、`future_pass_net`、落台相关 reward，以及 WebRTC 可视化是否真的打出有效回球。

## 2026-07-01 中后期分析

状态：

- `stage5g_wide`：约 `9636/10000` iter；
- `stage5h_hitquality`：约 `7501/10000` iter；
- 两者 `TT_success_rate` 仍为 0。

关键曲线：

| run | hit_rate 峰值 | 当前 hit_rate | 当前 mean_reward | 当前 reward_contact | 当前 future_pass_net | 当前 table_success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage5g_wide | 0.0923 @599 | 0.00081 | 38.64 | 0.000069 | 4.3e-15 | 0 |
| stage5h_hitquality | 0.0881 @4399 | 0.0343 | 5.81 | 0.00121 | 1.44e-05 | 0 |

分析：

- `stage5g_wide` 的高 `mean_reward` 主要来自几何/dense 项，例如 `reward_future_touch_point≈0.675`，不是来自真实接触或过网；
- 两个 run 的 hit rate 都出现“先升后降”，说明早期探索/随机动作产生了一些接触，但 PPO 后期更偏向稳定、接近未来点等低风险 dense reward；
- `stage5h_hitquality` 保住了更多 hit/contact，但 `future_pass_net` 仍极小、`table_success` 始终为 0，说明触球没有转化为有效出球；
- 这不是单纯训练轮数不够。当前奖励链中，成功回球的稀疏信号从未出现，继续同配置长训大概率只会继续优化 dense reward 或局部 hit，而不会自然打开成功率。

建议：

- 让当前两个 10000-iter run 跑完，作为完整对照；
- 不建议直接从当前配置继续 resume 更长训练；
- 下一步优先评估 hit 峰值附近 checkpoint 的 WebRTC 行为：
  - `stage5g_wide`: `model_500.pt`, `model_750.pt`
  - `stage5h_hitquality`: `model_4250.pt`, `model_4500.pt`, `model_5000.pt`
- 后续优化应围绕“真实有效击球”而不是继续堆 dense mean reward：收紧 true-hit 指标、增加过网前的有效出球奖励、降低可被静态接近刷分的 dense 项，并考虑按能力重新引入 ball curriculum。

## 2026-07-01 后续实验安排：stage5h 续训 + stage5i 稳定增强

用户 WebRTC 观察结论：

- `stage5h_hitquality` 已经能明显看到挥拍动作；
- `stage5g_wide` 和 `stage5h_hitquality` 都没有学会站稳，身体仍会向前倾倒；
- 因此需要一条线继续验证 `stage5h` 长训是否能改善，另一条线专门验证“站稳约束增强”是否能解决前扑式挥拍。

实验 C：`stage5h_hitquality` 续训

- 目标：保留当前 hitquality 配置不变，从第一段 10000 iter 的最新 checkpoint 继续训练 10000 iter；
- 总训练量：约 20000 iter；
- 用途：验证仅靠更长训练是否能从“有挥拍但站不稳”自然过渡到稳定击球；
- 不修改代码。
- 状态：已启动 watcher，等待 `2026-07-01_01-08-13_a3_stage5h_hitquality_a3adapted_10000_4096/model_9999.pt` 出现后自动 resume。

计划 resume 命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5h_hitquality \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5h_hitquality_resume10000_to20000_4096 \
  --resume True \
  --load_run 2026-07-01_01-08-13_a3_stage5h_hitquality_a3adapted_10000_4096 \
  --checkpoint model_9999.pt \
  --headless \
  --predictor
```

实验 D：`stage5i_stable_hitquality`

- 目标：在 `stage5h_hitquality` 基础上只增强 A3 站稳约束，不改变 T1，也不覆盖 `stage5h`；
- 新 task：`a3_tt_stage5i_stable_hitquality` / `a3_tt_stage5i_stable_hitquality_eval`；
- 核心假设：当前失败不是 hitquality 无效，而是策略学到了“前扑接近球路/挥拍”的捷径。

修改原则：

- 保留 `stage5h` 的 A3 FK 几何、宽球路、严格 contact、拍面法向、击球窗口、挥拍速度；
- 增强稳定评分里的速度项，尤其抑制 base 前向速度和角速度；
- 降低不稳定状态下击球/挥拍 reward 的 gate floor，避免前倒时仍保留过多任务奖励；
- 增加击球窗口前后的 forward-fall penalty，使“前扑挥拍”在 contact 前也被惩罚；
- 暂时不重新引入 ball curriculum，避免和稳定增强变量混在一起。

启动状态：

- smoke test：`a3_tt_stage5i_stable_hitquality`, `num_envs=1`, `max_iterations=1` 通过；
- GPU：0；
- tmux：`a3_stage5i_stable_hitquality_4096`；
- run：`logs/a3_table_tennis/2026-07-01_12-59-55_a3_stage5i_stable_hitquality_10000_4096`。

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxl/miniconda3/envs/zxl-pace/bin/python -m legged_lab.scripts.train \
  --task=a3_tt_stage5i_stable_hitquality \
  --logger=tensorboard \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=a3_stage5i_stable_hitquality_10000_4096 \
  --headless \
  --predictor
```

早期日志（约 iter 5）：

- `mean_reward≈-404.5`
- `mean_episode_length≈139.3`
- `reward_standing_stability≈11.19`
- `penalty_forward_fall_during_strike≈-0.162`
- `reward_paddle_swing_velocity≈0.028`
- `reward_table_success=0`

早期判断：

- 新惩罚项正常生效；
- 初期回报显著变负，主要因为稳定约束更强且 A3 仍有大量 undesired contact；
- 这版不应只看早期 hit rate，重点看 250/500/1000 iter 后 `mean_episode_length`、`termination_penalty`、`undesired_contacts`、`penalty_forward_fall_during_strike` 是否下降，以及 WebRTC 中身体是否仍前扑。
