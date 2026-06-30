# PACE-ICRA2026 代码修改日志（2026-06-22）

## 背景

本次目的是做最小训练验证，确认代码环境是否可以跑通。使用的命令为：

```bash
CUDA_VISIBLE_DEVICES=1 python legged_lab/scripts/train.py \
  --task=t1_tt \
  --logger=tensorboard \
  --num_envs=64 \
  --headless \
  --predictor
```

## 复现到的报错

训练启动后，在第一次环境 `step()` 过程中报错：

```text
Traceback (most recent call last):
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/scripts/train.py", line 109, in <module>
    train()
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/scripts/train.py", line 103, in train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  File "/mnt/ssd/zxl/PACE-ICRA2026/rsl_rl/rsl_rl/runners/on_policy_predictor_regression_runner.py", line 169, in learn
    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/envs/base/tt_env.py", line 745, in step
    self.aero.apply_to_rigid_object(self.ball)
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/physics/aerodynamics.py", line 93, in apply_to_rigid_object
    ball_asset.set_external_force_and_torque(
TypeError: RigidObject.set_external_force_and_torque() got an unexpected keyword argument 'is_global'
```

## 问题原因

根因是 `legged_lab/physics/aerodynamics.py` 中调用了：

```python
ball_asset.set_external_force_and_torque(
    forces=F_w.unsqueeze(1), torques=T_w.unsqueeze(1), is_global=True
)
```

但是本地使用的 IsaacLab 4.5 源码中，`RigidObject.set_external_force_and_torque()` 的签名不接受 `is_global` 参数，而且该接口要求输入的外力和外力矩位于 **刚体局部坐标系**，不是世界坐标系。

也就是说，这里不是简单“删掉一个参数”就结束了：

1. 直接保留世界坐标系的 `F_w / T_w` 传入，会造成坐标系不匹配。
2. 即使代码不再报 `TypeError`，空气动力的实际施加方向也可能是错的。

## 修改目标

让仓库中的空气动力施加逻辑兼容当前 IsaacLab 4.5 接口，同时保持物理语义正确：

1. 去掉不被支持的 `is_global` 参数；
2. 将空气动力模型输出的世界坐标系力/矩转换到 ball 的局部坐标系；
3. 再调用 `set_external_force_and_torque()`。

## 修改内容

修改文件：

- `legged_lab/physics/aerodynamics.py`

具体修改：

1. 新增：

```python
import isaaclab.utils.math as math_utils
```

2. 在 `apply_to_rigid_object()` 中，增加世界系到局部系的转换：

```python
root_quat_w = ball_asset.data.root_link_quat_w
F_b = math_utils.quat_rotate_inverse(root_quat_w, F_w)
T_b = math_utils.quat_rotate_inverse(root_quat_w, T_w)
```

3. 将最终接口调用改为：

```python
ball_asset.set_external_force_and_torque(
    forces=F_b.unsqueeze(1), torques=T_b.unsqueeze(1)
)
```

## 为什么这样修改

因为：

1. `F_w` 和 `T_w` 是空气动力模型在世界坐标系下计算出来的结果；
2. IsaacLab 4.5 的 `RigidObject.set_external_force_and_torque()` 需要的是刚体局部坐标系下的输入；
3. `quat_rotate_inverse(root_link_quat_w, ·)` 可以把世界向量旋转到刚体局部坐标系；
4. 这样既解决了接口不兼容，也避免了力方向错误带来的隐性物理问题。

## 修改位置索引

- 报错入口：`legged_lab/envs/base/tt_env.py:745`
- 原始问题位置：`legged_lab/physics/aerodynamics.py:93`
- 本次修改文件：`legged_lab/physics/aerodynamics.py`

## 备注

这次修复的是一个 **代码接口兼容性问题**，不是训练配置问题。

日志中还出现了若干 warning，例如：

- `CUDA_VISIBLE_DEVICES` 可能导致 Omniverse / IsaacSim 设备枚举行为与 CUDA 不一致；
- 材质、显示、GPU 状态等 warning。

这些 warning 不一定会阻止程序启动，但这次导致训练中断的直接原因是 `aerodynamics.py` 中对 `set_external_force_and_torque()` 的错误调用。

---

## 第二次新增报错（2026-06-22）

在修复 `aerodynamics.py` 之后，训练继续向前执行，并暴露出下一个 IsaacLab 4.5 兼容性问题：

```text
Traceback (most recent call last):
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/scripts/train.py", line 109, in <module>
    train()
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/scripts/train.py", line 103, in train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  File "/mnt/ssd/zxl/PACE-ICRA2026/rsl_rl/rsl_rl/runners/on_policy_predictor_regression_runner.py", line 169, in learn
    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/envs/base/tt_env.py", line 781, in step
    reward_buf = self.reward_manager.compute(self.step_dt)
  File "/mnt/ssd/zxl/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py", line 148, in compute
    value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
  File "/mnt/ssd/zxl/PACE-ICRA2026/legged_lab/mdp/rewards.py", line 257, in body_orientation_l2
    body_orientation = math_utils.quat_apply_inverse(
AttributeError: module 'isaaclab.utils.math' has no attribute 'quat_apply_inverse'. Did you mean: 'quat_rotate_inverse'?
```

## 第二次报错原因

`legged_lab/mdp/rewards.py` 中仍然使用了旧函数名 `quat_apply_inverse()`。

但在你本地 IsaacLab 4.5 的 `isaaclab.utils.math` 中，对应接口名称是：

```python
quat_rotate_inverse()
```

因此在 reward 计算阶段，程序会因为找不到旧接口而中断。

## 第二次修改内容

修改文件：

- `legged_lab/mdp/rewards.py`

替换了以下两处调用：

1. `track_lin_vel_xy_yaw_frame_exp()`
2. `body_orientation_l2()`

将：

```python
math_utils.quat_apply_inverse(...)
```

改为：

```python
math_utils.quat_rotate_inverse(...)
```

## 第二次修改原因

这属于 IsaacLab 不同版本之间的数学工具函数命名差异。

在当前环境里：

- `quat_apply()` 仍然存在；
- 但 `quat_apply_inverse()` 不存在；
- 需要使用 `quat_rotate_inverse()` 作为等价替代。

为了减少“修一个、再冒一个”的情况，这次额外检查了仓库内同类调用，确认目前 `legged_lab` 目录下的 `quat_apply_inverse` 只出现了上述两处，已经一并修复。
