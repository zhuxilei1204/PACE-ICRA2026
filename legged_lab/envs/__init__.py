# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.legged_env import LeggedEnv
from legged_lab.envs.base.tt_env import TTEnv

from legged_lab.envs.t1_tt.t1_tt_config import (
    T1TableTennisEnvCfg,
    T1TableTennisAgentCfg,
    T1TT_EvalEnvCfg,
)
from legged_lab.envs.a3_tt.a3_tt_config import (
    A3TableTennisEnvCfg,
    A3TableTennisAgentCfg,
    A3TT_EvalEnvCfg,
    A3StableReturnEnvCfg,
    A3StableReturnEvalEnvCfg,
    A3StableReturnAgentCfg,
    A3Stage4bEnvCfg,
    A3Stage4bEvalEnvCfg,
    A3Stage4bAgentCfg,
    A3Stage4cEnvCfg,
    A3Stage4cEvalEnvCfg,
    A3Stage4cAgentCfg,
    A3Stage4dEnvCfg,
    A3Stage4dEvalEnvCfg,
    A3Stage4dAgentCfg,
    A3Stage5ReadyEnvCfg,
    A3Stage5ReadyEvalEnvCfg,
    A3Stage5ReadyAgentCfg,
    A3Stage5bEnvCfg,
    A3Stage5bEvalEnvCfg,
    A3Stage5bAgentCfg,
    A3Stage5cEnvCfg,
    A3Stage5cEvalEnvCfg,
    A3Stage5cAgentCfg,
    A3Stage5dEnvCfg,
    A3Stage5dEvalEnvCfg,
    A3Stage5dAgentCfg,
    A3Stage5eEnvCfg,
    A3Stage5eEvalEnvCfg,
    A3Stage5eAgentCfg,
    A3Stage5fEnvCfg,
    A3Stage5fEvalEnvCfg,
    A3Stage5fAgentCfg,
    A3Stage5gEnvCfg,
    A3Stage5gEvalEnvCfg,
    A3Stage5gAgentCfg,
    A3Stage5gFixedBallEnvCfg,
    A3Stage5gFixedBallEvalEnvCfg,
    A3Stage5gFixedBallAgentCfg,
    A3Stage5gWideEnvCfg,
    A3Stage5gWideEvalEnvCfg,
    A3Stage5gWideAgentCfg,
    A3Stage5hHitQualityEnvCfg,
    A3Stage5hHitQualityEvalEnvCfg,
    A3Stage5hHitQualityAgentCfg,
    A3Stage5iStableHitQualityEnvCfg,
    A3Stage5iStableHitQualityEvalEnvCfg,
    A3Stage5iStableHitQualityAgentCfg,
    A3Stage4eEnvCfg,
    A3Stage4eEvalEnvCfg,
    A3Stage4eAgentCfg,
    A3Stage4fEnvCfg,
    A3Stage4fEvalEnvCfg,
    A3Stage4fAgentCfg,
    A3Stage4gEnvCfg,
    A3Stage4gEvalEnvCfg,
    A3Stage4gAgentCfg,
    A3Stage4hEnvCfg,
    A3Stage4hEvalEnvCfg,
    A3Stage4hAgentCfg,
)


from legged_lab.utils.task_registry import task_registry
task_registry.register("t1_tt", TTEnv, T1TableTennisEnvCfg(), T1TableTennisAgentCfg()) #TTEnv
task_registry.register("t1_tt_eval", TTEnv, T1TT_EvalEnvCfg(), T1TableTennisAgentCfg()) 
task_registry.register("a3_tt", TTEnv, A3TableTennisEnvCfg(), A3TableTennisAgentCfg())
task_registry.register("a3_tt_eval", TTEnv, A3TT_EvalEnvCfg(), A3TableTennisAgentCfg())
task_registry.register("a3_tt_stable", TTEnv, A3StableReturnEnvCfg(), A3StableReturnAgentCfg())
task_registry.register("a3_tt_stable_eval", TTEnv, A3StableReturnEvalEnvCfg(), A3StableReturnAgentCfg())
task_registry.register("a3_tt_stage4b", TTEnv, A3Stage4bEnvCfg(), A3Stage4bAgentCfg())
task_registry.register("a3_tt_stage4b_eval", TTEnv, A3Stage4bEvalEnvCfg(), A3Stage4bAgentCfg())
task_registry.register("a3_tt_stage4c", TTEnv, A3Stage4cEnvCfg(), A3Stage4cAgentCfg())
task_registry.register("a3_tt_stage4c_eval", TTEnv, A3Stage4cEvalEnvCfg(), A3Stage4cAgentCfg())
task_registry.register("a3_tt_stage4d", TTEnv, A3Stage4dEnvCfg(), A3Stage4dAgentCfg())
task_registry.register("a3_tt_stage4d_eval", TTEnv, A3Stage4dEvalEnvCfg(), A3Stage4dAgentCfg())
task_registry.register("a3_tt_stage5_ready", TTEnv, A3Stage5ReadyEnvCfg(), A3Stage5ReadyAgentCfg())
task_registry.register("a3_tt_stage5_ready_eval", TTEnv, A3Stage5ReadyEvalEnvCfg(), A3Stage5ReadyAgentCfg())
task_registry.register("a3_tt_stage5b", TTEnv, A3Stage5bEnvCfg(), A3Stage5bAgentCfg())
task_registry.register("a3_tt_stage5b_eval", TTEnv, A3Stage5bEvalEnvCfg(), A3Stage5bAgentCfg())
task_registry.register("a3_tt_stage5c", TTEnv, A3Stage5cEnvCfg(), A3Stage5cAgentCfg())
task_registry.register("a3_tt_stage5c_eval", TTEnv, A3Stage5cEvalEnvCfg(), A3Stage5cAgentCfg())
task_registry.register("a3_tt_stage5d", TTEnv, A3Stage5dEnvCfg(), A3Stage5dAgentCfg())
task_registry.register("a3_tt_stage5d_eval", TTEnv, A3Stage5dEvalEnvCfg(), A3Stage5dAgentCfg())
task_registry.register("a3_tt_stage5e", TTEnv, A3Stage5eEnvCfg(), A3Stage5eAgentCfg())
task_registry.register("a3_tt_stage5e_eval", TTEnv, A3Stage5eEvalEnvCfg(), A3Stage5eAgentCfg())
task_registry.register("a3_tt_stage5f", TTEnv, A3Stage5fEnvCfg(), A3Stage5fAgentCfg())
task_registry.register("a3_tt_stage5f_eval", TTEnv, A3Stage5fEvalEnvCfg(), A3Stage5fAgentCfg())
task_registry.register("a3_tt_stage5g", TTEnv, A3Stage5gEnvCfg(), A3Stage5gAgentCfg())
task_registry.register("a3_tt_stage5g_eval", TTEnv, A3Stage5gEvalEnvCfg(), A3Stage5gAgentCfg())
task_registry.register("a3_tt_stage5g_fixedball", TTEnv, A3Stage5gFixedBallEnvCfg(), A3Stage5gFixedBallAgentCfg())
task_registry.register("a3_tt_stage5g_fixedball_eval", TTEnv, A3Stage5gFixedBallEvalEnvCfg(), A3Stage5gFixedBallAgentCfg())
task_registry.register("a3_tt_stage5g_wide", TTEnv, A3Stage5gWideEnvCfg(), A3Stage5gWideAgentCfg())
task_registry.register("a3_tt_stage5g_wide_eval", TTEnv, A3Stage5gWideEvalEnvCfg(), A3Stage5gWideAgentCfg())
task_registry.register("a3_tt_stage5h_hitquality", TTEnv, A3Stage5hHitQualityEnvCfg(), A3Stage5hHitQualityAgentCfg())
task_registry.register("a3_tt_stage5h_hitquality_eval", TTEnv, A3Stage5hHitQualityEvalEnvCfg(), A3Stage5hHitQualityAgentCfg())
task_registry.register("a3_tt_stage5i_stable_hitquality", TTEnv, A3Stage5iStableHitQualityEnvCfg(), A3Stage5iStableHitQualityAgentCfg())
task_registry.register("a3_tt_stage5i_stable_hitquality_eval", TTEnv, A3Stage5iStableHitQualityEvalEnvCfg(), A3Stage5iStableHitQualityAgentCfg())
task_registry.register("a3_tt_stage4e", TTEnv, A3Stage4eEnvCfg(), A3Stage4eAgentCfg())
task_registry.register("a3_tt_stage4e_eval", TTEnv, A3Stage4eEvalEnvCfg(), A3Stage4eAgentCfg())
task_registry.register("a3_tt_stage4f", TTEnv, A3Stage4fEnvCfg(), A3Stage4fAgentCfg())
task_registry.register("a3_tt_stage4f_eval", TTEnv, A3Stage4fEvalEnvCfg(), A3Stage4fAgentCfg())
task_registry.register("a3_tt_stage4g", TTEnv, A3Stage4gEnvCfg(), A3Stage4gAgentCfg())
task_registry.register("a3_tt_stage4g_eval", TTEnv, A3Stage4gEvalEnvCfg(), A3Stage4gAgentCfg())
task_registry.register("a3_tt_stage4h", TTEnv, A3Stage4hEnvCfg(), A3Stage4hAgentCfg())
task_registry.register("a3_tt_stage4h_eval", TTEnv, A3Stage4hEvalEnvCfg(), A3Stage4hAgentCfg())
