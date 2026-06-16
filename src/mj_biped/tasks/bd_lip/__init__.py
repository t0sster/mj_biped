from mj_biped.tasks.bd_lip.bd_lip_env_cfg import (
  bd_lip_env_cfg,
  bd_lip_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="BD-Lip",
  env_cfg=bd_lip_env_cfg(),
  play_env_cfg=bd_lip_env_cfg(play=True),
  rl_cfg=bd_lip_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="BD-Lip-Play",
  env_cfg=bd_lip_env_cfg(play=True),
  play_env_cfg=bd_lip_env_cfg(play=True),
  rl_cfg=bd_lip_ppo_runner_cfg(),
)
