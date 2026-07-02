from .bd_optim_env_cfg import bd_optim_env_cfg
from .mdp.runner import bd_optim_ppo_runner_cfg

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-BD-Optim",
  env_cfg=bd_optim_env_cfg(),
  play_env_cfg=bd_optim_env_cfg(play=True),
  rl_cfg=bd_optim_ppo_runner_cfg(),
)
