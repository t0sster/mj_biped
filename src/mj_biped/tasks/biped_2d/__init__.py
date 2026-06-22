from .biped_2d_env_cfg import (
  biped_2d_env_cfg,
)
from mj_biped.tasks.biped_2d.mdp.runner import biped_2d_ppo_runner_cfg

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Biped-2D",
  env_cfg=biped_2d_env_cfg(),
  play_env_cfg=biped_2d_env_cfg(play=True),
  rl_cfg=biped_2d_ppo_runner_cfg(),
)
