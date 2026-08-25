from mjlab.rl import RslRlModelCfg
import torch

from .tinker_env_cfg import tinker_env_cfg
from .mdp.runner import tinker_ppo_runner_cfg

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Tinker",
  env_cfg=tinker_env_cfg(),
  play_env_cfg=tinker_env_cfg(play=True),
  rl_cfg=tinker_ppo_runner_cfg(),
)
