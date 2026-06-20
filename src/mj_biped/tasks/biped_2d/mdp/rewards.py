from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import root_joint_ids
from .terminations import bodies_contact_with_terrain


def forward_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  root_x_id, _, _ = root_joint_ids(env, asset_cfg)
  return torch.clamp(asset.data.joint_vel[:, root_x_id], min=0.0)


def contact_schedule(
  env,
  asset_cfg: SceneEntityCfg,
  terrain_cfg: SceneEntityCfg,
  command_name: str,
  sigma: float = 0.5,
) -> torch.Tensor:
  contacts = bodies_contact_with_terrain(
    env,
    asset_cfg=asset_cfg,
    terrain_cfg=terrain_cfg,
  ).float()
  desired_contacts = env.command_manager.get_command(command_name)[:, 2:4]
  error = torch.mean(torch.square(contacts - desired_contacts), dim=1)
  return torch.exp(-error / (sigma**2))
