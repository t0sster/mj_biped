from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .terminations import bodies_contact_with_terrain


def track_lin_vel_xy_exp(
  env,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  sigma: float = 0.25,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-error / (sigma**2))


def track_ang_vel_z_exp(
  env,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  sigma: float = 0.25,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / (sigma**2))


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
