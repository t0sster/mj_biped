from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def root_joint_ids(env, asset_cfg: SceneEntityCfg) -> tuple[int, int, int]:
  asset = env.scene[asset_cfg.name]
  joint_names = list(asset.joint_names)
  return (
    joint_names.index("root_x"),
    joint_names.index("root_z"),
    joint_names.index("root_pitch"),
  )


def base_lin_vel_2d(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  root_x_id, root_z_id, _ = root_joint_ids(env, asset_cfg)
  vel = torch.zeros(env.num_envs, 3, device=env.device)
  vel[:, 0] = asset.data.joint_vel[:, root_x_id]
  vel[:, 2] = asset.data.joint_vel[:, root_z_id]
  return vel


def base_ang_vel_2d(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  _, _, root_pitch_id = root_joint_ids(env, asset_cfg)
  vel = torch.zeros(env.num_envs, 3, device=env.device)
  vel[:, 1] = asset.data.joint_vel[:, root_pitch_id]
  return vel
