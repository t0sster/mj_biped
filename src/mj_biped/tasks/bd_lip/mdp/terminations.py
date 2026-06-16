from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


def exceeds_max_velocity(
  env,
  max_velocity: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.norm(asset.data.root_link_lin_vel_b, dim=1) > max_velocity
