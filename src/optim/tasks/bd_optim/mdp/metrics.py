from __future__ import annotations

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


class max_commanded_forward_speed:
  def __init__(self, cfg: MetricsTermCfg, env) -> None:
    asset_cfg = cfg.params["asset_cfg"]
    self._asset_name = asset_cfg.name
    self._max_speed = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

  def __call__(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    del asset_cfg
    asset = env.scene[self._asset_name]
    speed = torch.clamp(asset.data.root_link_lin_vel_b[:, 0], min=0.0)
    self._max_speed = torch.maximum(self._max_speed, speed)
    return self._max_speed

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._max_speed[env_ids] = 0.0
