"""Reward terms for the biped_2d locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import is_alive, joint_torques_l2, joint_vel_l2
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("biped_2d")
_DEFAULT_ACTUATOR_CFG = SceneEntityCfg("biped_2d")


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  return is_alive(env)


def forward_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  return torch.clamp(asset.data.root_link_lin_vel_w[:, 0], min=0.0)


def upright(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sigma: float = 0.35,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  gravity_xy_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-0.5 * gravity_xy_sq / (sigma**2))


def lateral_centering(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sigma: float = 0.15,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  lateral_offset = torch.abs(asset.data.root_link_pos_w[:, 1])
  return torch.exp(-0.5 * torch.square(lateral_offset / sigma))


def joint_torque(
  env: ManagerBasedRlEnv,
  actuator_cfg: SceneEntityCfg = _DEFAULT_ACTUATOR_CFG,
  sigma: float = 40.0,
) -> torch.Tensor:
  return joint_torques_l2(env, asset_cfg=actuator_cfg) / (sigma**2)


def joint_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sigma: float = 10.0,
) -> torch.Tensor:
  return joint_vel_l2(env, asset_cfg=asset_cfg) / (sigma**2)
