from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def swing_foot_force_l2(
  env,
  sensor_name: str,
  command_name: str,
  motion_command_name: str,
  force_scale: float = 50.0,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize ground force on a foot during its commanded swing phase."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  desired_contacts = env.command_manager.get_command(command_name)[:, 2:4]
  swing_mask = 1.0 - desired_contacts
  foot_force = torch.linalg.norm(sensor.data.force, dim=-1)
  cost = torch.mean(torch.square(foot_force / force_scale) * swing_mask, dim=1)
  return cost * _moving_command_mask(
    env,
    command_name=motion_command_name,
    threshold=command_threshold,
  )


def stance_foot_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  motion_command_name: str,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize horizontal velocity at a foot site during commanded stance."""
  asset = env.scene[asset_cfg.name]
  desired_contacts = env.command_manager.get_command(command_name)[:, 2:4]
  foot_velocity_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  foot_speed_squared = torch.sum(torch.square(foot_velocity_xy), dim=2)
  cost = torch.mean(foot_speed_squared * desired_contacts, dim=1)
  return cost * _moving_command_mask(
    env,
    command_name=motion_command_name,
    threshold=command_threshold,
  )


def biped_air_time(
  env,
  sensor_name: str,
  command_name: str,
  max_reward_time: float = 0.5,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward sustained single support while a non-zero velocity is commanded."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  assert sensor.data.current_contact_time is not None

  air_time = sensor.data.current_air_time
  contact_time = sensor.data.current_contact_time
  in_contact = contact_time > 0.0
  phase_time = torch.where(in_contact, contact_time, air_time)
  single_support = torch.sum(in_contact, dim=1) == 1
  reward = torch.min(
    torch.where(single_support.unsqueeze(-1), phase_time, 0.0),
    dim=1,
  ).values
  reward = torch.clamp(reward, max=max_reward_time)

  return reward * _moving_command_mask(
    env,
    command_name=command_name,
    threshold=command_threshold,
  )


def excessive_foot_force(
  env,
  sensor_name: str,
  max_force: float = 80.0,
) -> torch.Tensor:
  """Penalize only the part of the foot contact force above a soft limit."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_magnitude = torch.linalg.norm(sensor.data.force, dim=-1)
  return torch.sum((force_magnitude - max_force).clamp(min=0.0), dim=1)


def base_vertical_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize vertical base motion."""
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def _moving_command_mask(
  env,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  magnitude = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (magnitude > threshold).float()
