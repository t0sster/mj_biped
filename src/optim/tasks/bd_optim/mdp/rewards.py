from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def gait_swing_foot_force(
  env,
  sensor_name: str,
  command_name: str,
  force_scale: float = 50.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  desired_contacts = env.command_manager.get_command(command_name)[:, 2:4]
  swing_mask = 1.0 - desired_contacts
  foot_force = torch.linalg.norm(sensor.data.force, dim=-1)
  return torch.mean(torch.square(foot_force / force_scale) * swing_mask, dim=1)


def gait_stance_foot_velocity(
  env,
  asset_cfg: SceneEntityCfg,
  command_name: str,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  desired_contacts = env.command_manager.get_command(command_name)[:, 2:4]
  foot_vel_xy = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :2]
  foot_speed_sq = torch.square(foot_vel_xy).sum(dim=2)
  return torch.mean(foot_speed_sq * desired_contacts, dim=1)


def feet_air_time_positive_biped(
  env,
  sensor_name: str,
  command_name: str,
  threshold: float = 0.5,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  assert sensor.data.current_contact_time is not None
  air_time = sensor.data.current_air_time
  contact_time = sensor.data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0),
    dim=1,
  )[0]
  reward = torch.clamp(reward, max=threshold)

  command = env.command_manager.get_command(command_name)
  moving = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return reward * (moving > command_threshold).float()


def feet_contact_forces(
  env,
  sensor_name: str,
  max_force: float = 80.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_norm = torch.linalg.norm(sensor.data.force, dim=-1)
  return torch.sum((force_norm - max_force).clamp(min=0.0), dim=1)


def base_lin_vel_z_l2(
  env,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])
