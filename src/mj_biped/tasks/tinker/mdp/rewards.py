from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi


def gait_contact_match(
  env,
  sensor_name: str,
  command_name: str,
  motion_command_name: str,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  desired_contacts = env.command_manager.get_command(command_name)[:, 2:4]
  in_contact = (sensor.data.found > 0).float()
  match = 1.0 - torch.abs(in_contact - desired_contacts)
  reward = torch.mean(match, dim=1)
  return reward * _moving_command_mask(
    env,
    command_name=motion_command_name,
    threshold=command_threshold,
  )


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


def heading_travel_alignment(
  env,
  asset_cfg: SceneEntityCfg,
  std: float,
  min_speed: float = 0.1,
) -> torch.Tensor:
  """Reward the base facing the direction it is actually moving in.

  Compares world-frame heading (asset.data.heading_w) against the heading of
  the world-frame horizontal velocity vector, so the robot walks face-first
  instead of crab-walking sideways. Gated by min_speed since travel direction
  is undefined/noisy near-zero speed.
  """
  asset = env.scene[asset_cfg.name]
  vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  speed = torch.linalg.norm(vel_xy, dim=-1)
  travel_heading = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])
  heading_error = wrap_to_pi(travel_heading - asset.data.heading_w)
  moving_mask = (speed > min_speed).float()
  return torch.exp(-torch.square(heading_error) / std**2) * moving_mask


def _moving_command_mask(
  env,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  magnitude = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (magnitude > threshold).float()


def base_height(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
  """Reward the base tracking a target height (exp kernel, bounded [0, 1])."""
  asset = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  error = torch.square(height - target_height)
  return torch.exp(-error / std**2)

def step_width(
    env,
    target_width: float,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:

  asset = env.scene[asset_cfg.name]

  left_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
  right_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
  foot_vec_b = quat_apply_inverse(
    asset.data.root_link_quat_w, right_pos_w - left_pos_w
  )
  width = torch.abs(foot_vec_b[:, 1])

  error = torch.square(width - target_width)
  return torch.exp(-error / std**2)


def track_linear_velocity_global(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_w
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)
