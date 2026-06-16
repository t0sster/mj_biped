from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api import math as math_utils

from mj_biped.tasks.bd_lip.mdp.lip import gait_phase_from_command


def step_command_tracking(
  env,
  asset_cfg: SceneEntityCfg,
  command_name: str = "lip_step_command",
  position_sigma: float = 0.05,
  yaw_sigma: float = 0.25,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  root_pos_w = asset.data.root_link_pos_w
  root_quat_w = asset.data.root_link_quat_w

  forward_b = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
  right_forward_w = math_utils.quat_apply(foot_quat_w[:, 0, :], forward_b)
  left_forward_w = math_utils.quat_apply(foot_quat_w[:, 1, :], forward_b)
  foot_yaw_w = torch.stack(
    (
      torch.atan2(right_forward_w[:, 1], right_forward_w[:, 0]),
      torch.atan2(left_forward_w[:, 1], left_forward_w[:, 0]),
    ),
    dim=1,
  )

  command = env.command_manager.get_command(command_name)
  target_xy_w = torch.stack((command[:, 0:2], command[:, 3:5]), dim=1)
  target_yaw_w = torch.stack((command[:, 2], command[:, 5]), dim=1)

  root_quat_rep = root_quat_w.unsqueeze(1).repeat(1, foot_pos_w.shape[1], 1)
  foot_pos_b = math_utils.quat_apply_inverse(
    root_quat_rep,
    foot_pos_w - root_pos_w.unsqueeze(1),
  )

  target_pos_w = torch.zeros(target_xy_w.shape[0], target_xy_w.shape[1], 3, device=env.device)
  target_pos_w[:, :, :2] = target_xy_w
  target_pos_w[:, :, 2] = root_pos_w[:, 2:3]
  target_pos_b = math_utils.quat_apply_inverse(
    root_quat_rep,
    target_pos_w - root_pos_w.unsqueeze(1),
  )

  pos_err = torch.norm(foot_pos_b[:, :, :2] - target_pos_b[:, :, :2], dim=2)
  yaw_err = math_utils.wrap_to_pi(foot_yaw_w - target_yaw_w)
  reward_pos = torch.exp(-torch.square(pos_err) / position_sigma)
  reward_yaw = torch.exp(-torch.square(yaw_err) / yaw_sigma)
  return 0.5 * (reward_pos + reward_yaw).mean(dim=1)


def contact_schedule(
  env,
  sensor_name: str,
  command_name: str = "gait_command",
  threshold: float = 1.0,
  sigma: float = 0.25,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  right_phase, left_phase, duration = gait_phase_from_command(
    env.episode_length_buf, env.step_dt, command
  )
  desired = torch.stack((right_phase < duration, left_phase < duration), dim=1).float()

  if sensor.data.force is not None:
    force_norm = torch.norm(sensor.data.force, dim=-1)
    actual = (force_norm > threshold).float()
  else:
    assert sensor.data.found is not None
    actual = (sensor.data.found > 0).float()

  diff = actual - desired
  return torch.exp(-torch.square(diff) / sigma).mean(dim=1)


def heading_tracking(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  command_name: str = "base_velocity",
  heading_sigma: float = 0.15,
  speed_eps: float = 1.0e-3,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  base_heading = asset.data.heading_w.unsqueeze(1)
  command = env.command_manager.get_command(command_name)
  cmd_vel = command[:, :2]
  cmd_speed = torch.norm(cmd_vel, dim=1, keepdim=True)
  vel_heading = torch.atan2(cmd_vel[:, 1], cmd_vel[:, 0]).unsqueeze(1)
  desired_heading = math_utils.wrap_to_pi(base_heading + vel_heading)
  desired_heading = torch.where(cmd_speed > speed_eps, desired_heading, base_heading)
  err = math_utils.wrap_to_pi(base_heading - desired_heading)
  return torch.exp(-torch.square(err) / heading_sigma).squeeze(1)


def base_height_tracking_l2(
  env,
  command_name: str = "base_height_command",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  target = env.command_manager.get_command(command_name).squeeze(1)
  return torch.square(asset.data.root_link_pos_w[:, 2] - target)


def stand_still(
  env,
  lin_threshold: float = 0.02,
  ang_threshold: float = 0.1,
  command_name: str = "base_velocity",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  lin_commands = command[:, :2]
  ang_commands = command[:, 2]
  reward_lin = torch.sum(
    torch.abs(asset.data.root_link_lin_vel_w[:, :2])
    * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold),
    dim=1,
  )
  reward_ang = torch.abs(asset.data.root_link_ang_vel_w[:, 2]) * (
    torch.abs(ang_commands) < ang_threshold
  )
  return reward_lin + reward_ang


def foot_slip_penalty(
  env,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  contact_threshold: float = 1.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]
  if sensor.data.force is not None:
    in_contact = torch.norm(sensor.data.force, dim=-1) > contact_threshold
  else:
    assert sensor.data.found is not None
    in_contact = sensor.data.found > 0
  foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
  slip_speed = torch.norm(foot_vel_xy, dim=2)
  return torch.mean(slip_speed * in_contact.float(), dim=1)


def lin_vel_z_l2(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
