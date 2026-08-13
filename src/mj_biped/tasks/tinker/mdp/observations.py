from __future__ import annotations

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

import torch


def gait_phase_observation(env, command_name: str) -> torch.Tensor:
  """Encode gait phases continuously and append desired contacts/frequency."""
  gait_command = env.command_manager.get_command(command_name)
  phases = gait_command[:, :2]
  phase_angle = 2.0 * torch.pi * phases
  return torch.cat(
    (
      torch.sin(phase_angle),
      torch.cos(phase_angle),
      gait_command[:, 2:],
    ),
    dim=1,
  )

def current_base_height(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2:3]
  return height


def builtin_sensor_data(env, sensor_name: str) -> torch.Tensor:
  """Raw reading from a BuiltinSensorCfg term (e.g. imu gyro/accelerometer)."""
  return env.scene[sensor_name].data


def imu_rpy(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Roll/pitch/yaw of the imu site, in the site's own (chip) local frame.

  Matches the convention a real IMU's onboard fusion (accel+gyro+magnetometer)
  would report: roll about the site's local X, pitch about local Y, yaw about
  local Z. The site's quat in the XML already encodes the chip's physical
  mounting, so this lines up with the real sensor's raw axis convention.
  """
  asset = env.scene[asset_cfg.name]
  site_quat = asset.data.site_quat_w[:, asset_cfg.site_ids[0]]
  roll, pitch, yaw = euler_xyz_from_quat(site_quat)
  return torch.stack((roll, pitch, yaw), dim=-1)
