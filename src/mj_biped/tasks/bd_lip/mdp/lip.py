from __future__ import annotations

import torch


def gait_phase_from_command(
  episode_length_buf: torch.Tensor,
  step_dt: float,
  gait_command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute right/left gait phases from [frequency, phase_offset, duration]."""
  freq = gait_command[:, 0].clamp(min=1.0e-3)
  offset = gait_command[:, 1]
  duration = gait_command[:, 2].clamp(0.05, 0.95)

  phase = torch.remainder(episode_length_buf * step_dt * freq, 1.0)
  right_phase = phase
  left_phase = torch.remainder(phase + offset, 1.0)
  return right_phase, left_phase, duration


def compute_xcom_step_targets_b(
  root_pos_b: torch.Tensor,
  root_lin_vel_b: torch.Tensor,
  support_foot_pos_b: torch.Tensor,
  cmd_vel_xy_b: torch.Tensor,
  heading_b: torch.Tensor,
  step_time: torch.Tensor,
  step_width: torch.Tensor,
  step_length: torch.Tensor,
  left_swing: torch.Tensor | None = None,
  gravity: float = 9.81,
) -> torch.Tensor:
  """Compute an XCoM step target in the robot yaw-base frame.

  The formula is intentionally the same as the IsaacLab planner, but the frame
  naming is explicit: all XY quantities are in the yaw-aligned base frame.
  """
  z = root_pos_b[:, 2:3].clamp(min=0.05)
  omega = torch.sqrt(gravity / z)

  x0 = root_pos_b[:, 0:1] - support_foot_pos_b[:, 0:1]
  y0 = root_pos_b[:, 1:2] - support_foot_pos_b[:, 1:2]
  vx0 = root_lin_vel_b[:, 0:1]
  vy0 = root_lin_vel_b[:, 1:2]

  wt = step_time * omega
  x_f = x0 * torch.cosh(wt) + vx0 * torch.sinh(wt) / omega
  vx_f = x0 * omega * torch.sinh(wt) + vx0 * torch.cosh(wt)
  y_f = y0 * torch.cosh(wt) + vy0 * torch.sinh(wt) / omega
  vy_f = y0 * omega * torch.sinh(wt) + vy0 * torch.cosh(wt)

  x_f_b = x_f + support_foot_pos_b[:, 0:1]
  y_f_b = y_f + support_foot_pos_b[:, 1:2]

  eicp_x = x_f_b + vx_f / omega
  eicp_y = y_f_b + vy_f / omega

  offset_x = -step_length / (torch.exp(wt) - 1.0)
  lateral_offset = step_width / (torch.exp(wt) + 1.0)
  if left_swing is not None:
    offset_y = torch.where(left_swing.view(-1, 1), lateral_offset, -lateral_offset)
  else:
    offset_y = -lateral_offset

  target_b = torch.zeros(root_pos_b.shape[0], 3, device=root_pos_b.device)
  target_b[:, 0] = (eicp_x + offset_x).squeeze(1)
  target_b[:, 1] = (eicp_y + offset_y).squeeze(1)
  target_b[:, 2] = heading_b.squeeze(1) if heading_b.dim() > 1 else heading_b
  return target_b
