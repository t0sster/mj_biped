from __future__ import annotations

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
