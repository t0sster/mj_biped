from __future__ import annotations

import torch


def gait_sin_cos(env, command_name: str) -> torch.Tensor:
  gait = env.command_manager.get_command(command_name)
  phases = gait[:, :2]
  return torch.cat(
    [
      torch.sin(2.0 * torch.pi * phases),
      torch.cos(2.0 * torch.pi * phases),
      gait[:, 2:],
    ],
    dim=1,
  )
