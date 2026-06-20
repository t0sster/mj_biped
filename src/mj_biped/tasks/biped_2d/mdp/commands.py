from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class UniformGaitCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    frequencies: tuple[float, float] = (0.5, 1.5)
    duty_cycle: tuple[float, float] = (0.5, 0.5)

  ranges: Ranges

  def build(self, env) -> UniformGaitCommand:
    return UniformGaitCommand(self, env)


class UniformGaitCommand(CommandTerm):
  cfg: UniformGaitCommandCfg

  def __init__(self, cfg: UniformGaitCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self._frequency = torch.zeros(self.num_envs, device=self.device)
    self._duty_cycle = torch.zeros(self.num_envs, device=self.device)
    self._phase = torch.zeros(self.num_envs, device=self.device)
    self._command = torch.zeros(self.num_envs, 5, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def compute(self, dt: float) -> None:
    self._phase = torch.remainder(self._phase + dt * self._frequency, 1.0)
    super().compute(dt)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    freq_min, freq_max = self.cfg.ranges.frequencies
    duty_min, duty_max = self.cfg.ranges.duty_cycle
    self._frequency[env_ids] = self._frequency[env_ids].uniform_(freq_min, freq_max)
    self._duty_cycle[env_ids] = self._duty_cycle[env_ids].uniform_(duty_min, duty_max)
    self._phase[env_ids] = self._phase[env_ids].uniform_(0.0, 1.0)

  def _update_command(self) -> None:
    left_phase = self._phase
    right_phase = torch.remainder(self._phase + 0.5, 1.0)
    left_contact = left_phase < self._duty_cycle
    right_contact = right_phase < self._duty_cycle

    self._command[:, 0] = left_phase
    self._command[:, 1] = right_phase
    self._command[:, 2] = left_contact.float()
    self._command[:, 3] = right_contact.float()
    self._command[:, 4] = self._frequency
