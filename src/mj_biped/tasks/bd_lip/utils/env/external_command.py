from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    from mj_biped.tasks.bd_lip.utils.planning.planner import TrajPlanner


@dataclass(kw_only=True)
class ExternalVelocityCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)

    def build(self, env: ManagerBasedRlEnv) -> ExternalVelocityCommand:
        return ExternalVelocityCommand(self, env)


class ExternalVelocityCommand(CommandTerm):
    cfg: ExternalVelocityCommandCfg

    def __init__(self, cfg: ExternalVelocityCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._planner: Any = None

    @property
    def command(self) -> torch.Tensor:
        return self._vel_command

    def set(self, vel: torch.Tensor) -> None:
        self._vel_command[:] = vel

    def set_planner(self, planner: TrajPlanner) -> None:
        self._planner = planner

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        if self._planner is None:
            return
        from mj_biped.tasks.bd_lip.utils.vis.traj_vis import draw_trajectory
        draw_trajectory(visualizer, self._planner.waypoints, self._planner.current_idx)
