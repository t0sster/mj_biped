from __future__ import annotations

import torch


class TrajPlanner:
    def __init__(
        self,
        waypoints: torch.Tensor,
        waypoint_threshold: float = 0.15,
        device: str | torch.device = "cpu",
    ) -> None:
        self.waypoints = waypoints.to(device)
        self.waypoint_threshold = waypoint_threshold
        self.device = device
        self.current_idx = 0

    def step(self, robot_pos_xy: torch.Tensor) -> tuple[torch.Tensor, bool]:
        last_idx = len(self.waypoints) - 1
        pos = robot_pos_xy.to(self.device)

        dist = torch.norm(pos - self.waypoints[self.current_idx])
        if dist < self.waypoint_threshold and self.current_idx < last_idx:
            self.current_idx += 1

        target = self.waypoints[self.current_idx]
        dist_to_target = torch.norm(pos - target)
        done = self.current_idx >= last_idx and dist_to_target < self.waypoint_threshold
        return target, done

    def reset(self) -> None:
        self.current_idx = 0
