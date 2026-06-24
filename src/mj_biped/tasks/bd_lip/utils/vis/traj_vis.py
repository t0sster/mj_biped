from __future__ import annotations

import numpy as np
import torch

from mjlab.viewer.debug_visualizer import DebugVisualizer

_TRAJ_Z = 0.04

_COLOR_PASSED   = (0.45, 0.45, 0.45, 0.40)
_COLOR_CURRENT  = (1.00, 0.80, 0.00, 1.00)
_COLOR_FUTURE   = (0.20, 0.65, 0.95, 0.75)
_COLOR_STOP     = (0.20, 0.90, 0.30, 0.90)
_COLOR_SEG_DONE = (0.40, 0.40, 0.40, 0.30)
_COLOR_SEG_TODO = (0.25, 0.65, 0.95, 0.55)

_WP_RADIUS   = 0.055
_STOP_RADIUS = 0.09
_SEG_RADIUS  = 0.018


def draw_trajectory(
    visualizer: DebugVisualizer,
    waypoints: torch.Tensor,
    current_idx: int,
    z: float = _TRAJ_Z,
) -> None:
    pts = waypoints.cpu().numpy()
    n = len(pts)
    last = n - 1

    for i in range(n - 1):
        p0 = np.array([pts[i, 0], pts[i, 1], z])
        p1 = np.array([pts[i + 1, 0], pts[i + 1, 1], z])
        color = _COLOR_SEG_DONE if i < current_idx else _COLOR_SEG_TODO
        visualizer.add_cylinder(p0, p1, radius=_SEG_RADIUS, color=color, label=f"ts_{i}")

    for i in range(last):
        pos = np.array([pts[i, 0], pts[i, 1], z])
        if i < current_idx:
            color = _COLOR_PASSED
        elif i == current_idx:
            color = _COLOR_CURRENT
        else:
            color = _COLOR_FUTURE
        visualizer.add_sphere(pos, radius=_WP_RADIUS, color=color, label=f"tw_{i}")

    stop = np.array([pts[last, 0], pts[last, 1], z])
    stop_color = _COLOR_CURRENT if current_idx >= last else _COLOR_STOP
    visualizer.add_sphere(stop, radius=_STOP_RADIUS, color=stop_color, label="tw_stop")
