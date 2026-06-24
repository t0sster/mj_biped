from __future__ import annotations

import heapq
import math

import numpy as np


class AStarPlanner:
    def __init__(self, cell_size: float = 0.1, robot_radius: float = 0.15) -> None:
        self.cell_size = cell_size
        self.robot_radius = robot_radius

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        obstacles: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, dict]:
        """Find a collision-free path with A*.

        obstacles: list of (center_x, center_y, half_w, half_h) axis-aligned boxes.
        Returns (path_xy, grid_info):
            path_xy  — (N, 2) array of world XY positions along the path.
            grid_info — dict with grid data for plotting.
        """
        margin = 1.5
        x_min = min(start[0], goal[0]) - margin
        x_max = max(start[0], goal[0]) + margin
        y_min = min(start[1], goal[1]) - margin
        y_max = max(start[1], goal[1]) + margin

        cs = self.cell_size
        nx = int((x_max - x_min) / cs) + 1
        ny = int((y_max - y_min) / cs) + 1

        grid = np.zeros((nx, ny), dtype=bool)

        r = self.robot_radius
        for (cx, cy, hw, hh) in obstacles:
            ix0 = max(0, int((cx - hw - r - x_min) / cs))
            ix1 = min(nx - 1, int((cx + hw + r - x_min) / cs))
            iy0 = max(0, int((cy - hh - r - y_min) / cs))
            iy1 = min(ny - 1, int((cy + hh + r - y_min) / cs))
            grid[ix0 : ix1 + 1, iy0 : iy1 + 1] = True

        def to_cell(wx: float, wy: float) -> tuple[int, int]:
            return int((wx - x_min) / cs), int((wy - y_min) / cs)

        def to_world(ix: int, iy: int) -> tuple[float, float]:
            return x_min + ix * cs, y_min + iy * cs

        s = to_cell(*start)
        g = to_cell(*goal)

        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        COSTS = [1.0, 1.0, 1.0, 1.0, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]

        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_cost: dict[tuple[int, int], float] = {s: 0.0}

        def h(cell: tuple[int, int]) -> float:
            return math.hypot(cell[0] - g[0], cell[1] - g[1])

        found = False
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == g:
                found = True
                break
            for (dx, dy), cost in zip(DIRS, COSTS):
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                    continue
                if grid[nb[0], nb[1]]:
                    continue
                new_g = g_cost[cur] + cost
                if new_g < g_cost.get(nb, float("inf")):
                    g_cost[nb] = new_g
                    came_from[nb] = cur
                    heapq.heappush(open_heap, (new_g + h(nb), nb))

        if not found:
            raise RuntimeError("A* could not find a path to the goal")

        cells: list[tuple[int, int]] = []
        cur = g
        while cur != s:
            cells.append(cur)
            cur = came_from[cur]
        cells.append(s)
        cells.reverse()

        path = np.array([to_world(ix, iy) for (ix, iy) in cells])

        grid_info = {
            "grid": grid,
            "x_min": x_min,
            "y_min": y_min,
            "cell_size": cs,
            "nx": nx,
            "ny": ny,
        }
        return path, grid_info


def simplify_path(path: np.ndarray, min_step: float = 0.4) -> np.ndarray:
    """Reduce path to waypoints at least min_step meters apart."""
    if len(path) == 0:
        return path
    result = [path[0]]
    for pt in path[1:-1]:
        if np.linalg.norm(pt - result[-1]) >= min_step:
            result.append(pt)
    result.append(path[-1])
    return np.array(result)
