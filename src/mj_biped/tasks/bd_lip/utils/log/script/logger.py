from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_DEFAULT_LOG_DIR = Path(__file__).parent.parent


class RunLogger:
    def __init__(self, log_dir: str | Path = _DEFAULT_LOG_DIR) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._pos_history: list[np.ndarray] = []

    def record(self, robot_pos_xy: torch.Tensor) -> None:
        self._pos_history.append(robot_pos_xy.cpu().numpy().copy())

    def clear(self) -> None:
        self._pos_history.clear()

    def save(
        self,
        planned_path: np.ndarray,
        cube_positions: list[tuple[float, float, float]],
        cube_size: tuple[float, float, float],
        grid_info: dict,
        run_name: str | None = None,
    ) -> Path:
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        actual = np.array(self._pos_history) if self._pos_history else np.zeros((0, 2))

        out_path = self.log_dir / f"{run_name}.npz"
        np.savez_compressed(
            out_path,
            planned_path=planned_path,
            actual_path=actual,
            cube_positions=np.array(cube_positions, dtype=np.float64),
            cube_size=np.array(cube_size, dtype=np.float64),
            grid=grid_info["grid"],
            grid_x_min=np.float64(grid_info["x_min"]),
            grid_y_min=np.float64(grid_info["y_min"]),
            grid_cell_size=np.float64(grid_info["cell_size"]),
        )
        return out_path
