"""Reconstruct trajectory plot from a saved log file.

Usage:
    python -m mj_biped.tasks.bd_lip.utils.vis.plot_run <path>.npz
    python -m mj_biped.tasks.bd_lip.utils.vis.plot_run <path>.npz --save out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _dist_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Minimum distance from each point to the nearest segment of the polyline."""
    dists = np.full(len(points), np.inf)
    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 < 1e-12:
            d = np.linalg.norm(points - a, axis=1)
        else:
            t = np.clip(((points - a) @ ab) / ab_len2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(points - proj, axis=1)
        dists = np.minimum(dists, d)
    return dists


def compute_errors(actual_path: np.ndarray, planned_path: np.ndarray) -> dict:
    """Compute tracking errors: distance from each actual point to the planned polyline."""
    if len(actual_path) == 0 or len(planned_path) < 2:
        return {}
    errors = _dist_to_polyline(actual_path, planned_path)
    return {
        "mean_m": float(errors.mean()),
        "max_m": float(errors.max()),
        "errors": errors,
    }


def plot_from_log(log_path: str | Path, save_path: str | Path | None = None) -> None:
    data = np.load(log_path)

    planned_path = data["planned_path"]
    actual_path = data["actual_path"]
    cube_size = data["cube_size"]
    grid = data["grid"]
    x_min = float(data["grid_x_min"])
    y_min = float(data["grid_y_min"])
    cs = float(data["grid_cell_size"])
    x_max = x_min + grid.shape[0] * cs
    y_max = y_min + grid.shape[1] * cs

    if "cube_positions" in data:
        cube_positions = data["cube_positions"]
        if cube_positions.ndim == 1:
            cube_positions = cube_positions.reshape(1, -1)
    else:
        cube_positions = data["cube_pos"].reshape(1, -1)

    metrics = compute_errors(actual_path, planned_path)
    if metrics:
        print(f"Tracking error — mean: {metrics['mean_m']*100:.1f} cm  |  max: {metrics['max_m']*100:.1f} cm")

    _, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(
        grid.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="Greys",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )

    hw = float(cube_size[0])
    for i, pos in enumerate(cube_positions):
        cx, cy = float(pos[0]), float(pos[1])
        rect = plt.Rectangle(
            (cx - hw, cy - hw), 2 * hw, 2 * hw,
            linewidth=1.5, edgecolor="orange", facecolor="orange", alpha=0.7,
            label="cube" if i == 0 else None,
        )
        ax.add_patch(rect)

    ax.plot(planned_path[:, 0], planned_path[:, 1], "--", color="tomato", linewidth=1.5, label="A* path", zorder=2)
    ax.scatter(planned_path[:, 0], planned_path[:, 1], color="tomato", s=30, zorder=4)
    ax.scatter(planned_path[-1, 0], planned_path[-1, 1], color="green", s=80, zorder=5, label="goal")

    if len(actual_path) > 0:
        ax.plot(actual_path[:, 0], actual_path[:, 1], color="steelblue", linewidth=1.5, label="actual", zorder=3)
        ax.plot(actual_path[0, 0], actual_path[0, 1], "o", color="steelblue", markersize=7)

    if metrics:
        title = (
            f"A* Path vs Actual — {Path(log_path).stem}\n"
            f"mean error: {metrics['mean_m']*100:.1f} cm  |  max error: {metrics['max_m']*100:.1f} cm"
        )
    else:
        title = f"A* Path vs Actual Trajectory — {Path(log_path).stem}"

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", help="Path to .npz log file")
    p.add_argument("--save", default=None, help="Save plot to file instead of showing")
    args = p.parse_args()
    plot_from_log(args.log, save_path=args.save)


if __name__ == "__main__":
    main()
