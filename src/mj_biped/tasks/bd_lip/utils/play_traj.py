"""Play script: trajectory following with A* path planning.

Usage:
    python -m mj_biped.tasks.bd_lip.utils.play_traj \\
        --checkpoint logs/rsl_rl/bd_lip/<run>/model_5000.pt [--two-cubes] [--viser]
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mj_biped.tasks.bd_lip.bd_lip_env_cfg import bd_lip_ppo_runner_cfg
from mj_biped.tasks.bd_lip.utils.control.controllers import ContinuousController
from mj_biped.tasks.bd_lip.utils.env.external_command import ExternalVelocityCommand
from mj_biped.tasks.bd_lip.utils.env.play_cfg import CUBE_SIZE, make_traj_play_env_cfg
from mj_biped.tasks.bd_lip.utils.log.script.logger import RunLogger
from mj_biped.tasks.bd_lip.utils.planning.astar import AStarPlanner, simplify_path
from mj_biped.tasks.bd_lip.utils.planning.planner import TrajPlanner


class TrajPolicy:
    _FALL_HEIGHT = 0.15
    _PRINT_EVERY = 50

    def __init__(
        self,
        base_policy,
        vel_term: ExternalVelocityCommand,
        planner: TrajPlanner,
        controller: ContinuousController,
        env: RslRlVecEnvWrapper,
        logger: RunLogger | None = None,
    ) -> None:
        self.base_policy = base_policy
        self.vel_term = vel_term
        self.planner = planner
        self.controller = controller
        self.env = env
        self.logger = logger
        self._done = False
        self._pos_history: list[torch.Tensor] = []
        self._step = 0

    def reset(self) -> None:
        self._done = False
        self._step = 0
        self._pos_history.clear()
        self.planner.reset()
        self.controller.reset()
        if self.logger:
            self.logger.clear()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.env.unwrapped
        robot = raw.scene["robot"]

        robot_pos_xy = robot.data.root_link_pos_w[0, :2].cpu()
        robot_heading = robot.data.heading_w[0].cpu()
        root_z = robot.data.root_link_pos_w[0, 2].item()

        self._pos_history.append(robot_pos_xy.clone())
        if self.logger:
            self.logger.record(robot_pos_xy)
        self._step += 1

        if root_z < self._FALL_HEIGHT:
            print(f"\n[step {self._step:6d}] FALL (z={root_z:.3f}m) — respawning")
            self.vel_term.set(torch.zeros(raw.num_envs, 3, device=raw.device))
            obs, _ = self.env.reset()
            self.reset()
            return self.base_policy(obs)

        target_xy, self._done = self.planner.step(robot_pos_xy)

        if self._done:
            vel_cmd = torch.zeros(3)
        else:
            vel_cmd = self.controller.compute(
                robot_pos_xy, robot_heading, target_xy,
                wp_idx=self.planner.current_idx,
            )

        self.vel_term.set(vel_cmd.to(raw.device).unsqueeze(0).expand(raw.num_envs, -1))

        if self._step % self._PRINT_EVERY == 0:
            wp_idx = self.planner.current_idx
            n_wp = len(self.planner.waypoints)
            vx, vy, wz = vel_cmd.tolist()
            px, py = robot_pos_xy.tolist()
            status = "DONE" if self._done else f"wp {wp_idx}/{n_wp - 1} [{self.controller.state_name}]"
            print(
                f"[step {self._step:6d}] pos=({px:+.2f},{py:+.2f})  "
                f"cmd: vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.3f}  {status}"
            )

        return self.base_policy(obs)

    def plot(
        self,
        grid_info: dict | None = None,
        cube_positions: list[tuple[float, float, float]] | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if not self._pos_history:
            return

        actual = torch.stack(self._pos_history).numpy()
        planned = self.planner.waypoints.numpy()

        _, ax = plt.subplots(figsize=(12, 8))

        if grid_info is not None:
            g = grid_info["grid"]
            x_min = grid_info["x_min"]
            y_min = grid_info["y_min"]
            cs = grid_info["cell_size"]
            x_max = x_min + grid_info["nx"] * cs
            y_max = y_min + grid_info["ny"] * cs
            ax.imshow(
                g.T,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                cmap="Greys",
                alpha=0.4,
                vmin=0,
                vmax=1,
            )

        if cube_positions is not None:
            hw = CUBE_SIZE[0]
            for i, pos in enumerate(cube_positions):
                cx, cy = pos[0], pos[1]
                rect = plt.Rectangle(
                    (cx - hw, cy - hw), 2 * hw, 2 * hw,
                    linewidth=1.5, edgecolor="orange", facecolor="orange", alpha=0.7,
                    label="cube" if i == 0 else None,
                )
                ax.add_patch(rect)

        ax.plot(planned[:, 0], planned[:, 1], "--", color="tomato", linewidth=1.5, label="A* path", zorder=2)
        ax.scatter(planned[:, 0], planned[:, 1], color="tomato", s=30, zorder=4)
        ax.scatter(planned[-1, 0], planned[-1, 1], color="green", s=80, zorder=5, label="goal")

        ax.plot(actual[:, 0], actual[:, 1], color="steelblue", linewidth=1.5, label="actual", zorder=3)
        ax.plot(actual[0, 0], actual[0, 1], "o", color="steelblue", markersize=7)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Desired trajectory and real")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def _build_scenario(two_cubes: bool) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float, float]],
    tuple[float, float],
]:
    """Return (cube_positions, astar_obstacles, goal)."""
    if two_cubes:
        x1 = random.uniform(1.0, 1.8)
        x2 = x1 + random.uniform(1.0, 1.5)
        cube_positions = [
            (x1, 0.0, 0.15),
            (x2, 0.0, 0.15),
        ]
        goal = (x2 + 1.5, 0.0)
        print(f"Two cubes: x1={x1:.2f}, x2={x2:.2f}")
    else:
        cube_x = random.uniform(1.0, 3.0)
        cube_positions = [(cube_x, 0.0, 0.15)]
        goal = (cube_x + 1.5, 0.0)
        print(f"Cube at x={cube_x:.2f}")

    obstacles = [
        (p[0], p[1], CUBE_SIZE[0], CUBE_SIZE[1])
        for p in cube_positions
    ]
    return cube_positions, obstacles, goal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--two-cubes", action="store_true", help="Spawn two cubes and navigate between them")
    p.add_argument("--viser", action="store_true", help="Force Viser web viewer (localhost)")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    cube_positions, obstacles, goal = _build_scenario(args.two_cubes)

    env_cfg = make_traj_play_env_cfg(num_envs=1, cube_positions=cube_positions)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env)

    runner_cfg = bd_lip_ppo_runner_cfg()
    runner = MjlabOnPolicyRunner(env, asdict(runner_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    base_policy = runner.get_inference_policy(device=device)

    robot_init_xy = raw_env.scene["robot"].data.root_link_pos_w[0, :2].cpu().numpy()
    start = (float(robot_init_xy[0]), float(robot_init_xy[1]))

    astar = AStarPlanner(cell_size=0.1, robot_radius=0.35)
    raw_path, grid_info = astar.plan(start, goal, obstacles)
    waypoints_np = simplify_path(raw_path, min_step=1.0)
    print(f"A* path: {len(raw_path)} cells → {len(waypoints_np)} waypoints")

    waypoints = torch.tensor(waypoints_np, dtype=torch.float32)
    planner = TrajPlanner(waypoints=waypoints, waypoint_threshold=0.35, device="cpu")

    controller = ContinuousController(kang=5.0, vmax=0.15, wzmax=0.5)

    logger = RunLogger()
    vel_term: ExternalVelocityCommand = raw_env.command_manager.get_term("base_velocity")
    vel_term.set_planner(planner)
    policy = TrajPolicy(base_policy, vel_term, planner, controller, env, logger=logger)

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if args.viser or not has_display:
        ViserPlayViewer(env, policy).run()
    else:
        NativeMujocoViewer(env, policy).run()

    env.close()

    log_path = logger.save(waypoints_np, cube_positions, CUBE_SIZE, grid_info)
    print(f"Log saved: {log_path}")

    policy.plot(grid_info=grid_info, cube_positions=cube_positions)


if __name__ == "__main__":
    main()
