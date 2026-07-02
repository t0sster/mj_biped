from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from optim.core.config import ACTUATED_JOINT_NAMES, SimulationConfig, PDConfig
from optim.core.metrics import MotorMetricsLogger
from optim.core.model import load_model, new_data
from optim.core.pd import PDController
from optim.tasks.scripted.jump_vertical import JumpVerticalTask
from optim.tasks.scripted.squat_stand import SquatStandTask
from optim.tasks.scripted.stand import STAND_POSES, StandTask

TASK_NAMES = ("stand", "squat_stand", "jump_vertical")


@dataclass(frozen=True)
class OutputPaths:
  summary: Path | None = None
  timeseries: Path | None = None


def main() -> None:
  args = _parse_args()
  task = _build_task(args)
  cfg = SimulationConfig(
    model_path=args.model,
    duration=args.duration if args.duration is not None else _task_duration(task),
    timestep=args.timestep,
    pd=_pd_config(args),
  )

  context = load_model(cfg.model_path, timestep=cfg.timestep)
  data = new_data(context)
  controller = PDController(context, cfg.pd)
  steps = max(1, int(cfg.duration / context.model.opt.timestep))
  logger = MotorMetricsLogger(context)
  output_paths = _output_paths(args)

  def reset_episode() -> MotorMetricsLogger:
    mujoco.mj_resetData(context.model, data)
    data.qpos[:] = task.initial_qpos(context)
    mujoco.mj_forward(context.model, data)
    return MotorMetricsLogger(context)

  logger = reset_episode()
  if args.viewer:
    with mujoco.viewer.launch_passive(context.model, data) as viewer:
      episode_step = 0
      while viewer.is_running():
        step_start = time.time()
        control = _step(context, data, task, controller, logger)
        del control
        episode_step += 1
        if episode_step >= steps:
          _print_summary(logger.summary())
          logger = reset_episode()
          episode_step = 0
        viewer.sync()
        sleep = context.model.opt.timestep - (time.time() - step_start)
        if sleep > 0.0:
          time.sleep(sleep)
  else:
    episode = 0
    while args.loop or episode < args.episodes:
      logger = reset_episode()
      for _ in range(steps):
        _step(context, data, task, controller, logger)
      episode += 1
      if args.loop or args.episodes > 1:
        print(f"episode {episode}")
        _print_summary(logger.summary())

  if output_paths is not None:
    _write_outputs(logger, output_paths)
  if not args.loop and args.episodes == 1 and not args.viewer:
    _print_summary(logger.summary())


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run optim motor capability tasks.")
  parser.add_argument("--task", choices=TASK_NAMES, default="stand")
  parser.add_argument("--pose", choices=tuple(STAND_POSES), default="neutral")
  parser.add_argument(
    "--model",
    type=Path,
    default=SimulationConfig.model_path,
    help="MJCF model path.",
  )
  parser.add_argument("--duration", type=float, default=None)
  parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer.")
  parser.add_argument(
    "--loop",
    action="store_true",
    help="Keep replaying episodes. Viewer mode always loops.",
  )
  parser.add_argument(
    "--episodes",
    type=int,
    default=1,
    help="Number of headless episodes to run.",
  )
  parser.add_argument("--timestep", type=float, default=None)
  parser.add_argument("--kp", type=float, default=None)
  parser.add_argument("--kd", type=float, default=None)
  parser.add_argument(
    "--joint",
    action="append",
    default=[],
    metavar="NAME=RAD",
    help="Override a stand target joint angle. Can be passed multiple times.",
  )
  parser.add_argument("--output", type=Path, default=None, help="Write summary CSV.")
  parser.add_argument(
    "--timeseries-output",
    type=Path,
    default=None,
    help="Write per-step metrics CSV.",
  )
  parser.add_argument(
    "--log",
    action="store_true",
    help="Write summary.csv and timeseries.csv under logs/optim/<task>/<run-time>.",
  )
  return parser.parse_args()


def _build_task(args: argparse.Namespace):
  if args.task == "stand":
    joint_targets = _stand_pose(args.pose) | _parse_joint_targets(args.joint)
    return StandTask(joint_targets=joint_targets)
  if args.joint:
    raise ValueError("--joint overrides are currently only supported for --task stand.")
  if args.task == "squat_stand":
    return SquatStandTask()
  return JumpVerticalTask()


def _task_duration(task) -> float:
  return getattr(task, "duration", SimulationConfig.duration)


def _pd_config(args: argparse.Namespace) -> PDConfig:
  defaults = PDConfig()
  return PDConfig(
    kp=defaults.kp if args.kp is None else args.kp,
    kd=defaults.kd if args.kd is None else args.kd,
  )


def _output_paths(args: argparse.Namespace) -> OutputPaths | None:
  if args.log:
    run_dir = _default_run_dir(args.task)
    return OutputPaths(
      summary=args.output or run_dir / "summary.csv",
      timeseries=args.timeseries_output or run_dir / "timeseries.csv",
    )
  if args.output is not None or args.timeseries_output is not None:
    return OutputPaths(summary=args.output, timeseries=args.timeseries_output)
  return None


def _default_run_dir(task_name: str) -> Path:
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  return Path("logs") / "optim" / task_name / timestamp


def _write_outputs(logger: MotorMetricsLogger, paths: OutputPaths) -> None:
  if paths.summary is not None:
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    logger.write_csv(paths.summary)
    print(f"wrote summary: {paths.summary}")
  if paths.timeseries is not None:
    paths.timeseries.parent.mkdir(parents=True, exist_ok=True)
    logger.write_timeseries_csv(paths.timeseries)
    print(f"wrote timeseries: {paths.timeseries}")


def _step(context, data, task, controller, logger):
  q_des = task.desired_joint_positions(data.time, context)
  control = controller.apply(data, q_des)
  logger.record(context.model, data, control)
  mujoco.mj_step(context.model, data)
  return control


def _stand_pose(name: str) -> dict[str, float]:
  return dict(STAND_POSES[name])


def _parse_joint_targets(items: list[str]) -> dict[str, float]:
  targets: dict[str, float] = {}
  known = set(ACTUATED_JOINT_NAMES)
  for item in items:
    name, separator, value = item.partition("=")
    if not separator:
      raise ValueError(f"Joint override must look like NAME=RAD, got: {item}")
    if name not in known:
      raise ValueError(f"Unknown actuated joint {name!r}. Known: {sorted(known)}")
    targets[name] = float(value)
  return targets


def _print_summary(rows) -> None:
  header = (
    "actuator",
    "max_tau_m",
    "rms_tau_m",
    "max_tau_j",
    "max_w_m",
    "max_power",
    "sat_%",
    "rms_err",
  )
  print(
    f"{header[0]:<18} {header[1]:>10} {header[2]:>10} {header[3]:>10} "
    f"{header[4]:>10} {header[5]:>10} {header[6]:>8} {header[7]:>10}"
  )
  for row in rows:
    print(
      f"{row.actuator:<18} "
      f"{row.max_abs_motor_torque:10.3f} "
      f"{row.rms_motor_torque:10.3f} "
      f"{row.max_abs_joint_torque:10.3f} "
      f"{row.max_abs_motor_speed:10.3f} "
      f"{row.max_abs_power:10.3f} "
      f"{100.0 * row.saturation_fraction:8.2f} "
      f"{row.rms_tracking_error:10.4f}"
    )


if __name__ == "__main__":
  main()
