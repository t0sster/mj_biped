from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


MOTOR_PAIRS = (
  ("ML0_hip_pitch", "MR0_hip_pitch"),
  ("ML1_hip_roll", "MR1_hip_roll"),
  ("ML2_thigh_yaw", "MR2_thigh_yaw"),
  ("ML3_knee_pitch", "MR3_knee_pitch"),
  ("ML4_ankle_pitch", "MR4_ankle_pitch"),
)

JOINT_TO_ACTUATOR = {
  "JL0_hip_pitch": "ML0_hip_pitch",
  "JL1_hip_roll": "ML1_hip_roll",
  "JL2_thigh_yaw": "ML2_thigh_yaw",
  "JL3_knee_pitch": "ML3_knee_pitch",
  "JL4_ankle_pitch": "ML4_ankle_pitch",
  "JR0_hip_pitch": "MR0_hip_pitch",
  "JR1_hip_roll": "MR1_hip_roll",
  "JR2_thigh_yaw": "MR2_thigh_yaw",
  "JR3_knee_pitch": "MR3_knee_pitch",
  "JR4_ankle_pitch": "MR4_ankle_pitch",
}

PAIR_PLOT_VALUES = (
  "tau_joint",
  "tau_motor",
  "tau_motor_cmd",
  "motor_speed",
  "joint_velocity",
  "power",
  "q",
  "q_des",
  "q_error",
  "raw_action",
)


@dataclass(frozen=True)
class OutputPaths:
  plots_dir: Path
  timeseries: Path
  summary: Path


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  checkpoint_path = _resolve_checkpoint_path(args.checkpoint, args.full_path)
  output = _output_paths(checkpoint_path)
  output.plots_dir.mkdir(parents=True, exist_ok=True)

  rows = _run_episode(args, checkpoint_path)
  _write_timeseries(output.timeseries, rows)
  _write_summary(output.summary, rows)
  _write_plots(output.plots_dir, output.timeseries, rows)

  print(f"checkpoint: {checkpoint_path}")
  print(f"wrote timeseries: {output.timeseries}")
  print(f"wrote summary: {output.summary}")
  print(f"wrote plots: {output.plots_dir}")
  return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Play one BD-Optim checkpoint with fixed velocity/gait commands and save "
      "timeseries CSV plus all diagnostic plots."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "checkpoint",
    help=(
      "Checkpoint path under logs/rsl_rl, or a run directory. If a directory is "
      "given, the latest model_*.pt inside it is used."
    ),
  )
  parser.add_argument(
    "--full-path",
    "--full_path",
    dest="full_path",
    action="store_true",
    help="Treat checkpoint as an exact filesystem path.",
  )
  parser.add_argument("--duration", type=float, default=10.0)
  parser.add_argument("--device", default=None, help="cpu, cuda:0, etc.")
  parser.add_argument("--lin-vel-x", type=float, default=0.2)
  parser.add_argument("--lin-vel-y", type=float, default=0.0)
  parser.add_argument("--yaw-rate", type=float, default=0.0)
  parser.add_argument("--gait-frequency", type=float, default=0.75)
  parser.add_argument("--duty-cycle", type=float, default=0.6)
  parser.add_argument("--gait-phase", type=float, default=0.0)
  return parser.parse_args(argv)


def _run_episode(args: argparse.Namespace, checkpoint_path: Path) -> list[dict]:
  import optim.tasks  # noqa: F401

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  task_id = "Mjlab-BD-Optim"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    obs, _ = wrapped_env.reset()
    _set_fixed_commands(env, args, reset_phase=True)
    obs = wrapped_env.get_observations()

    rows: list[dict] = []
    steps = max(1, int(args.duration / env.step_dt))
    for _ in range(steps):
      with torch.inference_mode():
        action = policy(obs)
      obs, _, dones, _ = wrapped_env.step(action)
      _set_fixed_commands(env, args, reset_phase=False)
      rows.extend(_sample_metrics(env))
      if torch.any(dones):
        _set_fixed_commands(env, args, reset_phase=True)
        obs = wrapped_env.get_observations()
    return rows
  finally:
    env.close()


def _set_fixed_commands(env, args: argparse.Namespace, *, reset_phase: bool) -> None:
  velocity = env.command_manager.get_term("velocity")
  command = velocity.command
  command[:, 0] = args.lin_vel_x
  command[:, 1] = args.lin_vel_y
  command[:, 2] = args.yaw_rate
  if hasattr(velocity, "vel_command_w"):
    velocity.vel_command_w[:] = command
  for attr in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"):
    if hasattr(velocity, attr):
      getattr(velocity, attr)[:] = False
  velocity.time_left[:] = 1.0e6

  gait = env.command_manager.get_term("gait")
  if hasattr(gait, "_frequency"):
    gait._frequency[:] = args.gait_frequency
  if hasattr(gait, "_duty_cycle"):
    gait._duty_cycle[:] = args.duty_cycle
  if reset_phase and hasattr(gait, "_phase"):
    gait._phase[:] = args.gait_phase
  if hasattr(gait, "_update_command"):
    gait._update_command()
  gait.time_left[:] = 1.0e6


def _sample_metrics(env) -> list[dict]:
  action_term = env.action_manager.get_term("joint_pos")
  asset = env.scene[action_term.cfg.entity_name]

  target_ids = action_term.target_ids
  target_names = action_term.target_names
  q = asset.data.joint_pos[:, target_ids][0]
  qd = asset.data.joint_vel[:, target_ids][0]
  q_des = (
    action_term._processed_actions
    - asset.data.encoder_bias[:, target_ids]
  )[0]
  stiffness = action_term._stiffness[0]
  damping = action_term._damping[0]
  gear = action_term._gear[0]
  ctrl_min = action_term._ctrl_min[0]
  ctrl_max = action_term._ctrl_max[0]

  tau_joint_cmd = stiffness * (q_des - q) - damping * qd
  tau_motor_cmd = tau_joint_cmd / gear
  tau_motor = torch.clamp(tau_motor_cmd, min=ctrl_min, max=ctrl_max)
  tau_joint = tau_motor * gear
  motor_speed = gear * qd
  power = tau_joint * qd
  saturated = torch.abs(tau_motor_cmd) >= 0.999 * torch.maximum(
    torch.abs(ctrl_min),
    torch.abs(ctrl_max),
  )

  sensor = env.scene["feet_ground_contact"]
  force_w = sensor.data.force[0]
  left_foot_fz = abs(float(force_w[0, 2].detach().cpu()))
  right_foot_fz = abs(float(force_w[1, 2].detach().cpu()))
  time_s = float(env.common_step_counter * env.step_dt)

  rows = []
  raw_action = action_term.raw_action[0]
  for index, joint_name in enumerate(target_names):
    actuator = JOINT_TO_ACTUATOR.get(joint_name, joint_name)
    rows.append({
      "time": time_s,
      "actuator": actuator,
      "joint": joint_name,
      "raw_action": float(raw_action[index].detach().cpu()),
      "tau_motor_cmd": float(tau_motor_cmd[index].detach().cpu()),
      "tau_motor": float(tau_motor[index].detach().cpu()),
      "tau_joint": float(tau_joint[index].detach().cpu()),
      "motor_speed": float(motor_speed[index].detach().cpu()),
      "joint_velocity": float(qd[index].detach().cpu()),
      "power": float(power[index].detach().cpu()),
      "q": float(q[index].detach().cpu()),
      "q_des": float(q_des[index].detach().cpu()),
      "q_error": float((q_des[index] - q[index]).detach().cpu()),
      "left_foot_fz": left_foot_fz,
      "right_foot_fz": right_foot_fz,
      "saturated": bool(saturated[index].detach().cpu()),
    })
  return rows


def _write_timeseries(path: Path, rows: list[dict]) -> None:
  fieldnames = (
    "time",
    "actuator",
    "joint",
    "raw_action",
    "tau_motor_cmd",
    "tau_motor",
    "tau_joint",
    "motor_speed",
    "joint_velocity",
    "power",
    "q",
    "q_des",
    "q_error",
    "left_foot_fz",
    "right_foot_fz",
    "saturated",
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict]) -> None:
  grouped: dict[str, list[dict]] = {}
  for row in rows:
    grouped.setdefault(row["actuator"], []).append(row)

  fieldnames = (
    "actuator",
    "joint",
    "max_abs_motor_torque",
    "rms_motor_torque",
    "max_abs_joint_torque",
    "rms_joint_torque",
    "max_abs_motor_speed",
    "max_abs_power",
    "saturation_fraction",
    "rms_tracking_error",
  )
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for actuator, actuator_rows in grouped.items():
      writer.writerow({
        "actuator": actuator,
        "joint": actuator_rows[0]["joint"],
        "max_abs_motor_torque": _max_abs(actuator_rows, "tau_motor"),
        "rms_motor_torque": _rms(actuator_rows, "tau_motor"),
        "max_abs_joint_torque": _max_abs(actuator_rows, "tau_joint"),
        "rms_joint_torque": _rms(actuator_rows, "tau_joint"),
        "max_abs_motor_speed": _max_abs(actuator_rows, "motor_speed"),
        "max_abs_power": _max_abs(actuator_rows, "power"),
        "saturation_fraction": sum(r["saturated"] for r in actuator_rows)
        / max(1, len(actuator_rows)),
        "rms_tracking_error": _rms(actuator_rows, "q_error"),
      })


def _write_plots(plots_dir: Path, source: Path, rows: list[dict]) -> None:
  grouped = _group_rows(rows)
  for value in PAIR_PLOT_VALUES:
    _plot_pairs(grouped, value, source, plots_dir / f"{value}.png")
  _plot_contact_force(grouped, source, plots_dir / "contact_fz.png")


def _group_rows(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
  grouped: dict[str, dict[str, list[float]]] = {}
  for row in rows:
    values = grouped.setdefault(row["actuator"], {})
    for key, value in row.items():
      if key in {"actuator", "joint", "saturated"}:
        continue
      values.setdefault(key, []).append(float(value))
  return grouped


def _plot_pairs(
  samples: dict[str, dict[str, list[float]]],
  value: str,
  source: Path,
  save_path: Path,
) -> None:
  fig, axes = plt.subplots(len(MOTOR_PAIRS), 1, sharex=True, figsize=(12, 10))
  fig.suptitle(f"{value} from {source}")

  for axis, pair in zip(axes, MOTOR_PAIRS, strict=True):
    for actuator in pair:
      actuator_samples = samples.get(actuator)
      if not actuator_samples:
        continue
      axis.plot(
        actuator_samples["time"],
        actuator_samples[value],
        label=actuator,
      )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axis.set_ylabel(value)
    axis.legend(loc="upper right")
    axis.grid(True, alpha=0.25)

  axes[-1].set_xlabel("time, s")
  fig.tight_layout()
  fig.savefig(save_path, dpi=160)
  plt.close(fig)


def _plot_contact_force(
  samples: dict[str, dict[str, list[float]]],
  source: Path,
  save_path: Path,
) -> None:
  first_actuator = next(iter(samples.values()), None)
  if first_actuator is None:
    raise ValueError(f"No samples found in {source}")

  fig, axis = plt.subplots(1, 1, figsize=(12, 5))
  fig.suptitle(f"foot contact force z from {source}")
  axis.plot(first_actuator["time"], first_actuator["left_foot_fz"], label="left")
  axis.plot(first_actuator["time"], first_actuator["right_foot_fz"], label="right")
  axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
  axis.set_xlabel("time, s")
  axis.set_ylabel("contact force z, N")
  axis.legend(loc="upper right")
  axis.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(save_path, dpi=160)
  plt.close(fig)


def _resolve_checkpoint_path(raw_path: str, full_path: bool) -> Path:
  candidate = Path(raw_path).expanduser()
  if not full_path:
    candidate = Path("logs/rsl_rl") / candidate

  if candidate.is_file():
    return candidate.resolve()
  if candidate.is_dir():
    checkpoints = sorted(candidate.rglob("model_*.pt"), key=_checkpoint_sort_key)
    if checkpoints:
      return checkpoints[-1].resolve()
  raise FileNotFoundError(f"Checkpoint path not found: {candidate}")


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
  try:
    iteration = int(path.stem.split("_")[-1])
  except ValueError:
    iteration = -1
  return (iteration, path.stat().st_mtime, path.name)


def _output_paths(checkpoint_path: Path) -> OutputPaths:
  plots_dir = checkpoint_path.parent / "plots"
  stem = checkpoint_path.stem
  return OutputPaths(
    plots_dir=plots_dir,
    timeseries=plots_dir / f"{stem}_timeseries.csv",
    summary=plots_dir / f"{stem}_summary.csv",
  )


def _max_abs(rows: list[dict], key: str) -> float:
  return max(abs(float(row[key])) for row in rows)


def _rms(rows: list[dict], key: str) -> float:
  values = torch.tensor([float(row[key]) for row in rows])
  return float(torch.sqrt(torch.mean(values**2)))


if __name__ == "__main__":
  raise SystemExit(main())
