from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MOTOR_PAIRS = (
  ("ML0_hip_pitch", "MR0_hip_pitch"),
  ("ML1_hip_roll", "MR1_hip_roll"),
  ("ML2_thigh_yaw", "MR2_thigh_yaw"),
  ("ML3_knee_pitch", "MR3_knee_pitch"),
  ("ML4_ankle_pitch", "MR4_ankle_pitch"),
)


def main() -> None:
  args = _parse_args()
  samples = _load_samples(args.input)
  _plot_pairs(samples, args.value, args.input, args.save)
  if not args.no_show:
    plt.show()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Plot optim motor metrics.")
  parser.add_argument("--input", type=Path, required=True, help="Time-series CSV.")
  parser.add_argument(
    "--value",
    choices=(
      "tau_motor",
      "tau_motor_cmd",
      "tau_joint",
      "motor_speed",
      "power",
      "q_error",
    ),
    default="tau_motor",
    help="Metric column to plot.",
  )
  parser.add_argument("--save", type=Path, default=None, help="Save figure to file.")
  parser.add_argument("--no-show", action="store_true", help="Do not open a window.")
  return parser.parse_args()


def _load_samples(path: Path) -> dict[str, dict[str, list[float]]]:
  grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
  with path.open(newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
      actuator = row["actuator"]
      for key, value in row.items():
        if key in {"actuator", "joint", "saturated"}:
          continue
        grouped[actuator][key].append(float(value))
  return grouped


def _plot_pairs(
  samples: dict[str, dict[str, list[float]]],
  value: str,
  source: Path,
  save_path: Path | None,
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
  if save_path is not None:
    fig.savefig(save_path, dpi=160)


if __name__ == "__main__":
  main()
