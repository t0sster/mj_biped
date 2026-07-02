# optim

Small MuJoCo experiments for motor capability checks.

The package is intentionally split into:

- `core/`: model loading, shared actuator maps, PD control, motor metrics.
- `tasks/`: scripted scenarios that provide desired joint positions over time.
- `scripts/`: runnable entry points.

Run the first standing experiment:

```bash
uv run optim-play --task stand --duration 2.0
```

Run a scripted stand-to-crouch-to-stand transition:

```bash
uv run optim-play --task squat_stand
```

Edit `SQUAT_STAND_POSES` and `SquatStandTiming` in
`src/optim/tasks/squat_stand.py` to change the target joint poses and phase
durations.

Run a scripted vertical jump attempt:

```bash
uv run optim-play --task jump_vertical
```

Edit `JUMP_VERTICAL_POSES` and `JumpVerticalTiming` in
`src/optim/tasks/jump_vertical.py` to change crouch, extension, flight-pose
targets and phase durations.

Override target joint angles:

```bash
uv run optim-play \
  --joint JL3_knee_pitch=0.4 \
  --joint JR3_knee_pitch=0.4
```

Write the summary table to CSV:

```bash
uv run optim-play --output /tmp/stand_metrics.csv
```

Write per-step metrics and plot paired left/right motor torques:

```bash
uv run optim-play --pose crouch --duration 2.0 --log
uv run optim-plot --input logs/optim/stand/<run-time>/timeseries.csv
```

With `--log`, each run writes `summary.csv` and `timeseries.csv` under
`logs/optim/<task>/<run-time>/`.

Open the MuJoCo viewer and replay the episode in a loop:

```bash
uv run optim-play --task stand --pose crouch --viewer --duration 2.0
```

The controller interface is position-based:

```text
q_des -> PD joint torque -> motor torque / gear -> ctrl clipping -> MuJoCo
```

The logged metrics stay motor-focused so different tasks can be compared with
the same output schema.
