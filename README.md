# mj-biped

bipedal robot tasks for `mjlab` and MuJoCo.

## Development

Install the project and development tools with `uv`:

```bash
uv sync --dev
```

Run lint checks:

```bash
uv run ruff check .
```

Run tests:

```bash
uv run pytest
```

The test suite is intentionally lightweight: it checks package helpers and
environment configuration without running training or a full simulation.

## More

### Play bd_optim plotter

```bash
uv run optim-play-bd-checkpoint bd_optim/checkpoint_folder_name --duration 10 --lin-vel-x 0.2
```

### Policy Export

Export the latest checkpoint from a run under `logs/rsl_rl`:

```bash
uv run mj-biped-export-policy --path robot_2d/run-date/model_.pt
```

Export an exact checkpoint path:

```bash
uv run mj-biped-export-policy --full-path --path /path/to/model_.pt
```

## CI

GitHub Actions runs on pushes to `main`/`dev-rp` and on pull requests. The
workflow installs locked dependencies, runs `ruff`, and runs `pytest`.
