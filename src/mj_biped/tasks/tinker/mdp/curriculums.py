from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class _EventCurriculumStageOptional(TypedDict, total=False):
  params: dict[str, Any]


class EventCurriculumStage(_EventCurriculumStageOptional):
  step: int


class event_curriculum:
  """Update a domain-randomization event's params based on training steps.

  Lets a DR event (CoM offset, surface friction, ...) start narrow and widen
  its sampling range as training progresses, instead of randomizing at full
  difficulty from step 0. Only affects the event's *params* dict, so it works
  with any event whose ``mode`` re-samples periodically (e.g. ``"reset"``) --
  an event with ``mode="startup"`` only ever samples once and won't pick up
  later stages.

  Example::

    CurriculumTermCfg(
      func=mdp.event_curriculum,
      params={
        "event_name": "body_mass",
        "stages": [
          {"step": 0, "params": {"t1_range": (-0.01, 0.01)}},
          {"step": 30_000, "params": {"t1_range": (-0.04, 0.04)}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    stages: list[EventCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)
    self._stages = stages

    for i in range(1, len(stages)):
      if stages[i]["step"] < stages[i - 1]["step"]:
        raise ValueError(
          f"Curriculum stages for event '{event_name}' must be in "
          f"nondecreasing step order, but stage {i} has step "
          f"{stages[i]['step']} < {stages[i - 1]['step']}."
        )
    for stage in stages:
      unknown = stage.get("params", {}).keys() - self._term_cfg.params.keys()
      if unknown:
        raise KeyError(
          f"Stage at step {stage['step']} sets unknown param(s) {unknown} "
          f"on event '{event_name}'. Check for typos."
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[EventCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        self._term_cfg.params.update(stage.get("params", {}))

    logged_params: set[str] = set()
    for stage in self._stages:
      logged_params.update(stage.get("params", {}))

    result: dict[str, torch.Tensor] = {}
    for key in logged_params:
      value = self._term_cfg.params[key]
      if isinstance(value, tuple) and len(value) == 2:
        lo, hi = value
        result[f"{key}_low"] = torch.tensor(float(lo))
        result[f"{key}_high"] = torch.tensor(float(hi))
      elif isinstance(value, (int, float, bool)):
        result[key] = torch.tensor(value)
      elif isinstance(value, torch.Tensor):
        result[key] = value
    return result
