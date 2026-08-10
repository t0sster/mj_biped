from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

foot_friction: EventTermCfg = EventTermCfg(
  mode="reset",  # randomize each episode
  func=dr.geom_friction,
  params={
    "asset_cfg": SceneEntityCfg(
      "tinker", geom_names=["left_foot_collision", "right_foot_collision"]
    ),
    "ranges": (0.3, 1.2),
    "operation": "abs",
  },
)
