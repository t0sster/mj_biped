from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

body_mass: EventTermCfg = EventTermCfg(
    mode="reset",  # randomize each episode
    func=dr.pseudo_inertia,
    params={
        "asset_cfg": SceneEntityCfg("tinker", body_names=[".*_foot.*"]),
        "ranges": (0.3, 1.2),
        "operation": "abs",
    },
)