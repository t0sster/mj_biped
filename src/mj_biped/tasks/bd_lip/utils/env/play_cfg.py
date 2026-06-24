from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import TerrainEntityCfg

from mj_biped.tasks.bd_lip.bd_lip_env_cfg import make_bd_lip_env_cfg
from mj_biped.tasks.bd_lip.utils.env.external_command import ExternalVelocityCommandCfg

CUBE_SIZE: tuple[float, float, float] = (0.15, 0.15, 0.15)


def _make_cube_spec(name: str = "obstacle") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.pos = [0.0, 0.0, 0.0]
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(CUBE_SIZE),
        rgba=[0.85, 0.33, 0.1, 1.0],
    )
    return spec


def make_traj_play_env_cfg(
    num_envs: int = 1,
    cube_positions: list[tuple[float, float, float]] = [(2.0, 0.0, 0.15)],
) -> ManagerBasedRlEnvCfg:
    cfg = make_bd_lip_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 1_000_000.0
    cfg.terminations.clear()

    cfg.events["reset_base"] = EventTermCfg(
        func=env_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=env_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )

    cfg.commands["base_velocity"] = ExternalVelocityCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=True,
    )
    cfg.commands["lip_step_command"].debug_vis = False

    for i, pos in enumerate(cube_positions):
        name = f"obstacle_{i}"
        cfg.scene.entities[name] = EntityCfg(
            spec_fn=lambda n=name: _make_cube_spec(n),
            init_state=EntityCfg.InitialStateCfg(pos=pos),
        )

    return cfg
