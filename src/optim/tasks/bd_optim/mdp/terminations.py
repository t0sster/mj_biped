from __future__ import annotations

import math

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bodies_contact_with_terrain(
  env,
  asset_cfg: SceneEntityCfg,
  terrain_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  terrain = env.scene[terrain_cfg.name]

  body_ids = asset.data.indexing.body_ids[asset_cfg.body_ids]
  asset_geom_ids = asset.data.indexing.geom_ids
  asset_geom_body_ids = asset.data.model.geom_bodyid[asset_geom_ids]
  terrain_geom_ids = terrain.data.indexing.geom_ids[terrain_cfg.geom_ids]

  contact = env.sim.data.contact
  ncon = env.sim.data.nacon[0]
  contact_ids = torch.arange(contact.geom.shape[0], device=env.device)
  valid_contacts = contact_ids < ncon
  geom_0 = contact.geom[:, 0]
  geom_1 = contact.geom[:, 1]

  contacts = torch.zeros(
    (env.num_envs, len(body_ids)),
    dtype=torch.bool,
    device=env.device,
  )
  for body_index, body_id in enumerate(body_ids):
    body_geom_ids = asset_geom_ids[asset_geom_body_ids == body_id]
    body_ground_contact = (
      torch.isin(geom_0, body_geom_ids) & torch.isin(geom_1, terrain_geom_ids)
    ) | (
      torch.isin(geom_1, body_geom_ids) & torch.isin(geom_0, terrain_geom_ids)
    )
    env_ids = contact.worldid[valid_contacts & body_ground_contact].long()
    contacts[env_ids, body_index] = True

  return contacts


def base_contact_with_ground(
  env,
  asset_cfg: SceneEntityCfg,
  terrain_cfg: SceneEntityCfg,
) -> torch.Tensor:
  return torch.any(
    bodies_contact_with_terrain(env, asset_cfg=asset_cfg, terrain_cfg=terrain_cfg),
    dim=1,
  )


def bad_orientation(
  env,
  limit_angle: float = math.radians(70.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return projected_gravity[:, 2] > -math.cos(limit_angle)
