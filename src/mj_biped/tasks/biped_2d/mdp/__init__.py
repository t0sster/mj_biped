"""MDP terms for the biped_2d task."""

from .rewards import alive
from .rewards import forward_velocity
from .rewards import joint_torque
from .rewards import joint_velocity
from .rewards import lateral_centering
from .rewards import upright
from .runner import biped_2d_ppo_runner_cfg
