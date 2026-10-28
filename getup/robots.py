"""Robot configs. Any MuJoCo humanoid/legged model with a free-joint root and position actuators works:
point ``xml`` at a scene (robot + floor), name the standing keyframe and the target pelvis height."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


@dataclass
class RobotCfg:
    name: str
    xml: str
    stand_key: str  # keyframe used as the standing target pose
    stand_height: float  # root height when standing
    action_scale: float = 0.5  # rad per unit action, around the standing pose
    control_dt: float = 0.02
    sim_dt: float | None = None  # override the model timestep
    solver_iterations: int | None = None
    extra: dict = field(default_factory=dict)


ROBOTS = {
    # Unitree G1 29-DoF, MuJoCo Menagerie MJX variant (vendored in assets/unitree_g1, menagerie @ feadf76)
    "g1": RobotCfg(
        name="g1",
        xml=os.path.join(ASSETS, "unitree_g1", "scene_mjx.xml"),
        stand_key="knees_bent",
        stand_height=0.755,
        sim_dt=0.004,
        solver_iterations=10,
    ),
    # SMPL humanoid (ProtoMotions MJCF, converted by tools/smpl_to_scene.py): capsules/boxes, 23 balls x 3 hinges
    "smpl": RobotCfg(
        name="smpl",
        xml=os.path.join(ASSETS, "smpl_scene.xml"),
        stand_key="stand",
        stand_height=0.935,
    ),
}
