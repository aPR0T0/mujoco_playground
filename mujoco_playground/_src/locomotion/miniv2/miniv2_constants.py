# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for Berkeley Humanoid."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "miniv2"
NO_HEAD_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
NO_HEAD_ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_rough_terrain.xml"
NO_HEAD_FLAT_TERRAIN_BACKLASH_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain_nohead_backlash.xml"
FLAT_TERRAIN_BACKLASH_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain_backlash.xml"
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
ROUGH_TERRAIN_BACKLASH_XML = ROOT_PATH / "xmls" / "scene_mjx_rough_terrain_backlash.xml"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "no_head_flat_terrain": NO_HEAD_FLAT_TERRAIN_XML,
        "no_head_rough_terrain": NO_HEAD_ROUGH_TERRAIN_XML,
        "no_head_flat_terrain_backlash": NO_HEAD_FLAT_TERRAIN_BACKLASH_XML,
        "flat_terrain_backlash": FLAT_TERRAIN_BACKLASH_XML,
        "flat_terrain": FLAT_TERRAIN_XML,
        "rough_terrain_backlash": ROUGH_TERRAIN_BACKLASH_XML,
    }[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

LEFT_FEET_GEOMS = [
    "left_foot_bottom_tpu",
]

RIGHT_FEET_GEOMS = [
    "right_foot_bottom_tpu",
]

HIP_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
]

KNEE_JOINT_NAMES = [
    "left_knee",
    "right_knee",
]

# There should be a way to get that from the mjModel...
JOINTS_ORDER_NO_HEAD = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"