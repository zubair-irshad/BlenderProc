import blenderproc as bproc
import numpy as np
import sys
from os.path import join

sys.path.append("./scripts")
from load_helper import *
from render_configs import *
from blenderproc.python.sampler.Front3DPointInRoomSampler import (
    Front3DPointInRoomSampler,
)

LAYOUT_DIR = "/home/mirshad7/Downloads/3D-FRONT"


def construct_scene_list():
    """Construct a list of scenes and save to SCENE_LIST global variable."""
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        SCENE_LIST.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(SCENE_LIST)} scenes in total")


construct_scene_list()

room_bboxes = {}

for scene_idx in range(1):
    scene_idx = 25
    loaded_objects = load_scene_objects(scene_idx)
    point_sampler = Front3DPointInRoomSampler(loaded_objects)
    if len(point_sampler.used_floors) > 0:
        room_bboxes[scene_idx] = {}
        for i in range(len(point_sampler.used_floors)):
            floor_obj = point_sampler.used_floors[i]
            bounding_box = floor_obj.get_bound_box()
            min_corner = np.min(bounding_box, axis=0)
            max_corner = np.max(bounding_box, axis=0)

            room_bboxes[scene_idx][i] = {
                "bbox": [min_corner[:2].tolist(), max_corner[:2].tolist()]
            }

        # print("bounding box", bounding_box)
        # print("min_corner, max_corner,", i, ":::", min_corner, max_corner)

with open("/home/mirshad7/BlenderProc/scripts/room_bboxes.yaml", "w") as file:
    yaml.dump(room_bboxes, file)
