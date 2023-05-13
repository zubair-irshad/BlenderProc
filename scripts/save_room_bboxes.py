import blenderproc as bproc
import numpy as np
import sys
from os.path import join
import yaml
import os

sys.path.append("./scripts")
from blenderproc.python.sampler.Front3DPointInRoomSampler import (
    Front3DPointInRoomSampler,
)


LAYOUT_DIR = "/home/mirshad7/Downloads/3D-FRONT"
TEXTURE_DIR = "/home/mirshad7/Downloads/3D-FRONT-texture"
MODEL_DIR = "/home/mirshad7/Downloads/3D-FUTURE-model"

mapping_file = bproc.utility.resolve_resource(
    os.path.join("front_3D", "3D_front_mapping.csv")
)
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)


def check_cache_dir(scene_idx):
    if not os.path.isdir(f"./cached/{scene_idx}"):
        os.makedirs(f"./cached/{scene_idx}")


def load_scene_objects_wotexture(scene_idx):
    check_cache_dir(scene_idx)
    loaded_objects = bproc.loader.load_front3d(
        json_path=SCENE_LIST[scene_idx],
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        ceiling_light_strength=1,
        lamp_light_strength=30,
    )
    return loaded_objects


SCENE_LIST = []


def construct_scene_list():
    """Construct a list of scenes and save to SCENE_LIST global variable."""
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        SCENE_LIST.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(SCENE_LIST)} scenes in total")


construct_scene_list()

room_bboxes = {}

for scene_idx in range(30):
    # scene_idx = 14
    loaded_objects = load_scene_objects_wotexture(scene_idx)
    point_sampler = Front3DPointInRoomSampler(loaded_objects)
    if len(point_sampler.used_floors) > 0:
        room_bboxes[scene_idx] = []
        # room_bboxes[scene_idx] = {}
        for i in range(len(point_sampler.used_floors)):
            floor_obj = point_sampler.used_floors[i]
            bounding_box = floor_obj.get_bound_box()
            min_corner = np.min(bounding_box, axis=0)
            max_corner = np.max(bounding_box, axis=0)

            # room_bboxes[scene_idx][i] = {
            #     "bbox": [min_corner[:2].tolist(), max_corner[:2].tolist()]
            # }
            room_bboxes[scene_idx].append(
                [min_corner[:2].tolist(), max_corner[:2].tolist()]
            )
            # room_bboxes[scene_idx][i] = {
            #     "bbox": [min_corner[:2].tolist(), max_corner[:2].tolist()]
            # }
    del point_sampler
    del loaded_objects

    # print("bounding box", bounding_box)
    # print("min_corner, max_corner,", i, ":::", min_corner, max_corner)

with open("/home/mirshad7/BlenderProc/scripts/room_bboxes_modified.yaml", "w") as file:
    yaml.dump(room_bboxes, file)
