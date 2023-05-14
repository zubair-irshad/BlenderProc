import blenderproc as bproc
import numpy as np
import sys
from os.path import join
import yaml
import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scene_idx", type=int, required=True)


sys.path.append("./scripts")
from blenderproc.python.sampler.Front3DPointInRoomSampler import (
    Front3DPointInRoomSampler,
)
from load_helper import load_scene_objects, build_and_save_scene_cache

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


def load_scene_objects_wotexture(scene_idx, scene_list_all):
    check_cache_dir(scene_idx)
    loaded_objects = bproc.loader.load_front3d(
        json_path=scene_list_all[scene_idx],
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        ceiling_light_strength=1,
        lamp_light_strength=30,
    )
    return loaded_objects


def construct_scene_list():
    scene_list_all = []
    """Construct a list of scenes and save to SCENE_LIST global variable."""
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        scene_list_all.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(scene_list_all)} scenes in total")


scene_list_all = construct_scene_list()

room_bboxes = {}


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scene_idx", type=int, required=True)
    return parser.parse_args()


def save_bbox_and_scene_cache(idx):
    # for idx in range(30):
    # scene_idx = 13

    cache_dir = f"./cached/{idx}"
    # loaded_objects = load_scene_objects_wotexture(idx, scene_list_all)

    loaded_objects = load_scene_objects(idx, scene_list_all)
    scene_objs_dict = build_and_save_scene_cache(cache_dir, loaded_objects)

    point_sampler = Front3DPointInRoomSampler(loaded_objects)
    floors = point_sampler.used_floors
    if len(floors) > 0:
        # room_bboxes[scene_idx] = []
        room_bboxes[idx] = {}
        for i, floor_obj in enumerate(floors):
            # floor_obj = point_sampler.used_floors[i]
            bounding_box = floor_obj.get_bound_box()
            min_corner = np.min(bounding_box, axis=0)
            max_corner = np.max(bounding_box, axis=0)

            room_bboxes[idx][i] = {
                "bbox": [min_corner[:2].tolist(), max_corner[:2].tolist()]
            }
        save_path = os.path.join(
            "/home/mirshad7/BlenderProc/scripts/all_bboxes_w_cache",
            "bbox_" + str(idx) + ".yaml",
        )
        with open(save_path, "w") as file:
            yaml.dump(room_bboxes, file)


def main():
    args = parse_args()
    save_bbox_and_scene_cache(args.scene_idx)


if __name__ == "__main__":
    main()
    print("Success.")
