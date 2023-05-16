import blenderproc as bproc
import sys
from os.path import join
import yaml
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scene_idx", type=int, required=True)


import sys

sys.path.append("./scripts")
from render_utils.pose_utils import *
from render_utils.front3d_utils import *
from load_helper import load_scene_objects
from utils import build_and_save_scene_cache

# LAYOUT_DIR = "/home/mirshad7/Downloads/3D-FRONT"
# TEXTURE_DIR = "/home/mirshad7/Downloads/3D-FRONT-texture"
# MODEL_DIR = "/home/mirshad7/Downloads/3D-FUTURE-model"


LAYOUT_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FRONT"
TEXTURE_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FRONT-texture"
MODEL_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FUTURE-model"


def check_cache_dir(scene_idx):
    if not os.path.isdir(f"./cached/{scene_idx}"):
        os.makedirs(f"./cached/{scene_idx}")


def construct_scene_list():
    scene_list_all = []
    """Construct a list of scenes and save to SCENE_LIST global variable."""
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        scene_list_all.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(scene_list_all)} scenes in total")
    return scene_list_all


scene_list_all = construct_scene_list()

room_bboxes = {}


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scene_idx", type=int, required=True)
    return parser.parse_args()


def check_valid_poses(scene_idx):
    args = parse_args()

    room_config_folder = "./scripts/all_bboxes"
    room_config_path = os.path.join(
        room_config_folder, "bbox_" + str(scene_idx) + ".yaml"
    )

    with open(room_config_path, "r") as f:
        room_config = yaml.load(f, Loader=yaml.FullLoader)

    scene_list_all = construct_scene_list()

    cache_dir = f"./cached/{scene_idx}"
    scene_objects = load_scene_objects(scene_idx, scene_list_all)
    scene_objs_dict = build_and_save_scene_cache(cache_dir, scene_objects)

    all_value = True
    for room_idx in room_config[scene_idx].keys():
        print("==============Clearing Keyframaes")
        # Clear all key frames from the previous run
        print("room idx", room_idx)

        room_bbox = get_room_bbox(
            args.scene_idx,
            room_idx,
            scene_objs_dict=scene_objs_dict,
            room_config=room_config,
        )
        room_objs_dict = get_room_objs_dict(room_bbox, scene_objs_dict)
        room_objs_dict = filter_objs_in_dict(room_objs_dict)

        try:
            assert (
                len(room_objs_dict["objects"]) > 0
            ), "no objects in the room, moving to next ..."
        except AssertionError as msg:
            print(msg)
            sys.exit(1)  # exit the script with a non-zero status code

        poses, num_closeup, num_global = generate_room_poses(
            scene_idx,
            room_idx,
            room_objs_dict,
            room_bbox,
            num_poses_per_object=args.pos_per_obj,
            max_global_pos=args.max_global_pos,
            global_density=args.global_density,
            room_config=room_config,
        )

        value = True if len(poses) > 80 or len(poses) < 600 else False
        all_value = all_value and value

    data = {"value": all_value}

    save_path = os.path.join(
        "/home/ubuntu/zubair/BlenderProc/scripts/all_is_valid",
        "is_valid_" + str(scene_idx) + ".yaml",
    )
    with open(save_path, "w") as file:
        yaml.dump(data, file)


def main():
    args = parse_args()
    check_valid_poses(args.scene_idx)


if __name__ == "__main__":
    main()
    print("Success.")
