import blenderproc as bproc
import sys
from os.path import join
import yaml
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scene_idx", type=int, required=True)
    parser.add_argument(
        "-ppo",
        "--pos_per_obj",
        type=int,
        default=15,
        help="Number of close-up poses for each object.",
    )
    parser.add_argument(
        "-gp",
        "--max_global_pos",
        type=int,
        default=150,
        help="Max number of global poses.",
    )
    parser.add_argument(
        "-gd",
        "--global_density",
        type=float,
        default=0.15,
        help="The radius interval of global poses. Smaller global_density -> more global views",
    )
    parser.add_argument(
        "-nc",
        "--no_check",
        action="store_true",
        default=True,
        help="Do not check the poses. Render directly.",
    )
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Relabel the objects in the scene by rewriting transforms.json.",
    )
    parser.add_argument(
        "--rotation",
        action="store_true",
        help="output rotation bounding boxes if it is true.",
    )
    parser.add_argument(
        "--bbox_type",
        type=str,
        default="obb",
        choices=["aabb", "obb"],
        help="Output aabb or obb",
    )
    parser.add_argument(
        "--render_root",
        type=str,
        default="/wild6d_data/zubair/FRONT3D_render",
        help="Output directory. If not specified, use the default directory.",
    )

    parser.add_argument(
        "--seg_res",
        type=int,
        default=256,
        help="The max grid resolution for 3D segmentation map.",
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="",
        help="The directory containing the poses (transforms.json) for 2D mask rendering.",
    )


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


# def parse_args():
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument("--scene_idx", type=int, required=True)
#     return parser.parse_args()


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
