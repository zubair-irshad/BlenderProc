# python cli.py run ./scripts/utils.py
import blenderproc as bproc

"""
    Example commands:

        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode plan
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode overview 
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -ppo 10 -gd 0.15
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -ppo 0 -gp 5
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -nc

"""

import os
from os.path import join
import numpy as np
import sys
import yaml

# sys.path.append('/data/jhuangce/BlenderProc/scripts')
# sys.path.append('/data2/jhuangce/BlenderProc/scripts')
# from floor_plan import *
# from load_helper import *
# from render_configs import *
# from utils import *

from os.path import join
import argparse

import sys

sys.path.append("./scripts")
from render_utils.pose_utils import *
from render_utils.front3d_utils import *
from load_helper import load_scene_objects
from utils import build_and_save_scene_cache

pi = np.pi
cos = np.cos
sin = np.sin
COMPUTE_DEVICE_TYPE = "CUDA"


def main():
    args = parse_args()

    room_config_folder = "./scripts/all_bboxes"
    room_config_path = os.path.join(
        room_config_folder, "bbox_" + str(args.scene_idx) + ".yaml"
    )

    print("======================================================\n\n\n")
    print("room_config_path", room_config_path)
    print("======================================================\n\n\n")
    with open(room_config_path, "r") as f:
        room_config = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    scene_list_all = construct_scene_list()

    cache_dir = f"./cached/{args.scene_idx}"
    # load objects
    compute_device = "cuda:" + args.gpu
    bproc.init(compute_device=compute_device, compute_device_type=COMPUTE_DEVICE_TYPE)
    scene_objects = load_scene_objects(args.scene_idx, scene_list_all)
    scene_objs_dict = build_and_save_scene_cache(cache_dir, scene_objects)

    for room_idx in room_config[args.scene_idx].keys():
        print("==============Clearing Keyframaes")
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()
        print("room idx", room_idx)
        dst_dir = join(
            args.render_root, "3dfront_{:04d}_{:02}".format(args.scene_idx, room_idx)
        )
        os.makedirs(dst_dir, exist_ok=True)

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
            args.scene_idx,
            room_idx,
            room_objs_dict,
            room_bbox,
            num_poses_per_object=args.pos_per_obj,
            max_global_pos=args.max_global_pos,
            global_density=args.global_density,
            room_config=room_config,
        )
        if not args.no_check:
            print("Render for scene {}, room {}:".format(args.scene_idx, room_idx))
            for obj_dict in room_objs_dict["objects"]:
                print(f"\t{obj_dict['aabb']}")
            print(
                "Total poses: {}[global] + {}[closeup] x {}[object] = {} poses".format(
                    num_global,
                    args.pos_per_obj,
                    len(room_objs_dict["objects"]),
                    len(poses),
                )
            )
            print("Estimated time: {} minutes".format(len(poses) * 25 // 60))
            input("Press Enter to continue...")

        bbox_type = args.bbox_type
        save_in_ngp_format(
            None, poses, K, room_bbox, room_objs_dict, bbox_type, dst_dir
        )  # late rendering


###########################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--scene_idx", type=int, required=True)
    parser.add_argument("-r", "--room_idx", type=int, required=False)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["plan", "overview", "render", "bbox", "seg", "depth"],
        help="plan: Generate the floor plan of the scene. \
                              overview:Generate 4 corner overviews with bbox projected. \
                              render: Render images in the scene. \
                              bbox: Overwrite bboxes by regenerating transforms.json."
        "\nseg: Create 3D semantic/instance segmentation map.",
    )
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

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return parser.parse_args()


if __name__ == "__main__":
    main()
    print("Success.")
