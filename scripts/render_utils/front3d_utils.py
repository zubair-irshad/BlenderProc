import blenderproc as bproc
import numpy as np
import os
from os.path import join
import shutil
import cv2
import json
from typing import List

# LAYOUT_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FRONT"
# TEXTURE_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FRONT-texture"
# MODEL_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FUTURE-model"

LAYOUT_DIR = "/home/ubuntu/Downloads/3D-FRONT"
TEXTURE_DIR = "/home/ubuntu/Downloads/3D-FRONT-texture"
MODEL_DIR = "/home/ubuntu/Downloads/3D-FUTURE-model"


RENDER_TEMP_DIR = "./FRONT3D_render/temp"
IMG_WIDTH = 640
IMG_HEIGHT = 480
K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]])


def construct_scene_list():
    scene_list_all = []
    """Construct a list of scenes and save to SCENE_LIST global variable."""
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        scene_list_all.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(scene_list_all)} scenes in total")

    return scene_list_all


def get_scene_bbox(loaded_objects=None, scene_objs_dict=None):
    """Return the bounding box of the scene."""
    bbox_mins = []
    bbox_maxs = []
    if loaded_objects != None:
        for i, object in enumerate(loaded_objects):
            bbox = object.get_bound_box()
            bbox_mins.append(np.min(bbox, axis=0))
            bbox_maxs.append(np.max(bbox, axis=0))
            scene_min = np.min(bbox_mins, axis=0)
            scene_max = np.max(bbox_maxs, axis=0)
            return scene_min, scene_max
    elif scene_objs_dict != None:
        return scene_objs_dict["bbox"]
    else:
        raise ValueError("Either loaded_objects or scene_objs_dict should be provided.")


def get_room_bbox(
    scene_idx, room_idx, scene_objects=None, scene_objs_dict=None, room_config=None
):
    """Return the bounding box of the room."""
    # get global height
    scene_min, scene_max = get_scene_bbox(scene_objects, scene_objs_dict)
    room_config_loaded = room_config[scene_idx][room_idx]
    # overwrite width and length with room config
    scene_min[:2] = room_config_loaded["bbox"][0]
    scene_max[:2] = room_config_loaded["bbox"][1]

    return [scene_min, scene_max]


def bbox_contained(bbox_a, bbox_b):
    """Return whether the bbox_a is contained in bbox_b."""
    return (
        bbox_a[0][0] >= bbox_b[0][0]
        and bbox_a[0][1] >= bbox_b[0][1]
        and bbox_a[0][2] >= bbox_b[0][2]
        and bbox_a[1][0] <= bbox_b[1][0]
        and bbox_a[1][1] <= bbox_b[1][1]
        and bbox_a[1][2] <= bbox_b[1][2]
    )


OBJ_BAN_LIST = [
    "Baseboard",
    "Pocket",
    "Floor",
    "SlabSidde.",
    "WallInner",
    "WallOuter",
    "Front",
    "WallTop",
    "WallBottom",
    "Ceiling.",
    "FeatureWall",
    "LightBand",
    "SlabSide",
    "ExtrusionCustomizedCeilingModel",
    "Cornice",
    "ExtrusionCustomizedBackgroundWall",
    "Back",
]


def filter_objs_in_dict(room_objs_dict):
    """Clean up objects according to merge_list, global OBJ_BAN_LIST, keyword_ban_list, and fullname_ban_list."""

    # check merge_list
    # TODO: merge_list support obb
    # room_objs_dict = merge_bbox_in_dict(scene_idx, room_idx, room_objs_dict)

    ori_objects = room_objs_dict["objects"]
    result_objects = []
    for obj_dict in ori_objects:
        obj_name = obj_dict["name"]
        flag_use = True
        # check global OBJ_BAN_LIST
        for ban_word in OBJ_BAN_LIST:
            if ban_word in obj_name:
                flag_use = False
        # check keyword_ban_list
        # if "keyword_ban_list" in ROOM_CONFIG[scene_idx][room_idx].keys():
        #     for ban_word in ROOM_CONFIG[scene_idx][room_idx]["keyword_ban_list"]:
        #         if ban_word in obj_name:
        #             flag_use = False
        # # check fullname_ban_list
        # if "fullname_ban_list" in ROOM_CONFIG[scene_idx][room_idx].keys():
        #     for fullname in ROOM_CONFIG[scene_idx][room_idx]["fullname_ban_list"]:
        #         if fullname == obj_name.strip():
        #             flag_use = False

        if flag_use:
            result_objects.append(obj_dict)

    room_objs_dict["objects"] = result_objects

    return room_objs_dict


def render_poses(poses, temp_dir=RENDER_TEMP_DIR) -> List:
    """Render a scene with a list of poses.
    No room idx is needed because the poses can be anywhere in the room."""

    # add camera poses to render queue
    for cam2world_matrix in poses:
        bproc.camera.add_camera_pose(cam2world_matrix)

    # render
    bproc.renderer.set_light_bounces(
        diffuse_bounces=200,
        glossy_bounces=200,
        max_bounces=200,
        transmission_bounces=200,
        transparent_max_bounces=200,
    )
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)
    data = bproc.renderer.render(output_dir=temp_dir)
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data["colors"]]

    return imgs


def save_in_ngp_format(
    imgs, poses, intrinsic, room_bbox, room_objs_dict, bbox_type, dst_dir
):
    """Save images and poses to ngp format dataset."""
    print("Save in instant-ngp format...")
    train_dir = join(dst_dir, "train")
    imgdir = join(dst_dir, "train", "images")

    if os.path.isdir(imgdir) and len(os.listdir(imgdir)) > 0:
        input(
            "Warning: The existing images will be overwritten. Press enter to continue..."
        )
        shutil.rmtree(imgdir)
    os.makedirs(imgdir, exist_ok=True)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    angle_x = 2 * np.arctan(cx / fx)
    angle_y = 2 * np.arctan(cy / fy)

    print("room_objs_dict[bbox]", room_objs_dict)
    room_bbox = np.array(room_objs_dict["bbox"])
    # room_bbox = np.array(room_objs_dict)
    scale = 1.5 / np.max(room_bbox[1] - room_bbox[0])
    cent_after_scale = scale * (room_bbox[0] + room_bbox[1]) / 2.0
    offset = np.array([0.5, 0.5, 0.5]) - cent_after_scale

    out = {
        "camera_angle_x": float(angle_x),
        "camera_angle_y": float(angle_y),
        "fl_x": float(fx),
        "fl_y": float(fy),
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "cx": float(cx),
        "cy": float(cy),
        "w": int(IMG_WIDTH),
        "h": int(IMG_HEIGHT),
        "aabb_scale": 2,
        "scale": float(scale),
        "offset": offset.tolist(),
        "room_bbox": room_bbox.tolist(),
        # "num_room_objects": len(room_objs_dict["objects"]),
        "frames": [],
        "bounding_boxes": [],
    }

    for i, pose in enumerate(poses):
        frame = {
            "file_path": join("images/{:04d}.jpg".format(i)),
            "transform_matrix": pose.tolist(),
        }
        out["frames"].append(frame)

    # out["bounding_boxes"] = get_ngp_type_boxes(room_objs_dict, bbox_type)

    # out['is_merge_bbox'] = 'No'

    with open(join(train_dir, "transforms.json"), "w") as f:
        json.dump(out, f, indent=4)

    if imgs == None:  # support late rendering
        imgs = render_poses(poses, imgdir)

    for i, img in enumerate(imgs):
        cv2.imwrite(join(imgdir, "{:04d}.jpg".format(i)), img)


def get_room_objs_dict(room_bbox, scene_objs_dict):
    """Get the room object dictionary containing all the objects in the room."""
    room_objects = []
    scene_objects = scene_objs_dict["objects"]
    for obj_dict in scene_objects:
        if bbox_contained(obj_dict["aabb"], room_bbox):
            room_objects.append(obj_dict)

    room_objs_dict = {}
    room_objs_dict["bbox"] = np.array(room_bbox)
    room_objs_dict["objects"] = room_objects

    return room_objs_dict
