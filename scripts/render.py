# python cli.py run ./scripts/utils.py 
import blenderproc as bproc

"""
    Example commands:

        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --plan
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --overview 
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render -ppo 10 -gd 0.15
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render -ppo 0 -gp 5
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render -nc

"""

from random import shuffle
import shutil
import sys
sys.path.append('/home/jhuangce/miniconda3/lib/python3.9/site-packages')
sys.path.append('/home/yliugu/BlenderProc/scripts')
import cv2
import os
from os.path import join
import numpy as np

import imageio
import sys
sys.path.append('/data/jhuangce/BlenderProc/scripts')
sys.path.append('/data2/jhuangce/BlenderProc/scripts')
from floor_plan import *
from load_helper import *
from render_configs import *
from utils import *
from bbox_proj import get_aabb_coords, project_aabb_to_image, project_obb_to_image
import json
from typing import List
from os.path import join
import glob
import argparse
from mathutils import Vector, Matrix


pi = np.pi
cos = np.cos
sin = np.sin
COMPUTE_DEVICE_TYPE = "CUDA"


def construct_scene_list():
    """ Construct a list of scenes and save to SCENE_LIST global variable. """
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        SCENE_LIST.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(SCENE_LIST)} scenes in total")


############################## poses generation ##################################

def normalize(x, axis=-1, order=2):
    l2 = np.linalg.norm(x, order, axis)
    l2 = np.expand_dims(l2, axis)
    l2[l2 == 0] = 1
    return x / l2,

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = np.zeros_like(camera_position)
    else:
        at = np.array(at)
    if up is None:
        up = np.zeros_like(camera_position)
        up[2] = -1
    else:
        up = np.array(up)
    
    z_axis = normalize(camera_position - at)[0]
    x_axis = normalize(np.cross(up, z_axis))[0]
    y_axis = normalize(np.cross(z_axis, x_axis))[0]

    R = np.concatenate([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R

def c2w_from_loc_and_at(cam_pos, at, up=(0, 0, 1)):
    """ Convert camera location and direction to camera2world matrix. """
    c2w = np.eye(4)
    cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
    c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
    return c2w

def generate_four_corner_poses(scene_idx, room_idx):
    """ Return a list of matrices of 4 corner views in the room. """
    bbox_xy = ROOM_CONFIG[scene_idx][room_idx]['bbox']
    corners = [[i+0.3 for i in bbox_xy[0]], [i-0.3 for i in bbox_xy[1]]]
    x1, y1, x2, y2 = corners[0][0], corners[0][1], corners[1][0], corners[1][1]
    at = [(x1+x2)/2, (y1+y2)/2, 1.2]
    locs = [[x1, y1, 2], [x1, y2, 2], [x2, y1, 2], [x2, y2, 2]]

    c2ws = [c2w_from_loc_and_at(pos, at) for pos in locs]
    
    return c2ws

def pos_in_bbox(pos, bbox):
    """
    Check if a point is inside a bounding box.
    Input:
        pos: 3 x 1
        bbox: 2 x 3
    Output:
        True or False
    """
    return  pos[0] >= bbox[0][0] and pos[0] <= bbox[1][0] and \
            pos[1] >= bbox[0][1] and pos[1] <= bbox[1][1] and \
            pos[2] >= bbox[0][2] and pos[2] <= bbox[1][2]

def check_pos_valid(pos, room_objs_dict, room_bbox):
    """ Check if the position is in the room, not too close to walls and not conflicting with other objects. """
    room_bbox_small = [[item+0.5 for item in room_bbox[0]], [room_bbox[1][0]-0.5, room_bbox[1][1]-0.5, room_bbox[1][2]-0.8]] # ceiling is lower
    if not pos_in_bbox(pos, room_bbox_small):
        return False
    for obj_dict in room_objs_dict['objects']:
        obj_bbox = obj_dict['aabb']
        if pos_in_bbox(pos, obj_bbox):
            return False

    return True

def generate_room_poses(scene_idx, room_idx, room_objs_dict, room_bbox, num_poses_per_object, max_global_pos, global_density):
    """ Return a list of poses including global poses and close-up poses for each object."""

    poses = []
    num_closeup, num_global = 0, 0
    h_global = 1.2

    # close-up poses for each object.
    if num_poses_per_object>0:
        for obj_dict in room_objs_dict['objects']:
            obj_bbox = np.array(obj_dict['aabb'])
            cent = np.mean(obj_bbox, axis=0)
            rad = np.linalg.norm(obj_bbox[1]-obj_bbox[0])/2 * 1.7 # how close the camera is to the object
            if np.max(obj_bbox[1]-obj_bbox[0])<1:
                rad *= 1.2 # handle small objects

            positions = []
            n_hori_sects = 30
            n_vert_sects = 10
            theta_bound = [0, 2*pi]
            phi_bound = [-pi/4, pi/4]
            theta_sect = (theta_bound[1] - theta_bound[0]) / n_hori_sects
            phi_sect = (phi_bound[1] - phi_bound[0]) / n_vert_sects
            for i_vert_sect in range(n_vert_sects):
                for i_hori_sect in range(n_hori_sects):
                    theta_a = theta_bound[0] + i_hori_sect * theta_sect
                    theta_b = theta_a + theta_sect
                    phi_a = phi_bound[0] + i_vert_sect * phi_sect
                    phi_b = phi_a + phi_sect
                    theta = np.random.uniform(theta_a, theta_b)
                    phi = np.random.uniform(phi_a, phi_b)
                    pos = [cos(phi)*cos(theta), cos(phi)*sin(theta), sin(phi)]
                    positions.append(pos)
            positions = np.array(positions)
            positions = positions * rad + cent

            positions = [pos for pos in positions if check_pos_valid(pos, room_objs_dict, room_bbox)]
            shuffle(positions)
            if len(positions) > num_poses_per_object:
                positions = positions[:num_poses_per_object]

            poses.extend([c2w_from_loc_and_at(pos, cent) for pos in positions])

            num_closeup = len(positions)

    # global poses
    if max_global_pos>0:
        bbox = ROOM_CONFIG[scene_idx][room_idx]['bbox']
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
        rm_cent = np.array([(x1+x2)/2, (y1+y2)/2, h_global])

        # flower model
        rad_bound = [0.3, 5]
        rad_intv = global_density
        theta_bound = [0, 2*pi]
        theta_sects = 20
        theta_intv = (theta_bound[1] - theta_bound[0]) / theta_sects
        h_bound = [0.8, 2.0]

        positions = []
        theta = theta_bound[0]
        for i in range(theta_sects):
            rad = rad_bound[0]
            while rad < rad_bound[1]:
                h = np.random.uniform(h_bound[0], h_bound[1])
                pos = [rm_cent[0] + rad * cos(theta), rm_cent[1] + rad * sin(theta), h]
                if check_pos_valid(pos, room_bbox_meta, room_bbox):
                    positions.append(pos)
                rad += rad_intv
            theta += theta_intv
        positions = np.array(positions)
        np.random.shuffle(positions)

        if len(positions) > max_global_pos:
            positions = positions[:max_global_pos]

        poses.extend([c2w_from_loc_and_at(pos, [rm_cent[0], rm_cent[1], pos[2]]) for pos in positions])

        num_global = len(positions)
        

    return poses, num_closeup, num_global


#################################################################################

def get_scene_bbox(loaded_objects=None, scene_objs_dict=None):
    """ Return the bounding box of the scene. """
    bbox_mins = []
    bbox_maxs = []
    if loaded_objects!=None:
        for i, object in enumerate(loaded_objects):
            bbox = object.get_bound_box()
            bbox_mins.append(np.min(bbox, axis=0))
            bbox_maxs.append(np.max(bbox, axis=0))
            scene_min = np.min(bbox_mins, axis=0)
            scene_max = np.max(bbox_maxs, axis=0)
            return scene_min, scene_max
    elif scene_objs_dict!=None:
        return scene_objs_dict['bbox']
    else:
        raise ValueError('Either loaded_objects or scene_objs_dict should be provided.')
    

def get_room_bbox(scene_idx, room_idx, scene_objects=None, scene_objs_dict=None):
    """ Return the bounding box of the room. """
    # get global height
    scene_min, scene_max = get_scene_bbox(scene_objects, scene_objs_dict)
    room_config = ROOM_CONFIG[scene_idx][room_idx]
    # overwrite width and length with room config
    scene_min[:2] = room_config['bbox'][0]
    scene_max[:2] = room_config['bbox'][1]

    return [scene_min, scene_max]

def bbox_contained(bbox_a, bbox_b):
    """ Return whether the bbox_a is contained in bbox_b. """
    return bbox_a[0][0]>=bbox_b[0][0] and bbox_a[0][1]>=bbox_b[0][1] and bbox_a[0][2]>=bbox_b[0][2] and \
           bbox_a[1][0]<=bbox_b[1][0] and bbox_a[1][1]<=bbox_b[1][1] and bbox_a[1][2]<=bbox_b[1][2]

def get_room_objects(scene_idx, room_idx, scene_objects, cleanup=False):
    """ Return the objects within the room bbox. Cleanup unecessary objects. """
    objects = []

    room_bbox = get_room_bbox(scene_idx, room_idx, loaded_objects=scene_objects)
    for object in scene_objects:
        obj_bbox_8 = object.get_bound_box()
        obj_bbox = [np.min(obj_bbox_8, axis=0), np.max(obj_bbox_8, axis=0)]
        if bbox_contained(obj_bbox, room_bbox):
            objects.append(object)

    return objects

def merge_bbox(scene_idx, room_idx, room_bbox_meta):
    """ Merge the bounding box of the room. """
    if 'merge_list' in ROOM_CONFIG[scene_idx][room_idx]:
        merge_dict = ROOM_CONFIG[scene_idx][room_idx]['merge_list']
        for label, merge_items in merge_dict.items():
            result_room_bbox_meta, merge_mins, merge_maxs = [], [], []
            for obj in room_bbox_meta:
                if obj[0] in merge_items:
                    merge_mins.append(obj[1][0])
                    merge_maxs.append(obj[1][1])
                else:
                    result_room_bbox_meta.append(obj)
            if len(merge_mins) > 0:
                result_room_bbox_meta.append((label, [np.min(np.array(merge_mins), axis=0), np.max(np.array(merge_maxs), axis=0)]))
            room_bbox_meta = result_room_bbox_meta
    return room_bbox_meta

def merge_bbox_in_dict(scene_idx, room_idx, room_objs_dict):
    """ Merge the bounding box of the room. Operate on the object dictionary with obb """
    if 'merge_list' in ROOM_CONFIG[scene_idx][room_idx]:
        merge_dict = ROOM_CONFIG[scene_idx][room_idx]['merge_list']
        objects = room_objs_dict['objects']
        for merged_label, merge_items in merge_dict.items():
            # select objs to be merged
            result_objects = [obj for obj in objects if obj['name'] not in merge_items]
            objs_to_be_merged = [obj for obj in objects if obj['name'] in merge_items]

            # find the largest object
            largest_obj = None
            largest_vol = 0
            for obj in objs_to_be_merged:
                if obj['volume'] > largest_vol:
                    largest_vol = obj['volume']
                    largest_obj = obj
            
            # extend the largest bbox to include all the other bbox
            local2world = Matrix(largest_obj['l2w'])
            local_maxs, local_mins = np.max(largest_obj['coords_local'], axis=0), np.min(largest_obj['coords_local'], axis=0)
            local_cent = (local_maxs + local_mins) / 2
            global_cent = local2world @ Vector(local_cent)
            h_diag = (local_maxs - local_mins) / 2
            local_vecs = np.array([[h_diag[0], 0, 0], [0, h_diag[1], 0], [0, 0, h_diag[2]]]) + local_cent  # (3, 3)
            global_vecs = [(local2world @ Vector(vec) - local2world @ Vector(local_cent)).normalized() for vec in local_vecs] # (3, 3)
            global_norms = [vec for vec in global_vecs] # (3, 3)
            local_offsets = np.array([-h_diag, h_diag]) # [[x-, y-, z-], [x+, y+, z+]]

            for obj in objs_to_be_merged:
                update = [[0, 0, 0], [0, 0, 0]]
                for point in obj['coords']:
                    for i in range(3):
                        offset = (Vector(point) - global_cent) @ global_norms[i]
                        if offset < local_offsets[0][i]:
                            local_offsets[0][i] = offset
                            update[0][i] = 1
                        elif offset > local_offsets[1][i]:
                            local_offsets[1][i] = offset
                            update[1][i] = 1
            
            # TODO: update: coords, aabb, volume, coords_local
            merged_local_mins, merged_local_maxs = local_offsets + local_cent
            merged_coords_local = get_aabb_coords(np.concatenate([merged_local_mins, merged_local_maxs], axis=0))[:, :3]
            merged_coords = np.array([local2world @ Vector(cord) for cord in merged_coords_local])
            merged_aabb_mins, merged_aabb_maxs = np.min(merged_coords, axis=0), np.max(merged_coords, axis=0)
            merged_aabb = np.array([merged_aabb_mins, merged_aabb_maxs])
            merged_diag_local = merged_local_maxs - merged_local_mins
            merged_volume = merged_diag_local[0] * merged_diag_local[1] * merged_diag_local[2]


            merged_object = {'name': merged_label,
                             'coords': merged_coords,
                             'aabb': merged_aabb,
                             'volume': merged_volume,
                             'l2w': largest_obj['l2w'],
                             'coords_local': merged_coords_local,}
            result_objects.append(merged_object)
            objects = result_objects

        room_objs_dict['objects'] = objects

    return room_objs_dict

def filter_bbox(scene_idx, room_idx, room_bbox_meta):
    """ Clean up according to merge_list, global OBJ_BAN_LIST, keyword_ban_list, and fullname_ban_list. """

    # check merge_list
    room_bbox_meta = merge_bbox(scene_idx, room_idx, room_bbox_meta)
    result_room_bbox_meta = []
    for bbox_meta in room_bbox_meta:
        flag_use = True
        obj_name = bbox_meta[0]

        # check global OBJ_BAN_LIST
        for ban_word in OBJ_BAN_LIST:
            if ban_word in obj_name:
                flag_use=False
        
        # check keyword_ban_list
        if 'keyword_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for ban_word in ROOM_CONFIG[scene_idx][room_idx]['keyword_ban_list']:
                if ban_word in obj_name:
                    flag_use=False
        
        # check fullname_ban_list
        if 'fullname_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for fullname in ROOM_CONFIG[scene_idx][room_idx]['fullname_ban_list']:
                if fullname == obj_name.strip():
                    flag_use=False
        
        if flag_use:
            result_room_bbox_meta.append(bbox_meta)
    
    return result_room_bbox_meta

def filter_objs_in_dict(scene_idx, room_idx, room_objs_dict):
    """ Clean up objects according to merge_list, global OBJ_BAN_LIST, keyword_ban_list, and fullname_ban_list. """

    # check merge_list
    # TODO: merge_list support obb
    room_objs_dict = merge_bbox_in_dict(scene_idx, room_idx, room_objs_dict)

    ori_objects = room_objs_dict['objects']
    result_objects = []
    for obj_dict in ori_objects:
        obj_name = obj_dict['name']
        flag_use = True
        # check global OBJ_BAN_LIST
        for ban_word in OBJ_BAN_LIST:
            if ban_word in obj_name:
                flag_use=False
        # check keyword_ban_list
        if 'keyword_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for ban_word in ROOM_CONFIG[scene_idx][room_idx]['keyword_ban_list']:
                if ban_word in obj_name:
                    flag_use=False
        # check fullname_ban_list
        if 'fullname_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for fullname in ROOM_CONFIG[scene_idx][room_idx]['fullname_ban_list']:
                if fullname == obj_name.strip():
                    flag_use=False
        
        if flag_use:
            result_objects.append(obj_dict)
    
    room_objs_dict['objects'] = result_objects

    return room_objs_dict

def render_poses(poses, temp_dir=RENDER_TEMP_DIR) -> List:
    """ Render a scene with a list of poses. 
        No room idx is needed because the poses can be anywhere in the room. """

    # add camera poses to render queue
    for cam2world_matrix in poses:
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    # render
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200, transmission_bounces=200, transparent_max_bounces=200)
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)
    data = bproc.renderer.render(output_dir=temp_dir)
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data['colors']]

    return imgs

##################################### save to dataset #####################################
def get_ngp_type_boxes(room_objs_dict, bbox_type):
    """ Return a list of bbox in instant-ngp format. """
    bounding_boxes = []
    for i, obj_dict in enumerate(room_objs_dict['objects']):
        if bbox_type == 'aabb':
            obj_aabb = obj_dict['aabb']
            obj_bbox_ngp = {
                "extents": (obj_aabb[1]-obj_aabb[0]).tolist(),
                "orientation": np.eye(3).tolist(),
                "position": ((obj_aabb[0]+obj_aabb[1])/2.0).tolist(),
            }
            bounding_boxes.append(obj_bbox_ngp)
        elif bbox_type == 'obb':
            obj_coords = obj_dict['coords']
            # TODO: 8 point to [x, y, z, w, l, h, theta]
            np.set_printoptions(precision=2)
            obb = poly2obb_3d(obj_coords)
            extents, orientation, position = obb2ngp(obb)

            if obj_dict['name'] == 'chair':
                print('x y z w l h theta', obb)
                print('extents', extents)
                print('orientation', orientation)
                print('position', position)
            obj_bbox_ngp = {
                "extents": extents,
                "orientation": orientation,
                "position": position,
            }
            bounding_boxes.append(obj_bbox_ngp)
    return bounding_boxes

def save_in_ngp_format(imgs, poses, intrinsic, room_objs_dict, bbox_type, dst_dir):
    """ Save images and poses to ngp format dataset. """
    print('Save in instant-ngp format...')
    train_dir = join(dst_dir, 'train')
    imgdir = join(dst_dir, 'train', 'images')

    if os.path.isdir(imgdir) and len(os.listdir(imgdir))>0:
        input("Warning: The existing images will be overwritten. Press enter to continue...")
        shutil.rmtree(imgdir)
    os.makedirs(imgdir, exist_ok=True)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    angle_x = 2*np.arctan(cx/fx)
    angle_y = 2*np.arctan(cy/fy)

    room_bbox = np.array(room_objs_dict['bbox'])
    scale = 1.5 / np.max(room_bbox[1] - room_bbox[0])
    cent_after_scale = scale * (room_bbox[0] + room_bbox[1])/2.0
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
            "num_room_objects": len(room_objs_dict['objects']),
			"frames": [],
            "bounding_boxes": []
		}
    
    for i, pose in enumerate(poses):
        frame = {
            "file_path": join('images/{:04d}.jpg'.format(i)),
            "transform_matrix": pose.tolist()
        }
        out['frames'].append(frame)
    
    out['bounding_boxes'] = get_ngp_type_boxes(room_objs_dict, bbox_type)

    # out['is_merge_bbox'] = 'No'
    
    with open(join(train_dir, 'transforms.json'), 'w') as f:
        json.dump(out, f, indent=4)
    
    if imgs == None: # support late rendering
        imgs = render_poses(poses, imgdir)
    
    for i, img in enumerate(imgs):
        cv2.imwrite(join(imgdir, '{:04d}.jpg'.format(i)), img)

def get_room_objs_dict(room_bbox, scene_objs_dict):
    """ Get the room object dictionary containing all the objects in the room. """
    room_objects = []
    scene_objects = scene_objs_dict['objects']
    for obj_dict in scene_objects:
        if bbox_contained(obj_dict['aabb'], room_bbox):
            room_objects.append(obj_dict)

    room_objs_dict = {}
    room_objs_dict['bbox'] = np.array(room_bbox)
    room_objs_dict['objects'] = room_objects
    
    return room_objs_dict

###########################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--scene_idx', type=int, required=True)
    parser.add_argument('-r', '--room_idx', type=int, required=True)
    parser.add_argument('--mode', type=str, choices=['plan', 'overview', 'render', 'bbox'], 
                        help="plan: Generate the floor plan of the scene. \
                              overview:Generate 4 corner overviews with bbox projected. \
                              render: Render images in the scene. \
                              bbox: Overwrite bboxes by regenerating transforms.json.")
    parser.add_argument('-ppo', '--pos_per_obj', type=int, default=15, help='Number of close-up poses for each object.')
    parser.add_argument('-gp', '--max_global_pos', type=int, default=150, help='Max number of global poses.')
    parser.add_argument('-gd', '--global_density', type=float, default=0.15, help='The radius interval of global poses. Smaller global_density -> more global views')
    parser.add_argument('-nc', '--no_check', action='store_true', default=False, help='Do not check the poses. Render directly.')
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--relabel', action='store_true', help='Relabel the objects in the scene by rewriting transforms.json.')
    parser.add_argument('--rotation', action='store_true', help = 'output rotation bounding boxes if it is true.')
    parser.add_argument('--bbox_type', type=str, default="aabb", choices=['aabb', 'obb'], help='Output aabb or obb')
    parser.add_argument('--render_root', type=str, default='./FRONT3D_render', help='Output directory. If not specified, use the default directory.')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dst_dir = join(args.render_root, '3dfront_{:04d}_{:02}'.format(args.scene_idx, args.room_idx))
    os.makedirs(dst_dir, exist_ok=True)

    construct_scene_list()

    if args.mode == 'plan':
        if args.scene_idx < 0 or args.scene_idx > 6812:
            raise ValueError("%d is not a valid scene_idx. Should provide a scene_idx between 0 and 6812 inclusively")
        os.makedirs(os.path.join(dst_dir, 'overview'), exist_ok=True)
        if args.rotation:
            floor_plan = FloorPlan_rot(args.scene_idx)
            floor_plan.drawgroups_and_save(os.path.join(dst_dir, 'overview'))

        else:
            floor_plan = FloorPlan(args.scene_idx)
            floor_plan.drawgroups_and_save(os.path.join(dst_dir, 'overview'))
        
        return
        

    cache_dir = f'./cached/{args.scene_idx}'
    if args.mode in ['overview', 'bbox'] and os.path.isfile(cache_dir + '/scene_objects_dict.npz') > 0 and \
            len(glob.glob(join(dst_dir, 'overview/raw/*'))) > 0:
        # if cached information is available & there's no need to render -> use cached scene object information
        scene_objs_dict = np.load(cache_dir + '/scene_objects_dict.npz', allow_pickle=True)
    else: 
        # load objects
        bproc.init(compute_device='cuda:0', compute_device_type=COMPUTE_DEVICE_TYPE)
        scene_objects = load_scene_objects(args.scene_idx)
        scene_objs_dict = build_and_save_scene_cache(cache_dir, scene_objects)

    room_bbox = get_room_bbox(args.scene_idx, args.room_idx, scene_objs_dict=scene_objs_dict)
    room_objs_dict = get_room_objs_dict(room_bbox, scene_objs_dict)
    room_objs_dict = filter_objs_in_dict(args.scene_idx, args.room_idx, room_objs_dict)

    if args.mode == 'overview':
        overview_dir = os.path.join(dst_dir, 'overview')
        os.makedirs(overview_dir, exist_ok=True)
        poses = generate_four_corner_poses(args.scene_idx, args.room_idx)

        cache_dir = join(dst_dir, 'overview/raw')
        cached_img_paths = glob.glob(cache_dir+'/*')
        imgs = []
        if len(cached_img_paths) > 0 and True:
            # use cached overview images
            for img_path in sorted(cached_img_paths):
                imgs.append(cv2.imread(img_path))
        else:
            # render overview images
            imgs = render_poses(poses, overview_dir)
            os.makedirs(cache_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(join(cache_dir, f'raw_{i}.jpg'), img)

        # project aabb and obb to images
        labels, coords, aabb_codes, colors = [], [], [], []
        for obj_dict in room_objs_dict['objects']:
            labels.append(obj_dict['name'])
            coords.append(np.concatenate((np.array(obj_dict['coords']), np.ones((len(np.array(obj_dict['coords'])), 1))), axis=1))
            aabb_codes.append(obj_dict['aabb'])
            colors.append(random_color())
        aabb_codes = np.array(aabb_codes).reshape(-1, 6)
        
        for i, (img, pose) in enumerate(zip(imgs, poses)):
            img_aabb = project_aabb_to_image(img, K, np.linalg.inv(pose), aabb_codes, labels, colors)
            img_obb = project_obb_to_image(img, K, np.linalg.inv(pose), coords, labels, colors)
            cv2.imwrite(os.path.join(os.path.join(dst_dir, 'overview'), 'proj_aabb_{}.png'.format(i)), img_aabb)
            cv2.imwrite(os.path.join(os.path.join(dst_dir, 'overview'), 'proj_obb_{}.png'.format(i)), img_obb)
            
        for label in sorted(labels):
            print(label)
        print(f"{len(labels)} objects in total.\n")
    
    elif args.mode == 'render':
        poses, num_closeup, num_global = generate_room_poses(args.scene_idx, args.room_idx, room_objs_dict, room_bbox, 
                                    num_poses_per_object = args.pos_per_obj,
                                    max_global_pos = args.max_global_pos,
                                    global_density=args.global_density
                                    )
        if not args.no_check:
            print('Render for scene {}, room {}:'.format(args.scene_idx, args.room_idx))
            for obj_dict in room_objs_dict['objects']:
                print(f"\t{obj_dict['aabb']}")
            print('Total poses: {}[global] + {}[closeup] x {}[object] = {} poses'.format(num_global, args.pos_per_obj, len(room_objs_dict['objects']), len(poses)))
            print('Estimated time: {} minutes'.format(len(poses)*25//60))
            input('Press Enter to continue...')

        save_in_ngp_format(None, poses, K, room_bbox, room_objs_dict, dst_dir) # late rendering
        
    elif args.mode == 'bbox':
        json_path = os.path.join(dst_dir, 'train/transforms.json')
        json_path_result = os.path.join(dst_dir, 'train/transforms.json')

        with open(json_path, 'r') as f:
            meta = json.load(f)
        
        meta['bounding_boxes'] = get_ngp_type_boxes(room_objs_dict, args.bbox_type)
        
        with open(json_path_result, 'w') as f:
            json.dump(meta, f, indent=4)



if __name__ == '__main__':
    main()
    print("Success.")