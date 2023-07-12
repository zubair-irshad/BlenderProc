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

from render import parse_args, construct_scene_list, COMPUTE_DEVICE_TYPE, get_room_bbox, \
    get_room_objs_dict, filter_objs_in_dict

def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dst_dir = join(args.render_root, '3dfront_{:04d}_{:02}'.format(args.scene_idx, args.room_idx))
    os.makedirs(dst_dir, exist_ok=True)

    construct_scene_list()
    
    cache_dir = f'./cached/{args.scene_idx}'
    bproc.init(compute_device='cuda:0', compute_device_type=COMPUTE_DEVICE_TYPE)
    scene_objects = load_scene_objects(args.scene_idx)
    scene_objs_dict = build_and_save_scene_cache(cache_dir, scene_objects)
    
    room_bbox = get_room_bbox(args.scene_idx, args.room_idx, scene_objs_dict=scene_objs_dict)
    room_objs_dict = get_room_objs_dict(room_bbox, scene_objs_dict)
    room_objs_dict = filter_objs_in_dict(args.scene_idx, args.room_idx, room_objs_dict)
    
    poses_file = os.path.join(dst_dir, 'train', 'transforms.json')
    poses = []
    with open(poses_file) as f:
        data = json.load(f)
        for frame in data['frames']:
            pose = np.array(frame['transform_matrix'])
            poses.append(pose)
    
    cache_dir = join(dst_dir, 'segmentation')
    imgs = []

    # render overview images
    os.makedirs(cache_dir, exist_ok=True)
    imgs = render_poses(poses, cache_dir)
    bproc.writer.write_hdf5(cache_dir, imgs)

if __name__ == '__main__':
    main()
    print("Success.")