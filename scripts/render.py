# python cli.py run ./scripts/utils.py 

import blenderproc as bproc
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
<<<<<<< HEAD
from floor_plan import *
from load_helper import *

sys.path.append('/data/jhuangce/BlenderProc/scripts')
sys.path.append('/data2/jhuangce/BlenderProc/scripts')

=======
sys.path.append('/data/jhuangce/BlenderProc/scripts')
sys.path.append('/data2/jhuangce/BlenderProc/scripts')
>>>>>>> f69790c962f2061cf42e20f86c21b7f11f11cd79
from render_configs import *
from bbox_proj import project_bbox_to_image, project_obbox_to_image
import json
from typing import List
from os.path import join
import glob


pi = np.pi
cos = np.cos
sin = np.sin


def construct_scene_list():
    """ Construct a list of scenes and save to SCENE_LIST global variable. """
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        SCENE_LIST.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(SCENE_LIST)} scenes in total")





def image_to_video(img_dir, video_dir):
    """ Args: 
            img_dir: directory of images
            video_dir: directory of output video to be saved in
    """
    img_list = os.listdir(img_dir)
    img_list.sort()
    rgb_maps = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in img_list]
    print(len(rgb_maps))

    imageio.mimwrite(os.path.join(video_dir, 'video.mp4'), np.stack(rgb_maps), fps=30, quality=8)


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

############################## poses generation ##################################

def generate_four_corner_poses(scene_idx, room_idx):
    """ Return a list of matrices of 4 corner views in the room. """
    bbox_xy = ROOM_CONFIG[scene_idx][room_idx]['bbox']
    corners = [[i+0.3 for i in bbox_xy[0]], [i-0.3 for i in bbox_xy[1]]]
    x1, y1, x2, y2 = corners[0][0], corners[0][1], corners[1][0], corners[1][1]
    at = [(x1+x2)/2, (y1+y2)/2, 1.2]
    locs = [[x1, y1, 2], [x1, y2, 2], [x2, y1, 2], [x2, y2, 2]]

    c2ws = [c2w_from_loc_and_at(pos, at) for pos in locs]
    
    return c2ws

def check_pos_valid(pos, room_bbox_meta, room_bbox):
    """ Check if the position is in the room, not too close to walls and not conflicting with other objects. """
    room_bbox_small = [[item+0.5 for item in room_bbox[0]], [room_bbox[1][0]-0.5, room_bbox[1][1]-0.5, room_bbox[1][2]-0.8]] # ceiling is lower
    if not pos_in_bbox(pos, room_bbox_small):
        return False
    for obj in room_bbox_meta:
        obj_bbox = obj[1]
        if pos_in_bbox(pos, obj_bbox):
            return False

    return True

def plot_3d_point_cloud(data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.savefig('./test.png')

def generate_room_poses(scene_idx, room_idx, room_bbox_meta, room_bbox, num_poses_per_object, max_global_pos, global_density):
    pass
    """ Return a list of poses including global poses and close-up poses for each object."""

    poses = []
    num_closeup, num_global = 0, 0
    h_global = 1.2

    # close-up poses for each object.
    if num_poses_per_object>0:
        for obj in room_bbox_meta:
            obj_bbox = np.array(obj[1])
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

            positions = [pos for pos in positions if check_pos_valid(pos, room_bbox_meta, room_bbox)]
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

def get_scene_bbox(loaded_objects=None, scene_bbox_meta=None):
    """ Return the bounding box of the scene. """
    
    bbox_mins = []
    bbox_maxs = []
    if loaded_objects!=None:
        for i, object in enumerate(loaded_objects):
            bbox = object.get_bound_box()
            bbox_mins.append(np.min(bbox, axis=0))
            bbox_maxs.append(np.max(bbox, axis=0))
    elif scene_bbox_meta!=None:
        _, bbox_mins, bbox_maxs = scene_bbox_meta
    scene_min = np.min(bbox_mins, axis=0)
    scene_max = np.max(bbox_maxs, axis=0)
    return scene_min, scene_max

def get_room_bbox(scene_idx, room_idx, loaded_objects=None, scene_bbox_meta=None):
    """ Return the bounding box of the room. """
    scene_min, scene_max = get_scene_bbox(loaded_objects, scene_bbox_meta)
    room_config = ROOM_CONFIG[scene_idx][room_idx]
    scene_min[:2] = room_config['bbox'][0]
    scene_max[:2] = room_config['bbox'][1]

    return [scene_min, scene_max]

def bbox_contained(bbox_a, bbox_b):
    """ Return whether the bbox_a is contained in bbox_b. """
    return bbox_a[0][0]>=bbox_b[0][0] and bbox_a[0][1]>=bbox_b[0][1] and bbox_a[0][2]>=bbox_b[0][2] and \
           bbox_a[1][0]<=bbox_b[1][0] and bbox_a[1][1]<=bbox_b[1][1] and bbox_a[1][2]<=bbox_b[1][2]

def get_room_objects(scene_idx, room_idx, loaded_objects, cleanup=False):
    """ Return the objects within the room bbox. Cleanup unecessary objects. """
    objects = []

    room_bbox = get_room_bbox(scene_idx, room_idx, loaded_objects=loaded_objects)
    # print(room_bbox) #debug
    for object in loaded_objects:
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

def filter_bbox(scene_idx, room_idx, room_bbox_meta):
    # clean up: TODO: move to a separate function
    # Steps:
    #   1. merge
    #   2. check globa OBJ_BAN_LIST
    #   2. check keyword_ban_list
    #   3. check fullname_ban_list
    room_bbox_meta = merge_bbox(scene_idx, room_idx, room_bbox_meta)
    result_room_bbox_meta = []
    for bbox_meta in room_bbox_meta:
        flag_use = True
        obj_name = bbox_meta[0]
        for ban_word in OBJ_BAN_LIST:
            if ban_word in obj_name:
                flag_use=False
        if 'keyword_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for ban_word in ROOM_CONFIG[scene_idx][room_idx]['keyword_ban_list']:
                if ban_word in obj_name:
                    flag_use=False
        if 'fullname_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for fullname in ROOM_CONFIG[scene_idx][room_idx]['fullname_ban_list']:
                if fullname == obj_name.strip():
                    flag_use=False
        if flag_use:
            result_room_bbox_meta.append(bbox_meta)
    return result_room_bbox_meta

def reg_label(room_bbox_meta):
    ...
    label_list = ['table', 'chair', 'cabinet', 'lighting', 'nightstand']
    for label in label_list:
        if label in obj_label:
            obj_label = label
    return obj_label

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

def save_in_ngp_format(imgs, poses, intrinsic, room_bbox, room_bbox_meta, dst_dir):
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

    room_bbox = np.array(room_bbox)
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
            "num_room_objects": len(room_bbox_meta),
			"frames": [],
            "bounding_boxes": []
		}
    
    for i, pose in enumerate(poses):
        frame = {
            "file_path": join('images/{:04d}.jpg'.format(i)),
            "transform_matrix": pose.tolist()
        }
        out['frames'].append(frame)
    
    for i, obj in enumerate(room_bbox_meta):
        obj_bbox = np.array(obj[1])
        obj_bbox_ngp = {
            "extents": (obj_bbox[1]-obj_bbox[0]).tolist(),
            "orientation": np.eye(3).tolist(),
            "position": ((obj_bbox[0]+obj_bbox[1])/2.0).tolist(),
        }
        out['bounding_boxes'].append(obj_bbox_ngp)

    out['is_merge_bbox'] = 'No'
    
    with open(join(train_dir, 'transforms.json'), 'w') as f:
        json.dump(out, f, indent=4)
    
    if imgs == None: # support late rendering
        imgs = render_poses(poses, imgdir)
    
    for i, img in enumerate(imgs):
        cv2.imwrite(join(imgdir, '{:04d}.jpg'.format(i)), img)

def rewrite_labels(room_bbox_meta, dst_dir):
    """ This function rewrite transforms.json by adding labels to bboxes. """

    json_path = os.path.join(dst_dir, 'train/transforms.json')
    with open(json_path, 'r') as f:
        meta = json.load(f)

    bounding_boxes = []
    for i, obj in enumerate(room_bbox_meta):
        label = str(obj[0])
        obj_bbox = np.array(obj[1])
        obj_bbox_ngp = {
            "label": label,
            "extents": (obj_bbox[1]-obj_bbox[0]).tolist(),
            "orientation": np.eye(3).tolist(),
            "position": ((obj_bbox[0]+obj_bbox[1])/2.0).tolist(),
        }
        bounding_boxes.append(obj_bbox_ngp)
    meta['bounding_boxes'] = bounding_boxes

    with open(os.path.join(dst_dir, 'train/transforms_new.json'), 'w') as f:
        json.dump(meta, f, indent=4)

def save_in_tensorf_format(imgs, poses, room_bbox, dst_dir):
    print('Save in TensoRF format...')
    from os.path import join
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    rgbdir = join(dst_dir, 'rgb')
    posedir = join(dst_dir, 'pose')
    os.mkdir(dst_dir)
    os.mkdir(rgbdir)
    os.mkdir(posedir)
    
    with open(join(dst_dir, 'intrinsic.txt'), 'w') as f:
        for line in K:
            f.write('%.1f %.1f %.1f\n' % (line[0], line[1], line[2]))
    with open(join(dst_dir, 'bbox.txt'), 'w') as f:
        f.write('{} {} {} {} {} {} 0.01\n'.format(*room_bbox[0], *room_bbox[1]))

    for i, (img, pose) in enumerate(zip(imgs, poses)):
        name = 'img%03d' % i
        cv2.imwrite(join(rgbdir, name+'.jpg'), img)
        with open(join(posedir, name+'.txt'), 'w') as f:
            for line in pose:
                f.write('%f %f %f %f\n' % (line[0], line[1], line[2], line[3]))
    
    # Writing to .json
    frames = []
    for i in range(len(poses)):
        frames += [{"file_path": "rgb/img%03d.jpg" % i, "transform_matrix": poses[i].tolist()}]
    train_frames = frames[:-6]
    test_frames = frames[-8:] # 2 global poses in train set, 2 global poses not in train set, 4 corner poses
 
    with open(os.path.join(dst_dir, 'transforms_train.json'), 'w') as f:
        f.write(json.dumps({"frames": train_frames}, indent=4))
    with open(os.path.join(dst_dir, 'transforms_val.json'), 'w') as f:
        f.write(json.dumps({"frames": test_frames}, indent=4))
    with open(os.path.join(dst_dir, 'transforms_test.json'), 'w') as f:
        f.write(json.dumps({"frames": test_frames}, indent=4))


###########################################################################################


if __name__ == '__main__':

    """
        Example commands:

            python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --plan
            python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --overview 
            python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render
            python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render -ppo 10 -gd 0.15
            python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render -ppo 0 -gp 5
            python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --render -nc
        
    """

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--scene_idx', type=int, required=True)
    parser.add_argument('-r', '--room_idx', type=int, required=True)
    parser.add_argument('--plan', action='store_true', help='Generate the floor plan of the scene.')
    parser.add_argument('--overview', action='store_true', help='Generate 4 corner overviews with bbox projected.')
    parser.add_argument('--render', action='store_true', help='Render images in the scene')
    parser.add_argument('--check_obb', action='store_true', help='Check how to generate obb')
    parser.add_argument('--yichen', action='store_true', help='Overwrite bboxes by regenerating transforms.json.')
    parser.add_argument('-ppo', '--pos_per_obj', type=int, default=15, help='Number of close-up poses for each object.')
    parser.add_argument('-gp', '--max_global_pos', type=int, default=150, help='Max number of global poses.')
    parser.add_argument('-gd', '--global_density', type=float, default=0.15, help='The radius interval of global poses. Smaller global_density -> more global views')
    parser.add_argument('-nc', '--no_check', action='store_true', default=False, help='Do not check the poses. Render directly.')
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--relabel', action='store_true', help='Relabel the objects in the scene by rewriting transforms.json.')
    parser.add_argument('--rotation', action='store_true', help = 'output rotation bounding boxes if it is true.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # dst_dir = '/data/yliugu/BlenderProc/FRONT3D_render/3dfront_{:04d}_{:02}'.format(args.scene_idx, args.room_idx)

    # dst_dir = '/data/yliugu/BlenderProc/FRONT3D_render/3dfront_{:04d}_{:02}'.format(args.scene_idx, args.room_idx)
    dst_dir = './FRONT3D_render/3dfront_{:04d}_{:02}'.format(args.scene_idx, args.room_idx)

    os.makedirs(dst_dir, exist_ok=True)

    construct_scene_list()

    if args.plan:
        if args.scene_idx < 0 or args.scene_idx > 6812:
            raise ValueError("%d is not a valid scene_idx. Should provide a scene_idx between 0 and 6812 inclusively")
        os.makedirs(os.path.join(dst_dir, 'overview'), exist_ok=True)
        if args.rotation:
            floor_plan = FloorPlan_rot(args.scene_idx)
            floor_plan.drawgroups_and_save(os.path.join(dst_dir, 'overview'))

        else:
            floor_plan = FloorPlan(args.scene_idx)
            floor_plan.drawgroups_and_save(os.path.join(dst_dir, 'overview'))
    if args.overview and args.render:
        raise ValueError("Error: Cannot render overview and rendering at the same time. ")

    if args.overview or args.render:
        cache_dir = f'./cached/{args.scene_idx}'

        if args.rotation:
            # TODO: rotation render
            
            pass
        else:
            if args.overview and os.path.isfile(cache_dir + '/bboxes_mins.npy') and len(glob.glob(join(dst_dir, 'overview/raw/*'))) > 0:
                # use cached object information and overview images if available
                names = np.load(cache_dir + '/names.npy')
                bbox_maxs = np.load(cache_dir + '/bbox_maxs.npy')
                bbox_mins = np.load(cache_dir + '/bbox_mins.npy')
                room_bbox = get_room_bbox(args.scene_idx, args.room_idx, scene_bbox_meta=[names, bbox_mins, bbox_maxs])
                room_bbox_meta = []
                for i in range(len(names)):
                    if bbox_contained([bbox_mins[i], bbox_maxs[i]], room_bbox):
                        room_bbox_meta.append((names[i], [bbox_mins[i], bbox_maxs[i]]))
            else: 
                # init and load objects to blenderproc
                bproc.init(compute_device='cuda:0', compute_device_type='CUDA')
                loaded_objects = load_scene_objects(args.scene_idx)
                room_objects = get_room_objects(args.scene_idx, args.room_idx, loaded_objects)
                room_bbox = get_room_bbox(args.scene_idx, args.room_idx, loaded_objects=loaded_objects)
                room_bbox_meta = []
                for obj in room_objects:
                    obj_bbox_8 = obj.get_bound_box()
                    obj_bbox = np.array([np.min(obj_bbox_8, axis=0), np.max(obj_bbox_8, axis=0)])
                    room_bbox_meta.append((obj.get_name(), obj_bbox))
        room_bbox_meta = filter_bbox(args.scene_idx, args.room_idx, room_bbox_meta)

    if args.overview:
        overview_dir = os.path.join(dst_dir, 'overview')
        os.makedirs(overview_dir, exist_ok=True)
        poses = generate_four_corner_poses(args.scene_idx, args.room_idx)

        cache_dir = join(dst_dir, 'overview/raw')
        cached_img_paths = glob.glob(cache_dir+'/*')
        imgs = []
        if len(cached_img_paths) > 0 and True:
            # use cached overview images if available
            for img_path in sorted(cached_img_paths):
                imgs.append(cv2.imread(img_path))
        else:
            # render overview images
            imgs = render_poses(poses, overview_dir)
            os.makedirs(cache_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(join(cache_dir, f'raw_{i}.jpg'), img)

        # generate bounding box images
        aabb_codes, labels, colors = [], [], []
        for obj in room_bbox_meta:
            obj_bbox = obj[1]
            aabb_codes.append(np.concatenate([obj_bbox[0], obj_bbox[1]], axis=0))
            labels.append(obj[0])
            color = np.random.choice(range(256), size=3)
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        
        imgs_projected = []
        for img, pose in zip(imgs, poses):
            imgs_projected.append(project_bbox_to_image(img, K, np.linalg.inv(pose), aabb_codes, labels, colors))

        for i, img in enumerate(imgs_projected):
            cv2.imwrite(os.path.join(os.path.join(dst_dir, 'overview'), 'proj_{}.png'.format(i)), img)
        
        for label in sorted(labels):
            print(label)
        print(f"{len(labels)} objects in total.\n")

        if args.relabel:
            rewrite_labels(room_bbox_meta, dst_dir)

    if args.render:
        poses, num_closeup, num_global = generate_room_poses(args.scene_idx, args.room_idx, room_bbox_meta, room_bbox, 
                                    num_poses_per_object = args.pos_per_obj,
                                    max_global_pos = args.max_global_pos,
                                    global_density=args.global_density
                                    )
        if not args.no_check:
            print('Render for scene {}, room {}:'.format(args.scene_idx, args.room_idx))
            for obj in room_bbox_meta:
                print(f"\t{obj[1]}")
            print('Total poses: {}[global] + {}[closeup] x {}[object] = {} poses'.format(num_global, args.pos_per_obj, len(room_bbox_meta), len(poses)))
            print('Estimated time: {} minutes'.format(len(poses)*25//60))
            input('Press Enter to continue...')

        save_in_ngp_format(None, poses, K, room_bbox, room_bbox_meta, dst_dir) # late rendering
    
    if args.check_obb:
        bproc.init(compute_device='cuda:0', compute_device_type=COMPUTE_DEVICE_TYPE)
        loaded_objects = load_scene_objects(args.scene_idx)
        room_objects = get_room_objects(args.scene_idx, args.room_idx, loaded_objects)

        overview_dir = os.path.join(dst_dir, 'overview')
        os.makedirs(overview_dir, exist_ok=True)

        cache_dir = join(dst_dir, 'overview/raw')
        cached_img_paths = glob.glob(cache_dir+'/*')
        imgs = []
        if len(cached_img_paths) > 0 and True:
            # use cached overview images if available
            for img_path in sorted(cached_img_paths):
                imgs.append(cv2.imread(img_path))
        else:
            # render overview images
            imgs = render_poses(poses, overview_dir)
            os.makedirs(cache_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(join(cache_dir, f'raw_{i}.jpg'), img)
        
        # generate bounding box images
        poses = generate_four_corner_poses(args.scene_idx, args.room_idx)
        obboxes, labels, colors = [], [], []
        for obj in room_objects:
            obbox = obj.get_bound_box() 
            obboxes.append(np.concatenate((obbox, np.ones((len(obbox), 1))), 1))
            labels.append(obj.get_name())
            color = np.random.choice(range(256), size=3)
            colors.append((int(color[0]), int(color[1]), int(color[2])))

        imgs_projected = []
        for img, pose in zip(imgs, poses):
            imgs_projected.append(project_obbox_to_image(img, K, np.linalg.inv(pose), obboxes, labels, colors))

        for i, img in enumerate(imgs_projected):
            cv2.imwrite(os.path.join(os.path.join(dst_dir, 'overview'), 'proj_obb_{}.png'.format(i)), img)
        

    if args.yichen:
        json_path = os.path.join(dst_dir, 'train/transforms.json')
        json_path_result = os.path.join(dst_dir, 'train/transforms.json')
        
        cache_dir = f'./cached/{args.scene_idx}'
        if args.overview and os.path.isfile(cache_dir + '/names.npy') and len(glob.glob(join(dst_dir, 'overview/raw/*'))) > 0:
            # use cached object information and overview images if available
            names = np.load(cache_dir + '/names.npy')
            bbox_maxs = np.load(cache_dir + '/bbox_maxs.npy')
            bbox_mins = np.load(cache_dir + '/bbox_mins.npy')
            room_bbox = get_room_bbox(args.scene_idx, args.room_idx, scene_bbox_meta=[names, bbox_mins, bbox_maxs])
            room_bbox_meta = []
            for i in range(len(names)):
                if bbox_contained([bbox_mins[i], bbox_maxs[i]], room_bbox):
                    room_bbox_meta.append((names[i], [bbox_mins[i], bbox_maxs[i]]))
        else: 
            # init and load objects to blenderproc
            bproc.init(compute_device='cuda:0', compute_device_type=COMPUTE_DEVICE_TYPE)
            loaded_objects = load_scene_objects(args.scene_idx)
            room_objects = get_room_objects(args.scene_idx, args.room_idx, loaded_objects)
            room_bbox = get_room_bbox(args.scene_idx, args.room_idx, loaded_objects=loaded_objects)
            room_bbox_meta = []
            for obj in room_objects:
                obj_bbox_8 = obj.get_bound_box()
                obj_bbox = np.array([np.min(obj_bbox_8, axis=0), np.max(obj_bbox_8, axis=0)])
                room_bbox_meta.append((obj.get_name(), obj_bbox))
        room_bbox_meta = filter_bbox(args.scene_idx, args.room_idx, room_bbox_meta)

        with open(json_path, 'r') as f:
            meta = json.load(f)
        
        bounding_boxes = []
        for i, obj in enumerate(room_bbox_meta):
            obj_bbox = np.array(obj[1])
            obj_bbox_ngp = {
                "extents": (obj_bbox[1]-obj_bbox[0]).tolist(),
                "orientation": np.eye(3).tolist(),
                "position": ((obj_bbox[0]+obj_bbox[1])/2.0).tolist(),
            }
            bounding_boxes.append(obj_bbox_ngp)
        meta['bounding_boxes'] = bounding_boxes
        
        with open(json_path_result, 'w') as f:
            json.dump(meta, f, indent=4)


    print("Success.")