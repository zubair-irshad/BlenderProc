# python cli.py run ./scripts/utils.py 

import blenderproc as bproc
from random import shuffle
import shutil
from blenderproc.python.types.MeshObjectUtility import MeshObject
import bpy
import sys
sys.path.append('/home/jhuangce/miniconda3/lib/python3.9/site-packages')
import cv2
import os
from os.path import join
import numpy as np
import re
import imageio
import sys
sys.path.append('/data2/jhuangce/BlenderProc/scripts')
from camera import *
import json
from typing import List
from bbox_proj import project_bbox_to_image



pi = np.pi
cos = np.cos
sin = np.sin
LAYOUT_DIR = '/data2/jhuangce/3D-FRONT'
TEXTURE_DIR = '/data2/jhuangce/3D-FRONT-texture'
MODEL_DIR = '/data2/jhuangce/3D-FUTURE-model'
RENDER_TEMP_DIR = '/data2/jhuangce/BlenderProc/FRONT3D_render'
SCENE_LIST = []
OBJ_BAN_LIST = ['Baseboard', 'Pocket', 'Floor', 'SlabSide.', 'WallInner', 'Front', 'WallTop', 'WallBottom', 'Ceiling.', 'Nightstand.001', 'Nightstand.003', 'Ceiling Lamp']

def construct_scene_list():
    """ Construct a list of scenes and save to SCENE_LIST global variable. """
    layout_list = [join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)]
    layout_list.sort()
    for scene_code in layout_list:
        SCENE_LIST.append(scene_code)


def check_cache_dir(scene_idx):
    if not os.path.isdir(f'./cached/{scene_idx}'):
        os.mkdir(f'./cached/{scene_idx}')


def add_texture(obj:MeshObject, tex_path):
    """ Add a texture to an object. 
        Args:
            obj: MeshObject
            tex_path: path to the texture
    """
    obj.clear_materials()
    mat = obj.new_material('my_material')
    bsdf = mat.nodes["Principled BSDF"]
    texImage = mat.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(tex_path)
    mat.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])


def load_scene_objects(scene_idx, overwrite=False):
    check_cache_dir(scene_idx)
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    loaded_objects = bproc.loader.load_front3d(
        json_path=SCENE_LIST[scene_idx],
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        ceiling_light_strength=1,
        lamp_light_strength=30
    )

    # add texture to wall and floor. Otherwise they will be white.
    for obj in loaded_objects:
        name = obj.get_name()
        if 'wall' in name.lower():
            add_texture(obj, "/data2/jhuangce/3D-FRONT-texture/1b57700d-f41b-4ac7-a31a-870544c3d608/texture.png")
        elif 'floor' in name.lower():
            add_texture(obj, "/data2/jhuangce/3D-FRONT-texture/0b48b46d-4f0b-418d-bde6-30ca302288e6/texture.png")
        # elif 'ceil' in name.lower():
        #     add_texture(obj, "/data2/jhuangce/3D-FRONT-texture/0a5adcc7-f17f-488f-9f95-8690cbc31321/texture.png")

    return loaded_objects

def get_cameras_in_oval_trajectory(scene_idx, room_idx = None):
    """ Generate the camera locations and rotations in a room according to oval trajectory. """
    locations, rotations = [], []
    config_dict = ROOM_CONFIG[scene_idx]
    for key, value in config_dict.items():
        if room_idx!=None and key!=room_idx:
            continue
        # the trajectory is an ellipse
        center = value['center'] # (3, 2.2)
        a = value['a'] # 1.5
        b = value['b'] # 2.2
        num = value['num_cam'] # 2
        for i in range(num):
            theta = 2*np.pi*i/num
            location = [a*np.cos(theta)+center[0], b*np.sin(theta)+center[1], 1.2]
            high_location = [a*np.cos(theta)+center[0], b*np.sin(theta)+center[1], 2]
            rot_root = [1.4, 0, np.arcsin(-b*np.cos(theta)/np.sqrt((a*np.sin(theta))**2+(b*np.cos(theta))**2))-np.pi]
            if location[1]<center[1]:
                rot_root[2] = 3*np.pi-rot_root[2]
            locations += 4 * [location]
            locations += [high_location]
            deg = np.pi / 6
            rotations += [rot_root]
            rotations += [[rot_root[0], rot_root[1], rot_root[2]-deg]]
            rotations += [[rot_root[0], rot_root[1], rot_root[2]+deg]]
            rotations += [[rot_root[0]+deg, rot_root[1], rot_root[2]]]
            rotations += [[rot_root[0]-deg, rot_root[1], rot_root[2]]]

    # print(locations) debug
    # print(rotations) debug
    return locations, rotations

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
class FloorPlan():
    def __init__(self, scene_idx):
        self.scene_idx = scene_idx
        self.names, self.bbox_mins, self.bbox_maxs = self.get_scene_information(scene_idx)

        self.scene_min = np.min(self.bbox_mins, axis=0)
        self.scene_max = np.max(self.bbox_maxs, axis=0)
        print('scene_min:', self.scene_min)
        print('scene_max', self.scene_max)

        self.scale = 200
        self.margin = 100

        self.width = int((self.scene_max-self.scene_min)[0]*self.scale)+self.margin*2
        self.height = int((self.scene_max-self.scene_min)[1]*self.scale)+self.margin*2

        self.image = np.ones((self.height,self.width,3), np.uint8)
    
    def get_scene_information(self, scene_idx, overwrite=False):
        check_cache_dir(scene_idx)
        if os.path.isfile('./cached/%d/names.npy' % scene_idx) and overwrite==False:
            print(f'Found cached information for scene {scene_idx}.')
            names = np.load(f'./cached/{scene_idx}/names.npy')
            bbox_mins = np.load(f'./cached/{scene_idx}/bbox_mins.npy')
            bbox_maxs = np.load(f'./cached/{scene_idx}/bbox_maxs.npy')
        else:
            loaded_objects = load_scene_objects(scene_idx, overwrite)
            names = []
            bbox_mins = []
            bbox_maxs = []
            for i in range(len(loaded_objects)):
                object = loaded_objects[i]
                name = object.get_name()
                bbox = object.get_bound_box()
                bbox_min = np.min(bbox, axis=0)
                bbox_max = np.max(bbox, axis=0)
                names.append(name)
                bbox_mins.append(bbox_min)
                bbox_maxs.append(bbox_max)

            np.save(f'./cached/{scene_idx}/names.npy', names)
            np.save(f'./cached/{scene_idx}/bbox_mins.npy', bbox_mins)
            np.save(f'./cached/{scene_idx}/bbox_maxs.npy', bbox_maxs)

        return names, bbox_mins, bbox_maxs
    
    def draw_samples(self, locs=None, rots=None):
        if locs == None:
            if self.scene_idx not in CAMERA_LOCS.keys():
                return
            for pos in CAMERA_LOCS[self.scene_idx]:
                cv2.circle(self.image, self.point_to_image(pos), radius=25, color=(0,255,0), thickness=3)
                cv2.putText(self.image, 'camera', self.point_to_image(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
        else:
            if rots!= None:
                for pos, rots in zip(locs, rots):
                    cv2.circle(self.image, self.point_to_image(pos), radius=25, color=(0,255,0), thickness=3)
                    point = [pos[0]+np.cos(rots[2]+np.pi/2)/10, pos[1]+np.sin(rots[2]+np.pi/2)/10]
                    cv2.line(self.image, self.point_to_image(pos), self.point_to_image(point), color=(0,255,0), thickness=3)
            else:
                for pos in locs:
                    cv2.circle(self.image, self.point_to_image(pos), radius=25, color=(0,255,0), thickness=3)

    def draw_coords(self):
        
        seg = 0.08
        x0, y0 = self.point_to_image([0,0,0])
        cv2.line(self.image, (0,y0), (self.width-1, y0), color=red, thickness=3)
        cv2.line(self.image, (x0,0), (x0, self.height-1), color=red, thickness=3)
        
        for i in range(int(np.floor(self.scene_min[0])), int(np.ceil(self.scene_max[0])+1)):
            cv2.line(self.image, self.point_to_image([i, -seg]), self.point_to_image([i, seg]), color=red, thickness=2)
        for i in range(int(np.floor(self.scene_min[1])), int(np.ceil(self.scene_max[1])+1)):
            cv2.line(self.image, self.point_to_image([-seg, i]), self.point_to_image([seg, i]), color=red, thickness=2)
        
        cv2.putText(self.image, 'x+', (self.width-80, y0-20), cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)
        cv2.putText(self.image, 'y+', (x0+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)
    
    def draw_room_bbox(self):
        for value in ROOM_CONFIG[self.scene_idx].values():
            scene_bbox = value['bbox']
            cv2.rectangle(self.image, self.point_to_image(scene_bbox[0]), self.point_to_image(scene_bbox[1]), color=blue, thickness=5)
    
    def draw_objects(self):
        for i in range(len(self.names)):
            x1, y1 = self.point_to_image(self.bbox_mins[i])
            x2, y2 = self.point_to_image(self.bbox_maxs[i])
            color = np.random.randint(0, 255, size=3)
            color = (int(color[0]), int(color[1]), int(color[2]))

            if self.names[i][:4] == 'Wall':
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 255, 0), -1)
            elif self.names[i][:5] == 'Floor':
                pass
            else:
                cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.image, self.names[i], (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
    
    def point_to_image(self, point_3d):
        """ Args: \\
                point_3d: raw float 3D point [x, y, z]
        """
        return int(point_3d[0]*self.scale - self.scene_min[0]*self.scale + self.margin), self.height-int(point_3d[1]*self.scale - self.scene_min[1]*self.scale + self.margin)
    
    def save(self, file_name):
        cv2.imwrite(f'./cached/{self.scene_idx}/{file_name}', self.image)
    
    def drawsamples_and_save(self):
        self.draw_objects()
        self.draw_coords()
        self.draw_samples() # customizable
        self.save('floor_plan.jpg')
    
    def drawgroups_and_save(self):
        locs, rots = get_cameras_in_oval_trajectory(self.scene_idx)
        self.draw_objects()
        self.draw_coords()
        self.draw_samples(locs, rots) # customizable
        self.draw_room_bbox()
        self.save('floor_plan2.jpg')

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

def render_sample(scene_idx, device):
    """ Each camera position render 4 images. """
    bproc.init(compute_device=device, compute_device_type='CUDA')
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

    loaded_objects = load_scene_objects(scene_idx)

    bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

    for xy in CAMERA_LOCS[scene_idx]:
        for i in range(4):
            location = np.array([xy[0], xy[1], 1.5])
            rotation = [1.6, 0, 2*np.pi*i/4]
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
            bproc.camera.add_camera_pose(cam2world_matrix)
            
    
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)

    data = bproc.renderer.render(output_dir='./output')

    if not os.path.isdir('./output/%04d' % scene_idx):
        os.mkdir('./output/%04d' % scene_idx)
    for i in range(len(data['colors'])):
        im_rgb = cv2.cvtColor(data['colors'][i], cv2.COLOR_BGR2RGB)
        cv2.imwrite('./output/%04d/img_%02d.jpg' % (scene_idx, i), im_rgb)

def render_room(scene_idx, room_idx, device):
    """ Each camera position render 8 images. """
    bproc.init(compute_device=device, compute_device_type='CUDA')
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

    loaded_objects = load_scene_objects(scene_idx)
    bbox_mins = []
    bbox_maxs = []
    for i in range(len(loaded_objects)):
        object = loaded_objects[i]
        bbox = object.get_bound_box()
        bbox_min = np.min(bbox, axis=0)
        bbox_max = np.max(bbox, axis=0)
        bbox_mins.append(bbox_min)
        bbox_maxs.append(bbox_max)
    scene_min = np.min(bbox_mins, axis=0)
    scene_max = np.max(bbox_maxs, axis=0)

    bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])
    
    room_config = ROOM_CONFIG[scene_idx][room_idx]
    scene_min[:2] = room_config['bbox'][0]
    scene_max[:2] = room_config['bbox'][1]

    poses = []
    locs, rots = get_cameras_in_oval_trajectory(scene_idx, room_idx)
    for loc, rot in zip(locs, rots):
        cam2world_matrix = bproc.math.build_transformation_mat(loc, rot)
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses.append(np.array(cam2world_matrix).tolist())
    
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)

    from os.path import join
    out_root = './FRONT3D_render/'
    outdir = out_root+'%03d_%d' % (scene_idx, room_idx)
    rgbdir = join(outdir, 'rgb')
    posedir = join(outdir, 'pose')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    os.mkdir(rgbdir)
    os.mkdir(posedir)
    
    with open(join(outdir, 'intrinsic.txt'), 'w') as f:
        for line in K:
            f.write('%.1f %.1f %.1f\n' % (line[0], line[1], line[2]))
    with open(join(outdir, 'bbox.txt'), 'w') as f:
        f.write(f'{scene_min[0]} {scene_min[1]} {scene_min[2]} {scene_max[0]} {scene_max[1]} {scene_max[2]} 0.01\n')

    data = bproc.renderer.render(output_dir=out_root)
    for i in range(len(data['colors'])):
        im_rgb = cv2.cvtColor(data['colors'][i], cv2.COLOR_BGR2RGB)
        name = 'img%03d' % i
        cv2.imwrite(join(rgbdir, name+'.jpg'), im_rgb)
        with open(join(posedir, name+'.txt'), 'w') as f:
            for line in poses[i]:
                f.write('%f %f %f %f\n' % (line[0], line[1], line[2], line[3]))
    
    # Writing to .json
    frames = []
    for i in range(len(poses)):
        frames += [{"file_path": "rgb/img%03d.jpg" % i, "transform_matrix": poses[i]}]
    
    import random
    randIndex = random.sample(range(len(frames)), 6)
    randIndex.sort()
    train_frames, test_frames = [], []
    for i in range(len(frames)):
        if i in randIndex:
            test_frames += [frames[i]]
        else:
            train_frames += [frames[i]]
 
    with open(os.path.join(outdir, 'transforms_train.json'), 'w') as f:
        f.write(json.dumps({"frames": train_frames}, indent=4))
    with open(os.path.join(outdir, 'transforms_val.json'), 'w') as f:
        f.write(json.dumps({"frames": test_frames}, indent=4))
    with open(os.path.join(outdir, 'transforms_test.json'), 'w') as f:
        f.write(json.dumps({"frames": test_frames}, indent=4))

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

    # c2ws = []
    # for cam_pos in locs:
    #     c2w = np.eye(4)
    #     cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
    #     c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
    #     c2ws.append(c2w)
    # return c2ws

def pos_in_bbox(pos, bbox):
    """
    Check if a point is inside a bounding box.
    Input:
        pos: 3 x 1
        bbox: 2 x 3
    Output:
        True or False
    """
    return np.all(pos >= bbox[0]) and np.all(pos <= bbox[1])

############################## poses generation ##################################

def generate_four_corner_poses(scene_idx, room_idx):
    """ Return a list of matrices of 4 corner views in the room. """
    corners = ROOM_CONFIG[scene_idx][room_idx]['corners']
    x1, y1, x2, y2 = corners[0][0], corners[0][1], corners[1][0], corners[1][1]
    at = [(x1+x2)/2, (y1+y2)/2, 1.2]
    locs = [[x1, y1, 2], [x1, y2, 2], [x2, y1, 2], [x2, y2, 2]]

    c2ws = [c2w_from_loc_and_at(pos, at) for pos in locs]
    
    return c2ws

def check_pos_valid(pos, room_objects, room_bbox):
    """ Check if the position is in the room, not too close to walls and not conflicting with other objects. """
    room_bbox_small = [[item+0.5 for item in room_bbox[0]], [room_bbox[1][0]-0.5, room_bbox[1][1]-0.5, room_bbox[1][2]-0.8]] # ceiling is lower
    if not pos_in_bbox(pos, room_bbox_small):
        return False
    for obj in room_objects:
        obj_bbox_8 = obj.get_bound_box()
        obj_bbox = [np.min(obj_bbox_8, axis=0), np.max(obj_bbox_8, axis=0)]
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

def generate_room_poses(scene_idx, room_idx, room_objects, room_bbox, num_poses_per_object=8, num_poses_global=50):
    pass
    """ Return a list of poses including global poses and close-up poses for each object."""

    poses = []
    num_closeup, num_global = 0, 0
    h_global = 1.2

    # close-up poses for each object.
    if num_poses_per_object>0:
        for obj in room_objects:
            obj_bbox_8 = obj.get_bound_box()
            obj_bbox = [np.min(obj_bbox_8, axis=0), np.max(obj_bbox_8, axis=0)]
            cent = np.mean(obj_bbox_8, axis=0)
            rad = np.linalg.norm(obj_bbox[1]-obj_bbox[0])/2 * 1.7 # how close the camera is to the object
            if np.max(obj_bbox[1]-obj_bbox[0])<1:
                rad *= 1.6 # handle small objects

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

            positions = [pos for pos in positions if check_pos_valid(pos, room_objects, room_bbox)]
            shuffle(positions)
            if len(positions) > num_poses_per_object:
                positions = positions[:num_poses_per_object]

            poses.extend([c2w_from_loc_and_at(pos, cent) for pos in positions])

            num_closeup = len(positions)

    # global poses
    if num_poses_global>0:
        bbox = ROOM_CONFIG[scene_idx][room_idx]['bbox']
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
        rm_cent = np.array([(x1+x2)/2, (y1+y2)/2, h_global])

        # # sphere model
        # positions = []
        # rad_bound = [0.8, 5]
        # theta_bound = [0, 2*pi]
        # phi_bound = [-pi/12, pi/8]
        # n_try = 100000
        # for i in range(n_try):
        #     rad = np.random.uniform(rad_bound[0], rad_bound[1])
        #     theta = np.random.uniform(theta_bound[0], theta_bound[1])
        #     phi = np.random.uniform(phi_bound[0], phi_bound[1])
        #     pos = [rad * cos(phi)*cos(theta), rad * cos(phi)*sin(theta), rad * sin(phi)] + rm_cent # in position
        #     if check_pos_valid(pos, room_objects, room_bbox):
        #         positions.append(pos)

        #     if len(positions) >= num_poses_global:
        #         break
        #     elif len(positions) >= n_try:
        #         raise Exception("Cannot generate enough global images, check room configurations")
        # positions = np.array(positions)

        # poses.extend([c2w_from_loc_and_at(pos, rm_cent) for pos in positions])

        # cylinder model
        # positions = []
        # rad_bound = [1, 5]
        # theta_bound = [0, 2*pi]
        # h_bound = [0.8, 2.5]
        # view_at_height_bound = [1, 2]
        # n_try = 100000
        # for i in range(n_try):
        #     rad = np.random.uniform(rad_bound[0], rad_bound[1])
        #     theta = np.random.uniform(theta_bound[0], theta_bound[1])
        #     h = np.random.uniform(h_bound[0], h_bound[1])
        #     pos = [rad * cos(theta), rad * sin(theta), h] + rm_cent # in position
        #     if check_pos_valid(pos, room_objects, room_bbox):
        #         positions.append(pos)

        #     if len(positions) >= num_poses_global:
        #         break
        #     elif len(positions) >= n_try:
        #         raise Exception("Cannot generate enough global images, check room configurations")
        # positions = np.array(positions)

        # poses.extend([c2w_from_loc_and_at(pos, [rm_cent[0], rm_cent[1], np.random.uniform(view_at_height_bound[0], view_at_height_bound[1])]) for pos in positions])

        # flower model
        
        rad_bound = [0.3, 5]
        rad_intv = 0.25
        theta_bound = [0, 2*pi]
        theta_sects = 15
        theta_intv = (theta_bound[1] - theta_bound[0]) / theta_sects
        h_bound = [0.8, 2.0]

        positions = []
        theta = theta_bound[0]
        for i in range(theta_sects):
            rad = rad_bound[0]
            while rad < rad_bound[1]:
                h = np.random.uniform(h_bound[0], h_bound[1])
                pos = [rm_cent[0] + rad * cos(theta), rm_cent[1] + rad * sin(theta), h]
                if check_pos_valid(pos, room_objects, room_bbox):
                    positions.append(pos)
                rad += rad_intv
            theta += theta_intv
        positions = np.array(positions)

        poses.extend([c2w_from_loc_and_at(pos, [rm_cent[0], rm_cent[1], pos[2]]) for pos in positions])
        

    return poses


#################################################################################

def get_scene_bbox(loaded_objects):
    """ Return the bounding box of the scene. """
    bbox_mins = []
    bbox_maxs = []
    for i in range(len(loaded_objects)):
        object = loaded_objects[i]
        bbox = object.get_bound_box()
        bbox_min = np.min(bbox, axis=0)
        bbox_max = np.max(bbox, axis=0)
        bbox_mins.append(bbox_min)
        bbox_maxs.append(bbox_max)
    scene_min = np.min(bbox_mins, axis=0)
    scene_max = np.max(bbox_maxs, axis=0)
    return scene_min, scene_max

def get_room_bbox(scene_idx, room_idx, loaded_objects):
    """ Return the bounding box of the room. """
    scene_min, scene_max = get_scene_bbox(loaded_objects)
    room_config = ROOM_CONFIG[scene_idx][room_idx]
    scene_min[:2] = room_config['bbox'][0]
    scene_max[:2] = room_config['bbox'][1]

    return [scene_min, scene_max]

def bbox_contained(bbox_a, bbox_b):
    """ Return whether the bbox_a is contained in bbox_b. """
    return bbox_a[0][0]>=bbox_b[0][0] and bbox_a[0][1]>=bbox_b[0][1] and bbox_a[0][2]>=bbox_b[0][2] and \
           bbox_a[1][0]<=bbox_b[1][0] and bbox_a[1][1]<=bbox_b[1][1] and bbox_a[1][2]<=bbox_b[1][2]

def get_room_objects(scene_idx, room_idx, loaded_objects, cleanup=True):
    """ Return the objects within the room bbox. Cleanup unecessary objects. """
    objects = []

    room_bbox = get_room_bbox(scene_idx, room_idx, loaded_objects)
    # print(room_bbox) #debug
    for object in loaded_objects:
        obj_bbox_8 = object.get_bound_box()
        obj_bbox = [np.min(obj_bbox_8, axis=0), np.max(obj_bbox_8, axis=0)]
        if bbox_contained(obj_bbox, room_bbox):
            flag_use = True
            if cleanup:
                obj_name = object.get_name()
                for ban_word in OBJ_BAN_LIST:
                    if ban_word in obj_name:
                        flag_use=False 
            if flag_use:
                objects.append(object)

    return objects


def render_poses(poses) -> List:
    """ Render a scene with a list of poses. 
        No room idx is needed because the poses can be anywhere in the room. """

    # add camera poses to render queue
    for cam2world_matrix in poses:
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    # render
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200, transmission_bounces=200, transparent_max_bounces=200)
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)
    data = bproc.renderer.render(output_dir=RENDER_TEMP_DIR)
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data['colors']]

    return imgs

##################################### save to dataset #####################################

def save_in_ngp_format(imgs, poses, room_bbox, intrinsic, dst_dir):
    """ Save images and poses to ngp format dataset. """
    print('Save in TensoRF format...')
    from os.path import join
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    imgdir = join(dst_dir, 'images')
    os.mkdir(dst_dir)
    os.mkdir(imgdir)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    angle_x = 2*np.arctan(cx/fx)
    angle_y = 2*np.arctan(cy/fy)

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
			"aabb_scale": 16,
			"frames": [],
		}
    for i, pose in enumerate(poses):
        frame = {
            "file_path": join('images/{:04d}.jpg'.format(i)),
            "transform_matrix": pose.tolist()
        }
        out['frames'].append(frame)
    with open(join(dst_dir, 'transforms.json'), 'w') as f:
        json.dump(out, f, indent=4)
    
    if imgs == None: # support late rendering
        imgs = render_poses(poses)
    for i, img in enumerate(imgs):
        cv2.imwrite(join(imgdir, '{:04d}.jpg'.format(i)), img)


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

    construct_scene_list()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    render = True
    scene_idx = 3
    room_idx = 1
    num_poses_global = 100
    num_poses_per_object = 0
    generate_corners = False
    dst_dir = '/data2/jhuangce/BlenderProc/FRONT3D_render/{:03d}_{}_ngp'.format(scene_idx, room_idx)


    # init and load objects to blenderproc
    bproc.init(compute_device='cuda:0', compute_device_type='CUDA')
    loaded_objects = load_scene_objects(scene_idx)

    # get poses
    room_objects = get_room_objects(scene_idx, room_idx, loaded_objects)
    room_bbox = get_room_bbox(scene_idx, room_idx, loaded_objects)
    poses, num_global, num_closeup = generate_room_poses(scene_idx, room_idx, room_objects, room_bbox, 
                                num_poses_per_object = num_poses_per_object,
                                num_poses_global = num_poses_global
                                )
    if generate_corners:
        poses.extend(generate_four_corner_poses(scene_idx, room_idx)) # last four poses for validation
    print('Summary: {}[global] + {}[closeup] x {}[object] + {}[corner] = {} poses'.format(num_poses_global, num_poses_per_object, len(room_objects), 4 if generate_corners else 0,len(poses)))
    print('Estimated time: {} minutes'.format(len(poses)*25//60))
    input('Press Enter to continue...')

    # render img
    # imgs = render_poses(poses)

    # save to ngp format
    save_in_ngp_format(None, poses, room_bbox, K, dst_dir)

    # # save images
    # for i in range(len(imgs)):
    #     cv2.imwrite(os.path.join(RENDER_TEMP_DIR, '{}.png'.format(i)), imgs[i])


    # # get bboxes, labels, colors
    # aabb_codes, labels, colors = [], [], []
    # for object in room_objects:
    #     bbox = object.get_bound_box()
    #     aabb_codes.append(np.concatenate([np.min(bbox, axis=0), np.max(bbox, axis=0)], axis=0))
    #     labels.append(object.get_name())
    #     color = np.random.choice(range(256), size=3)
    #     colors.append((int(color[0]), int(color[1]), int(color[2])))
    
    # # project bboxes to images
    # imgs_projected = []
    # for img, pose in zip(imgs, poses):
    #     imgs_projected.append(project_bbox_to_image(img, K, np.linalg.inv(pose), aabb_codes, labels, colors))
    
    # # save projected images
    # for i, img in enumerate(imgs_projected):
    #     cv2.imwrite(os.path.join(RENDER_TEMP_DIR, 'proj_{}.png'.format(i)), img)
    
    
    # if render:
    #     render_room(scene_idx=scene_idx, room_idx=room_idx, device='cuda:0')
    # else:
    #     floor_plan = FloorPlan(scene_idx)
    #     floor_plan.drawsamples_and_save()

    print("Success.")