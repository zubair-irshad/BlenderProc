from pkgutil import extend_path
from telnetlib import NAOP
import imageio
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from mathutils import Vector, Matrix


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

def plot_3d_point_cloud(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.savefig('./test.png')

def build_and_save_scene_cache(cache_dir, scene_objects):
    """
        scene_objects_dict.npz (Objects in this file are not filtered yet!)
        {
            'scene_bbox': np.array(2, 3), [[xmin, ymin, zmin], [xmax, ymax, zmax]]
            'objects': [
                {
                    'name': str,
                    'aabb': np.array(2, 3), [[xmin, ymin, zmin], [xmax, ymax, zmax]]
                    'coords': np.array(8, 3), [[x1, y1, z1], [x2, y2, z2], ...]
                    'obb': [Optional] np.array(7), [x, y, z, w, l, h, theta]
                },
                ...
            ]
        }
    
    """
    scene_objects_dict = {}

    mins, maxs = [], []
    
    objects = []
    for obj in scene_objects:
        obj_dict = {}
        obj_dict['name'] = obj.get_name()
        obj_bound_box = np.array(obj.get_bound_box())
        obj_dict['coords'] = obj_bound_box
        obj_dict['aabb'] = np.array([np.min(obj_bound_box, axis=0), np.max(obj_bound_box, axis=0)])
        obj_dict['volume'] = obj.get_bound_box_volume()
        obj_dict['l2w'] = obj.get_local2world_mat()
        obj_dict['coords_local'] = obj.get_bound_box(local_coords=True)
        obj_dict['scale'] = obj.get_scale()
        
        objects.append(obj_dict)
        mins.append(obj_dict['aabb'][0])
        maxs.append(obj_dict['aabb'][1])

    scene_objects_dict['bbox'] = np.array([np.min(mins, axis=0), np.max(maxs, axis=0)])
    scene_objects_dict['objects'] = objects
    np.savez(os.path.join(cache_dir, 'scene_objects_dict.npz'), **scene_objects_dict)

    return scene_objects_dict

def random_color():
    levels = range(0,255)
    return tuple(random.choice(levels) for _ in range(3))



pi = np.pi

def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def regular_obb(obboxes):
    x, y, w, h, theta = obboxes
    w_regular = np.where(w > h, w, h)
    h_regular = np.where(w > h, h, w)
    theta_regular = np.where(w > h, theta, theta+pi/2)
    theta_regular = regular_theta(theta_regular)
    return np.stack([x, y, w_regular, h_regular, theta_regular])


def rectpoly2obb_2d(polys):
    # polys: (N, 8), [[x1, y1, x2, y2, x3, y3, x4, y4]]
    eps=1e-7
    theta = np.arctan2(-(polys[..., 3] - polys[..., 1]), polys[..., 2] - polys[..., 0] + eps)
    Cos, Sin = np.cos(theta), np.sin(theta)
    Matrix = np.stack([Cos, -Sin, Sin, Cos], axis=-1)
    Matrix = Matrix.reshape(*Matrix.shape[:-1], 2, 2)

    x = polys[..., 0::2].mean(-1)
    y = polys[..., 1::2].mean(-1)
    center = np.expand_dims(np.stack([x, y], axis=-1),-2)
    center_polys = polys.reshape(*polys.shape[:-1], 4, 2) - center
    rotate_polys = np.matmul(center_polys, Matrix.transpose(0, 2, 1))

    xmin = np.min(rotate_polys[..., :, 0], axis=-1)
    xmax = np.max(rotate_polys[..., :, 0], axis=-1)
    ymin = np.min(rotate_polys[..., :, 1], axis=-1)
    ymax = np.max(rotate_polys[..., :, 1], axis=-1)
    w = xmax - xmin
    h = ymax - ymin

    obboxes = np.stack([x, y, w, h, theta])
    return regular_obb(obboxes)

def poly2obb_3d(polys: np.array):
    # polys: (8, 3), [[x1, y1, z1], [x2, y2, z2], ...]"""
    # return: (7), [x, y, z, w, l, h, theta]

    # Rectify the polys to obb
    mins, maxs = np.min(polys, axis=0), np.max(polys, axis=0)
    z0, z1 = mins[2], maxs[2]
    z = (z0+z1) * 0.5
    h = z1-z0
    polys_2d = []
    for point in polys:
        if point[2] > z:
            point[2] = z1
            polys_2d.append(point)
        elif point[2] < z:
            point[2] = z0
    polys_2d = np.array(polys_2d)[:, :2]
    polys_2d = polys_2d[[0, 2, 3, 1]].reshape(-1, 8)

    # Get the obb
    x, y, w, l, theta = rectpoly2obb_2d(polys_2d)
    return np.array([x, y, z, w, l, h, theta], dtype=np.float32)

def obb2ngp(obb_3d):
    # obb_3d: (7), [x, y, z, w, l, h, theta]
    position = obb_3d[:3]
    extents = obb_3d[3:6]
    theta = -obb_3d[-1]
    Cos, Sin = float(np.cos(theta)), float(np.sin(theta))
    orientation = np.array([[Cos, -Sin, 0], [Sin, Cos, 0], [0, 0, 1]])

    return extents.tolist(), orientation.tolist(), position.tolist()