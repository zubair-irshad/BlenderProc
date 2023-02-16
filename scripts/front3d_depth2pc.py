import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt


def write_ply(pcd):
    colors = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8)

    num_points = np.sum([p.shape[0] for p in pcd.values()])

    with open('scene.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {num_points}\n'
                'property float x\n'
                'property float y\n'
                'property float z\n' 
                'property uchar red\n'
                'property uchar green\n' 
                'property uchar blue\n'
                'end_header\n')
        
        for obj_id, p in pcd.items():
            color = colors[obj_id]
            for num in range(p.shape[0]):
                f.write("{:.4f} ".format(p[num][0]))
                f.write("{:.4f} ".format(p[num][1]))
                f.write("{:.4f} ".format(p[num][2]))
                f.write("{:d} ".format(color[0]))
                f.write("{:d} ".format(color[1]))
                f.write("{:d}". format(color[2]))
                f.write("\n")


def depth2pc(depth_dir, mask_dir, scene_dir, points_per_obj=100000):
    scene_name = os.path.basename(scene_dir)

    with open(os.path.join(scene_dir, 'train', 'transforms.json'), 'r') as f:
        json_file = json.load(f)

    pcd = defaultdict(list)

    depth_files = sorted(os.listdir(depth_dir), key=lambda x: int(x.split('.')[0]))
    mask_files = sorted(os.listdir(mask_dir), key=lambda x: int(x.split('.')[0]))

    assert len(depth_files) == len(mask_files), 'depth and mask files are not matched'
    assert len(depth_files) == len(json_file['frames']), 'depth and json files are not matched'
    
    fx, fy, cx, cy = json_file['fl_x'], json_file['fl_y'], json_file['cx'], json_file['cy']
    for i in range(len(depth_files)):
        frame = json_file['frames'][i]
        
        depth_img = h5py.File(os.path.join(depth_dir, depth_files[i]), 'r')
        depth_img = np.array(depth_img['depth'][:])

        mask_img = h5py.File(os.path.join(mask_dir, mask_files[i]), 'r')
        mask_img = np.array(mask_img['cp_instance_id_segmaps'][:])
        
        H, W = mask_img.shape
        assert H == depth_img.shape[0] and W == depth_img.shape[1], 'depth and mask shapes not matched'
                
        x = np.linspace(0, H-1, H, endpoint=True)
        y = np.linspace(0, W-1, W, endpoint=True)
        j,i = np.meshgrid(x, y, indexing='ij') 

        c_x = (i + 0.5 - cx) / fx * depth_img 
        c_y = (H - j - 0.5 - cy) / fy * depth_img
        c_z = -depth_img
        c_coord = np.stack([c_x, c_y, c_z], axis = -1)
        
        c2w = np.array(frame['transform_matrix'])

        c_coord = c_coord.reshape([-1, 3])
        w_coord = c2w[:3,:3] @ c_coord.T + c2w[:3, 3][:, np.newaxis]
        w_coord = w_coord.T
        valid_depth = (depth_img.reshape(-1) > 0) & (depth_img.reshape(-1) < 15)

        ids = np.unique(mask_img)
        for id in ids:
            if id == 0:
                continue
            mask = mask_img == id
            mask = mask.reshape(-1)
            mask = mask & valid_depth
            pcd[id].append(w_coord[mask, :])
    
    for id in pcd.keys():
        pcd[id] = np.concatenate(pcd[id], axis=0)
        if pcd[id].shape[0] > points_per_obj:
            pcd[id] = pcd[id][np.random.choice(pcd[id].shape[0], points_per_obj, replace=False), :]

    return pcd


def write_npz(pcd, path):
    ids = []
    points = []
    for id, p in pcd.items():
        ids.extend([id] * p.shape[0])
        points.append(p)

    points = np.concatenate(points, axis=0)
    ids = np.array(ids)

    np.savez(path, points=points, ids=ids)


if __name__ == '__main__':

    depth_dir = '/data/bhuai/BlenderProc/FRONT3D_render/depth'
    mask_dir = '/data/bhuai/BlenderProc/FRONT3D_render/seg'
    out_dir = '/data/bhuai/BlenderProc/FRONT3D_render/pcd'
    os.makedirs(out_dir, exist_ok=True)

    scenes = os.listdir('/data/bhuai/front3d_ngp')
    for s in tqdm(scenes):
        pcd = depth2pc(os.path.join(depth_dir, s), os.path.join(mask_dir, s), 
                       os.path.join('/data/bhuai/front3d_ngp', s))
        write_npz(pcd, os.path.join(out_dir, s + '.npz'))
    