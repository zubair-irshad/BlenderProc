# python cli.py run ./scripts/utils.py 

import blenderproc as bproc
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

LAYOUT_DIR = '/data2/jhuangce/3D-FRONT'
TEXTURE_DIR = '/data2/jhuangce/3D-FRONT-texture'
MODEL_DIR = '/data2/jhuangce/3D-FUTURE-model'
SCENE_LIST = []

def construct_scene_list():
    layout_list = [join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)]
    layout_list.sort()
    for scene_code in layout_list:
        SCENE_LIST.append(scene_code)

def check_cache_dir(scene_idx):
    if not os.path.isdir(f'./cached/{scene_idx}'):
        os.mkdir(f'./cached/{scene_idx}')

def load_scene_objects(scene_idx, overwrite=False):
    check_cache_dir(scene_idx)
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    loaded_objects = bproc.loader.load_front3d(
        json_path=SCENE_LIST[scene_idx],
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping
    )

    return loaded_objects

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
    
    def draw_cameras(self):
        for pos in CAMERA_LOCS[self.scene_idx]:
            cv2.circle(self.image, self.point_to_image(pos), 50, color=(0,255,0), thickness=3)
            cv2.putText(self.image, 'camera', self.point_to_image(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))

    def draw_coords(self):
        red = (0,0,255)
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
    
    def save(self):
        cv2.imwrite(f'./cached/{self.scene_idx}/floor_plan.jpg', self.image)
    
    def draw_and_save(self):
        self.draw_objects()
        self.draw_coords()
        self.draw_cameras() # customizable
        self.save()

def image_to_video(img_dir, video_dir):
    img_list = os.listdir(img_dir)
    img_list.sort()
    rgb_maps = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in img_list]
    print(len(rgb_maps))

    imageio.mimwrite(os.path.join(video_dir, 'video.mp4'), np.stack(rgb_maps), fps=30, quality=8)


def get_scene_information(scene_idx, overwrite=False):
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

def draw_map(scene_idx):
    floor_plan = FloorPlan(scene_idx)
    floor_plan.draw_and_save()

def render_sample(scene_idx, device):
    """ Each camera position render 8 images. """
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
            rotation = [1.4, 0, 2*np.pi*i/4]
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
            bproc.camera.add_camera_pose(cam2world_matrix)
            
    
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)

    data = bproc.renderer.render()
    # data.update(bproc.renderer.render_segmap(map_by="class")) # render segmentation map

    for i in range(len(data['colors'])):
        cv2.imwrite('./output/%04d/img_%02d.jpg' % (scene_idx, i), data['colors'][i])


if __name__ == '__main__':

    construct_scene_list()
    # bproc.init(compute_device='cuda:3', compute_device_type='CUDA')
    # mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    # mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # for i in range(5):
    #     draw_map(scene_idx=i)

    render_sample(scene_idx=1, device='cuda:0')

    # image_to_video('./output/test6/images', './output/test6')

    print("Success.")