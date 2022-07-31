# python cli.py run ./scripts/utils.py 

import blenderproc as bproc
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


def add_texture(obj:MeshObject, tex_path):
    obj.clear_materials()
    mat = obj.new_material('my_material')
    bsdf = mat.nodes["Principled BSDF"]
    texImage = mat.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(tex_path)
    mat.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

    # mat = obj.blender_obj.data.materials[0]
    # bsdf = mat.node_tree.nodes["Principled BSDF"]
    # texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    # texImage.image = bpy.data.images.load(tex_path)
    # mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    


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

    for obj in loaded_objects:
        name = obj.get_name()
        if 'wall' in name.lower():
            add_texture(obj, "/data2/jhuangce/3D-FRONT-texture/1b57700d-f41b-4ac7-a31a-870544c3d608/texture.png")
        elif 'floor' in name.lower():
            add_texture(obj, "/data2/jhuangce/3D-FRONT-texture/0b48b46d-4f0b-418d-bde6-30ca302288e6/texture.png")
        # elif 'ceil' in name.lower():
        #     add_texture(obj, "/data2/jhuangce/3D-FRONT-texture/0a5adcc7-f17f-488f-9f95-8690cbc31321/texture.png")

    return loaded_objects

def get_locs_and_rots(scene_idx, room_idx = None):
    """ Generate the camera locations and rotations in a room. """
    locations, rotations = [], []
    config_dict = ROOM_CONFIG[scene_idx]
    for key, value in config_dict.items():
        if room_idx!=None and key!=room_idx:
            continue
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
        locs, rots = get_locs_and_rots(self.scene_idx)
        self.draw_objects()
        self.draw_coords()
        self.draw_samples(locs, rots) # customizable
        self.draw_room_bbox()
        self.save('floor_plan2.jpg')

def image_to_video(img_dir, video_dir):
    img_list = os.listdir(img_dir)
    img_list.sort()
    rgb_maps = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in img_list]
    print(len(rgb_maps))

    imageio.mimwrite(os.path.join(video_dir, 'video.mp4'), np.stack(rgb_maps), fps=30, quality=8)

def draw_map(scene_idx):
    floor_plan = FloorPlan(scene_idx)
    floor_plan.drawsamples_and_save()

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
    locs, rots = get_locs_and_rots(scene_idx, room_idx)
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




if __name__ == '__main__':

    construct_scene_list()

    scene_idx = 5
    room_idx = 2
    render = False

    if render:
        render_room(scene_idx=scene_idx, room_idx=room_idx, device='cuda:0')
    else:
        floor_plan = FloorPlan(scene_idx)
        floor_plan.drawsamples_and_save()

    print("Success.")