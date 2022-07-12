import blenderproc_arcive_arcive as bproc
import argparse
import os
import numpy as np
import h5py
import cv2

import sys
sys.path.append('/data2/jhuangce/BlenderProc/scripts')
from utils import LAYOUT_DIR, TEXTURE_DIR, MODEL_DIR
from utils import SCENE_LIST, construct_scene_list
construct_scene_list()


parser = argparse.ArgumentParser()
parser.add_argument("--future_folder", help="Path to the 3D Future Model folder.", default=MODEL_DIR)
parser.add_argument("--front_3D_texture_path", help="Path to the 3D FRONT texture folder.", default=TEXTURE_DIR)
parser.add_argument("--output_dir", help="Path to where the data should be saved")
args = parser.parse_args()

bproc.init(compute_device='cuda:3', compute_device_type='CUDA')
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=SCENE_LIST[0],
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False

num = 1
for i in range(num):
    # Sample point inside house
    height = 1.5
    location = np.array([0, 2.5, height])
    # Sample rotation (fix around X and Y axis)
    rotation = [1.35, 0, 2*np.pi*i/num]
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    bproc.camera.add_camera_pose(cam2world_matrix)

bproc.renderer.set_denoiser('INTEL')

image_width = 640
image_height = 480
K = np.array([
    [544, 0, 320],
    [0, 544, 240],
    [0, 0, 1]
])
bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)


# Also render normals
# bproc.renderer.enable_normals_output()

# render the whole pipeline
output_dir = './output/test6'
data = bproc.renderer.render(output_dir=output_dir)
# data.update(bproc.renderer.render_segmap(map_by="class")) # render segmentation map

print(data['colors'][0].shape)
for i in range(len(data['colors'])):
    cv2.imwrite('./output/img%03d.jpg' % i, data['colors'][i])

# write the data to a .hdf5 container
# bproc.writer.write_hdf5(args.output_dir, data)

# file_list = os.listdir(args.output_dir)
# file_list = [x for x in file_list if x[-5:]=='.hdf5']
# os.mkdir(os.path.join(args.output_dir, 'imgs'))

# for hdf5_name in file_list:
#     with h5py.File(os.path.join(args.output_dir, hdf5_name)) as f:
#         colors = np.array(f["colors"])
#     cv2.imwrite(os.path.join(args.output_dir, 'imgs', hdf5_name+'.jpg'), colors)