import os
import numpy as np

# Load the split file
split_file = '/wild6d_data/zubair/nerf_rpn/front3d_rpn_data/3dfront_split.npz'

split = np.load(split_file)
# Get the list of scenes from the features directory
features_dir = '/wild6d_data/zubair/FRONT3D_MAE/features'

out_file = '/wild6d_data/zubair/FRONT3D_MAE/3dfront_split.npz'
scenes = []
for file_name in os.listdir(features_dir):
    if file_name.endswith('.npz'):
        scene_name = os.path.splitext(file_name)[0]
        scenes.append(scene_name)
# Replace train_scenes in the split file with the list of scenes
split['train_scenes'] = np.array(scenes)
# Save the modified split file
np.savez(out_file, **split)