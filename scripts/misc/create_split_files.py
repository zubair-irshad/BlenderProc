import os
import numpy as np

# Load the split file
split_file = "/wild6d_data/zubair/nerf_rpn/front3d_rpn_data/front3d_split.npz"

split = np.load(split_file)
# Get the list of scenes from the features directory
features_dir = "/wild6d_data/zubair/MAE_complete_data/features"

out_file = "/wild6d_data/zubair/MAE_complete_data/front3d_split.npz"
scenes = []
for file_name in os.listdir(features_dir):
    if file_name.endswith(".npz"):
        scene_name = os.path.splitext(file_name)[0]
        scenes.append(scene_name)

modified_split = dict(split)
# Replace train_scenes in the split file with the list of scenes
modified_split["train_scenes"] = np.array(scenes)
# Save the modified split file
np.savez(out_file, **modified_split)
