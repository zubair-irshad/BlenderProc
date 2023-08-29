import os
import numpy as np

# Load the split file
dataset_name = "front3d"

# split_file = (
#     f"/wild6d_data/zubair/nerf_rpn/{dataset_name}_rpn_data/{dataset_name}_split.npz"
# )

split_file = (
    f"/wild6d_data/zubair/nerf_rpn/{dataset_name}_rpn_data/{dataset_name}_split.npz"
)
print("split_file", split_file)

# split_file = "/wild6d_data/zubair/nerf_rpn/front3d_rpn_data/front3d_split.npz"

split = np.load(split_file)
# Get the list of scenes from the features directory

out_dir = "/wild6d_data/zubair/MAE_complete_data"
features_dir = os.path.join(out_dir, "features")
# features_dir = "/wild6d_data/zubair/nerf_rpn/scannet_rpn_data_all/features"

out_file = os.path.join(out_dir, dataset_name + "_split.npz")
# out_file = "/wild6d_data/zubair/MAE_complete_data/front3d_split.npz"
scenes = []
for file_name in os.listdir(features_dir):
    if file_name.endswith(".npz"):
        scene_name = os.path.splitext(file_name)[0]
        scenes.append(scene_name)

# Make a random selection of 20 scenes
selected_scenes = np.random.choice(scenes, size=20, replace=False)
modified_split = dict(split)
# Replace train_scenes in the split file with the list of scenes
modified_split["train_scenes"] = np.array(scenes)
modified_split["val_scenes"] = selected_scenes[:10]  # First 10 scenes for validation
modified_split["test_scenes"] = selected_scenes[10:]  # Last 10 scenes for testing
# Save the modified split file
np.savez(out_file, **modified_split)
