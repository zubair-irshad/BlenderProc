import os
import numpy as np

# Load the split file
dataset_name = "front3d"
out_name = "hm3d"
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

out_dir = "/arkit_data/hm3d_rpn_data_ft"
features_dir = os.path.join(out_dir, "features")
# features_dir = "/wild6d_data/zubair/nerf_rpn/scannet_rpn_data_all/features"

out_file = os.path.join(out_dir, out_name + "_split.npz")
# out_file = "/wild6d_data/zubair/MAE_complete_data/front3d_split.npz"
scenes = []
for file_name in os.listdir(features_dir):
    if file_name.endswith(".npz"):
        scene_name = os.path.splitext(file_name)[0]
        scenes.append(scene_name)


# Create an array of indices representing all scenes
all_indices = np.arange(len(scenes))

# Shuffle the indices randomly
np.random.shuffle(all_indices)

# Select the first 20 indices for validation
val_indices = all_indices[:20]

# Select the next 18 indices for testing
test_indices = all_indices[20:38]

# The remaining indices are for training
train_indices = all_indices[38:]

print("len train val test", len(train_indices), len(val_indices), len(test_indices))

# Use the selected indices to create the split
selected_val_scenes = [scenes[i] for i in val_indices]
selected_test_scenes = [scenes[i] for i in test_indices]
selected_train_scenes = [scenes[i] for i in train_indices]

# Now you can update your modified_split dictionary
modified_split = dict(split)
modified_split["train_scenes"] = np.array(selected_train_scenes)
modified_split["val_scenes"] = selected_val_scenes
modified_split["test_scenes"] = selected_test_scenes


# OLD CODE FOR MAE PRETRAINING, UNCOMMENT FOR MAE PRETRAINING
# # Make a random selection of 20 scenes
# selected_scenes = np.random.choice(scenes, size=20, replace=False)
# modified_split = dict(split)
# # Replace train_scenes in the split file with the list of scenes
# modified_split["train_scenes"] = np.array(scenes)
# modified_split["val_scenes"] = selected_scenes[:10]  # First 10 scenes for validation
# modified_split["test_scenes"] = selected_scenes[10:]  # Last 10 scenes for testing


# Save the modified split file
np.savez(out_file, **modified_split)
