import os

# Source folder
source_folder = "/arkit_data/zubair/front3d_rpn_data_3k"

# Destination folder
destination_folder = "/arkit_data/zubair/MAE_invalid_grids"

# Read the filtered scenes from the file
with open("/home/ubuntu/zubair/BlenderProc/filtered_scenes.txt", "r") as file:
    scenes = file.read().splitlines()

# Move the scenes to the destination folder
for scene_name in scenes:
    source_path = os.path.join(source_folder, f"{scene_name}.npz")
    destination_path = os.path.join(destination_folder, f"{scene_name}.npz")
    os.rename(source_path, destination_path)
