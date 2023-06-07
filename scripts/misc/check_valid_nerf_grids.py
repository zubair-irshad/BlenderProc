import os
import numpy as np

feature_dir = '/wild6d_data/zubair/FRONT3D_MAE'
npz_files = os.listdir(feature_dir)
npz_files = [
    f
    for f in npz_files
    if f.endswith(".npz") and os.path.isfile(os.path.join(feature_dir, f))
]

scenes = [f.split(".")[0] for f in npz_files]

filtered_scenes = []
filtered_scenes_count = 0
for scene_name in scenes:
    feature = np.load(os.path.join(feature_dir, scene_name + ".npz"), allow_pickle=True)

    res = feature["resolution"]
    # rgbsigma = feature["rgbsigma"]

    # print("res", res)
    # print("rgbsigma original", rgbsigma.shape)

    if res[0] <40 or res[1] <40 or res[2] <40:
        print("scene_name", scene_name)
        print("res", res)
        filtered_scenes_count += 1
        filtered_scenes.append(scene_name)

print("Invalid number of grids", filtered_scenes_count)

# Save the filtered scene names to a text file
with open("filtered_scenes.txt", "w") as file:
    file.write("\n".join(filtered_scenes))
