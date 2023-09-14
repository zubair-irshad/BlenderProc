import os
import numpy as np


min_dim = 50
feature_dir = "/arkit_data/hm3d_rpn_data"
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
    # print("scene_name", scene_name)
    # print(
    #     "os.path.join(feature_dir, scene_name + .npz)",
    #     os.path.join(feature_dir, scene_name + ".npz"),
    # )
    feature = np.load(os.path.join(feature_dir, scene_name + ".npz"), allow_pickle=True)

    res = feature["resolution"]
    # rgbsigma = feature["rgbsigma"]

    print("res", res)
    # print("rgbsigma original", rgbsigma.shape)

    if sum(dim < 50 for dim in res) == 2:
        #     print(arr)
        # if res[0] <min_dim or res[1] <min_dim or res[2] <min_dim:
        print("scene_name", scene_name)
        print("res", res)
        filtered_scenes_count += 1
        filtered_scenes.append(scene_name)

print("Invalid number of grids", filtered_scenes_count)

# Save the filtered scene names to a text file
# with open("filtered_scenes.txt", "w") as file:
#     file.write("\n".join(filtered_scenes))
