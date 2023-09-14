import os
import numpy as np
import warnings

min_dim = 50
feature_dir = "/arkit_data/hm3d_rpn_data"
npz_files = os.listdir(feature_dir)
npz_files = [
    f
    for f in npz_files
    if f.endswith(".npz") and os.path.isfile(os.path.join(feature_dir, f))
]

scenes = [f.split(".")[0] for f in npz_files]


def density_to_alpha(density, scene_name):
    with warnings.catch_warnings():  # Catch any warnings that occur within this block
        warnings.filterwarnings(
            "error", category=RuntimeWarning
        )  # Raise warnings as errors
        try:
            alpha = np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)
        except RuntimeWarning as e:
            print(f"Warning occurred in scene {scene_name}: {e}")

    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


filtered_scenes = []
filtered_scenes_count = 0
count = 0
for scene_name in scenes:
    count += 1
    if count % 100 == 0:
        print("count", count)
    # print("scene_name", scene_name)
    # print(
    #     "os.path.join(feature_dir, scene_name + .npz)",
    #     os.path.join(feature_dir, scene_name + ".npz"),
    # )
    feature = np.load(os.path.join(feature_dir, scene_name + ".npz"), allow_pickle=True)

    res = feature["resolution"]
    rgbsigma = feature["rgbsigma"]

    density = rgbsigma[..., -1]

    # print("density min max", density.min(), density.max())

    alpha = density_to_alpha(density, scene_name)

    # print("alpha min max", alpha.min(), alpha.max())

    # print("res", res)
    # print("rgbsigma original", rgbsigma.shape)

    # if sum(dim < 50 for dim in res) == 2 or any(dim < 20 for dim in res):
    #     print("scene_name", scene_name)
    #     print("res", res)
    #     filtered_scenes_count += 1
    #     filtered_scenes.append(scene_name)

    # print("==========================================\n\n\n")
print("Invalid number of grids", filtered_scenes_count)

# # Save the filtered scene names to a text file
# with open("filtered_scenes_hm3d.txt", "w") as file:
#     file.write("\n".join(filtered_scenes))
