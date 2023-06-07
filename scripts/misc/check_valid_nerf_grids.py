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

for scene_name in scenes:
    feature = np.load(os.path.join(feature_dir, scene_name + ".npz"), allow_pickle=True)

    res = feature["resolution"]
    rgbsigma = feature["rgbsigma"]

    print("res", res)
    print("rgbsigma original", rgbsigma.shape)

