import os
import json
import numpy as np

folder_path = "/arkit_data/zubair/FRONT3D_render_3k"

for folder in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder)
    if os.path.isdir(folder_full_path):
        print("folder", folder_full_path)
        image_path = os.path.join(folder_full_path, "train/images")
        if os.path.isdir(image_path):
            # Check if first image has file format 0000.jpg or rgb_0000.png
            file_name = np.sort(os.listdir(image_path))[0]
            # print(file_name)

            if file_name.startswith("rgb"):
                # Update the file path in the transforms.json file
                json_path = os.path.join(folder_full_path, "train/transforms.json")
                with open(json_path, "r") as f:
                    transforms = json.load(f)
                    for frame in transforms["frames"]:
                        print("prev path", frame["file_path"])
                        file_name = frame["file_path"].split("/")[-1]
                        file_index = file_name.split(".")[0]
                        new_file_path = os.path.join(
                            "images", "rgb_" + file_index + ".png"
                        )
                        frame["file_path"] = new_file_path
                        print("new path", frame["file_path"])
                        print("==========================\n")
                with open(json_path, "w") as f:
                    json.dump(transforms, f, indent=4)
