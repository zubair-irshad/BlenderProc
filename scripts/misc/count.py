import os
import numpy as np

folder_path = "/wild6d_data/zubair/FRONT3D_render"

count = 0
countg600 = 0
countl80 = 0
all_file_count = 0

for folder in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder)
    if os.path.isdir(folder_full_path):
        image_path = os.path.join(folder_full_path, "train/images")
        if os.path.isdir(image_path):
            file_count = sum(
                [
                    1
                    for file in os.listdir(image_path)
                    if (file.endswith(".jpg") or (file.endswith(".png")))
                ]
            )
            if file_count > 120 and file_count < 450:
                count += 1
                all_file_count += file_count
            if file_count > 450:
                # print("folder, file_count", folder, file_count)
                countg600 += 1

            elif file_count < 120:
                countl80 += 1
                print("folder, file_count", folder, file_count)


print("===============================================\n\n")
print("images greater than 60 and less than 450", count)
print("all file count", all_file_count)
print("images greater than 450", countg600)
print("images less than 120", countl80)
print("===============================================\n\n")
