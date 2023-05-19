import yaml
import os

subentry_count = 0

directory_path = "/home/ubuntu/zubair/BlenderProc/scripts/all_valid_bboxes"

directory_path_valid = "/home/ubuntu/zubair/BlenderProc/scripts/all_valid_boxes"

# Loop through each YAML file from 2100 to 2200
for i in range(2100, 2200):
    filename = os.path.join(directory_path, f"bbox_{i}.yaml")
    try:
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
            # Count the number of subentries in the YAML file
            subentry_count += len(data.keys())
    except FileNotFoundError:
        continue

print("Total subentry count:", subentry_count)

# Loop through each YAML file from 2100 to 2200
for i in range(2100, 2200):
    filename = os.path.join(directory_path_valid, f"bbox_{i}.yaml")
    try:
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
            # Count the number of subentries in the YAML file
            subentry_count += len(data.keys())
    except FileNotFoundError:
        continue

print("Total subentry count valid:", subentry_count)
