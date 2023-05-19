import yaml
import os

subentry_count = 0

directory_path = "/home/mirshad7/BlenderProc/scripts/all_bboxes"

directory_path_valid = "/home/mirshad7/BlenderProc/scripts/all_valid_boxes"

start = 2300
end = 2400
# Loop through each YAML file from 2100 to 2200
for i in range(start, end):
    filename = os.path.join(directory_path, f"bbox_{i}.yaml")
    try:
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
            # Count the number of subentries in the YAML file
            subentry_count += len(data[i].keys())
    except FileNotFoundError:
        continue

print("Total subentry count:", subentry_count)


subentry_count = 0
# Loop through each YAML file from 2100 to 2200
for i in range(start, end):
    filename = os.path.join(directory_path_valid, f"bbox_{i}.yaml")
    try:
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
            # Count the number of subentries in the YAML file
            subentry_count += len(data[i].keys())
    except FileNotFoundError:
        continue

print("Total subentry count valid:", subentry_count)
