import os
import yaml

start_idx = 2300
end_idx = 2500
folder_path = "/home/ubuntu/BlenderProc/scripts/all_is_valid"

valid_files = []

for idx in range(start_idx, end_idx + 1):
    file_path = os.path.join(folder_path, f"is_valid_{idx}.yaml")
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            print("data", data)
            if data.get("value") == True:
                valid_files.append(idx)

print(valid_files)
print(len(valid_files))