import os

directory = "/wild6d_data/zubair/FRONT3D_render"

# Find all subfolders within the directory
for root, dirs, files in os.walk(directory):
    for dir_name in dirs:
        if dir_name == "images":
            images_dir = os.path.join(root, dir_name)
            files_count = len(os.listdir(images_dir))
            print(f"Files count in {images_dir}: {files_count}")
            if files_count < 25:
                # Delete the main subfolder
                main_folder_path = os.path.dirname(images_dir)
                main_folder = os.path.basename(main_folder_path)
                print(f"Deleting {main_folder_path}")
                # os.rmdir(main_folder_path)