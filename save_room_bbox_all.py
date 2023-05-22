import subprocess

# for scene_idx in range(3000, 4000):
#     cmd = f"python cli.py run ./scripts/save_room_bboxes.py --scene_idx {scene_idx}"
#     subprocess.run(cmd, shell=True)


for scene_idx in range(3000, 4000):
    cmd = f"python cli.py run ./scripts/check_valid_poses.py --scene_idx {scene_idx}"
    subprocess.run(cmd, shell=True)
