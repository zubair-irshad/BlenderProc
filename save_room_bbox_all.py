import subprocess

for scene_idx in range(2000, 3000):
    cmd = f"python cli.py run ./scripts/save_room_bboxes.py --scene_idx {scene_idx}"
    subprocess.run(cmd, shell=True)
