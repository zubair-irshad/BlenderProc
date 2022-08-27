
import yaml
import subprocess
import sys

with open('./scripts/room_configs.yaml', 'r') as f:
    ALL_ROOM_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

skip = {1003: [0,1,2,3], 1005: [0,1], 1008: [0], 1009:[0]}


all_scene_keys = ALL_ROOM_CONFIG.keys()
all_skip_keys = skip.keys()
for i in range(1012, 1016):
    if i not in all_scene_keys:
        print(f'Scene {i} does not have config.')
        continue
    scene_info = ALL_ROOM_CONFIG[i]


    scene_room_configs = scene_info.keys()

    for r in scene_room_configs:
        if i in all_skip_keys:
            if r in skip[i]:
                print(f'Skip room {r} of scene {i}.')
                continue
        bashCommand = f"python cli.py run ./scripts/render.py --gpu 1 -s {i} -r {r} --render -nc"
        process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=sys.stdout)
        output, error = process.communicate()
