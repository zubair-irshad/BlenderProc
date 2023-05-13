import os
import sys
import argparse
import math
from subprocess import Popen, PIPE
import torch
import yaml
import multiprocessing as mp
import subprocess


# Define a function to run the command
def run_command(scene_idx, room_idx, gpu):
    command = [
        "python",
        "cli.py",
        "run",
        "./scripts/render_scene.py",
        "-s",
        str(scene_idx),
        "-r",
        str(room_idx),
        "--mode",
        "render",
        "--gpu",
        str(gpu),
    ]
    subprocess.Popen(command).wait()


def main():
    # Create a pool of processes, one for each worker

    path = "./scripts/all_bboxes"

    start_scene_idx = 2000
    end_scene_idx = 2100

    scene_lists = []
    for i in range(start_scene_idx, end_scene_idx):
        yaml_path = os.path.join(path, "bbox_" + str(i) + ".yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                r_config = yaml.load(f, Loader=yaml.FullLoader)
            for scene_idx in r_config:
                for room_idx in r_config[scene_idx]:
                    scene_lists.append([scene_idx, room_idx])

    worker_per_gpu = 20
    workers = torch.cuda.device_count() * worker_per_gpu
    pool = mp.Pool(processes=workers)
    all_frames = range(0, len(scene_lists))
    frames_per_worker = math.ceil(len(all_frames) / workers)
    gpu_start = 2

    for i in range(workers):
        curr_gpu = (i // worker_per_gpu) + gpu_start

        start = i * frames_per_worker
        end = start + frames_per_worker

        print(i, curr_gpu)
        print(all_frames[start:end])
        print("start, : end", start, end)

        # Select a subset of scene_idx and room_idx for this worker
        scene_room_subset = scene_lists[start:end]

        # Create a list of arguments for the run_command function
        args_list = [
            (scene_idx, room_idx, curr_gpu) for scene_idx, room_idx in scene_room_subset
        ]

        # Run the commands in parallel
        pool.starmap(run_command, args_list)

    # Close the pool of processes
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
