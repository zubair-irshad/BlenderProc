import json
import multiprocessing
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import glob


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        path_pattern = "/wild6d_data/zubair/FRONT3D_render/3dfront_" + str(item) + "_*"

        if len(glob.glob(path_pattern)) > 0:
            queue.task_done()
            print("========", item, "rendered", "========")
            continue
        else:
            print("Path does not exist for pattern:", path_pattern, "generating data")
        # view_path = os.path.join(
        #     "/wild6d_data/zubair/FRONT3D_render", item.split("/")[-1][:-4]
        # )
        # if os.path.exists(view_path):
        #     queue.task_done()
        #     print("========", item, "rendered", "========")
        #     continue
        # else:
        #     os.makedirs(view_path, exist_ok=True)

        # Perform some operation on the item
        print(item, gpu)
        command = [
            "python",
            "cli.py",
            "run",
            "./scripts/render_scene.py",
            "-s",
            str(item),
            "--mode",
            "render",
            "--gpu",
            str(gpu),
        ]

        #     # f"export DISPLAY=:0.{gpu} &&"
        #     # f" GOMP_CPU_AFFINITY='0-47' OMP_NUM_THREADS=48 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE "
        #     f" CUDA_VISIBLE_DEVICES={gpu} "
        #     f" blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py --"
        #     f" --object_path {item}"
        # )
        print(command)
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    start_scene_idx = 2200
    end_scene_idx = 2300
    worker_per_gpu = 1
    num_gpus = 8  # 6
    gpu_start = 0  # 2
    workers = num_gpus * worker_per_gpu
    # Start worker processes on each of the GPUs
    for gpu_i in range(num_gpus):
        for worker_i in range(worker_per_gpu):
            worker_i = gpu_i * worker_per_gpu + worker_i
            process = multiprocessing.Process(target=worker, args=(queue, count, gpu_i))
            process.daemon = True
            process.start()

    # Add items to the queue

    scene_ids = np.arange(start=2200, stop=2300, step=1)

    # with open(args.input_models_path, "r") as f:
    #     model_paths = json.load(f)

    # model_keys = list(model_paths.keys())

    for item in scene_ids:
        queue.put(scene_ids[item])

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(num_gpus * worker_per_gpu):
        queue.put(None)
