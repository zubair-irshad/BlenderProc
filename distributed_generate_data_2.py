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

        # path_pattern = "/wild6d_data/zubair/FRONT3D_render/3dfront_" + str(item) + "_*"

        path_pattern = (
            "/arkit_data/zubair/FRONT3D_render_3k/3dfront_" + str(item) + "_*"
        )

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
        command = f"CUDA_VISIBLE_DEVICES={gpu} python cli.py run ./scripts/render_scene.py -s {item} --gpu {gpu}"

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

    start_scene_idx = 3000
    end_scene_idx = 3300
    worker_per_gpu = 5
    # num_gpus = 8  # 6
    # gpu_start = 0  # 2
    # num_gpus = 6
    # gpu_start = 0

    # gpus_available = [0, 2, 3, 4, 5, 6, 7]
    # gpus_available = [0, 1, 2, 3, 4, 5, 6, 7]
    # gpus_available = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    gpus_available = [2, 3, 4, 5, 6, 7]
    num_gpus = len(gpus_available)
    gpu_start = gpus_available[0]

    workers = num_gpus * worker_per_gpu
    # Start worker processes on each of the GPUs
    for gpu_i in range(num_gpus):
        gpu_id = gpus_available[gpu_i]
        for worker_i in range(worker_per_gpu):
            worker_i = (gpu_i - gpu_start) * worker_per_gpu + worker_i
            # worker_id = (gpu_i - gpu_start) * worker_per_gpu + worker_i
            # worker_i = gpu_i * worker_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_id)
            )
            process.daemon = True
            process.start()

    # Add items to the queue

    scene_ids = np.arange(start=start_scene_idx, stop=end_scene_idx, step=1)

    # with open(args.input_models_path, "r") as f:
    #     model_paths = json.load(f)

    # model_keys = list(model_paths.keys())

    for item in scene_ids:
        queue.put(item)

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(num_gpus * worker_per_gpu):
        queue.put(None)
