import os
import math
import torch
import yaml
import subprocess


# # Define a function to run the command
# def get_command(scene_idx, room_idx, gpu):
#     command = [
#         "python",
#         "cli.py",
#         "run",
#         "./scripts/render_scene.py",
#         "-s",
#         str(scene_idx),
#         "-r",
#         str(room_idx),
#         "--mode",
#         "render",
#         "--gpu",
#         str(gpu),
#     ]
#     return command
#     # subprocess.Popen(command).wait()


def main():
    # Create a pool of processes, one for each worker

    log_dir = "./scripts/logs"
    os.makedirs(log_dir, exist_ok=True)
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

    worker_per_gpu = 1
    workers = torch.cuda.device_count() * worker_per_gpu
    print("workers", workers)
    all_frames = range(0, len(scene_lists))
    frames_per_worker = math.ceil(len(all_frames) / workers)
    gpu_start = 2
    processes = []
    for i in range(workers):
        curr_gpu = (i // worker_per_gpu) + gpu_start

        start = i * frames_per_worker
        end = start + frames_per_worker

        print(i, curr_gpu)
        print(all_frames[start:end])
        print("start, : end", start, end)

        # Select a subset of scene_idx and room_idx for this worker
        scene_room_subset = scene_lists[start:end]

        # Construct the command to run
        command = ["python", "cli.py", "run", "./scripts/render_scene.py"]
        for scene_room in scene_room_subset:
            command += [
                "-s",
                str(scene_room[0]),
                "-r",
                str(scene_room[1]),
                "--mode",
                "render",
                "--gpu",
                str(curr_gpu),
            ]

        # Spawn a process to run the command
        log_file = os.path.join(log_dir, f"log_{i}.txt")
        log = open(log_file, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(curr_gpu)
        p = subprocess.Popen(command, env=env, stderr=log, stdout=log)
        processes.append(p)
    # Wait for all the processes to finish
    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
