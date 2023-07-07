# import subprocess
# import multiprocessing
import os
import subprocess
import math
from subprocess import Popen, PIPE


# def process_scene(gpu_id, scene_idx):
#     # gpu_id = scene_idx % 6 + 2  # round-robin scheduling among GPUs 2-7
#     cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python cli.py run ./scripts/render_scene.py -s {scene_idx} --gpu {gpu_id}"
#     subprocess.run(cmd, shell=True)


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

    # log_dir = "./scripts/logs"
    # os.makedirs(log_dir, exist_ok=True)
    # path = "./scripts/all_bboxes"

    # start_scene_idx = 2000
    # end_scene_idx = 2100

    # scene_lists = []
    # for i in range(start_scene_idx, end_scene_idx):
    #     yaml_path = os.path.join(path, "bbox_" + str(i) + ".yaml")
    #     if os.path.exists(yaml_path):
    #         with open(yaml_path, "r") as f:
    #             r_config = yaml.load(f, Loader=yaml.FullLoader)
    #         for scene_idx in r_config:
    #             for room_idx in r_config[scene_idx]:
    #                 scene_lists.append([scene_idx, room_idx])

    # # create a list of GPU ids to use
    # gpu_ids = list(range(2, 8))

    # with multiprocessing.Pool(6) as pool:
    #     for scene_idx in range(2000, 3000):
    #         pool.starmap(process_scene, [(gpu_id, scene_idx) for gpu_id in range(2, 8)])

    # num_workers = 6  # number of workers in the process pool

    # with Pool(num_workers) as pool:
    #     pool.map(process_scene, range(start_scene_idx, end_scene_idx))

    # # spawn a subprocess for each GPU-scene index pair
    # procs = []
    # for scene_idx in range(start_scene_idx, end_scene_idx):
    #     gpu_id = gpu_ids[scene_idx % len(gpu_ids)]  # round-robin scheduling
    #     # scene_room_subset = scene_lists[scene_idx]
    #     # scene_idx = scene_room_subset[0]
    #     # room_idx = scene_room_subset[1]
    #     cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python cli.py run ./scripts/render_scene.py -s {scene_idx} --gpu {gpu_id}"
    #     proc = subprocess.Popen(cmd, shell=True)
    #     procs.append(proc)

    # # wait for all subprocesses to complete
    # for proc in procs:
    #     proc.wait()

    start_scene_idx = 3000
    end_scene_idx = 3300
    path = "./scripts/all_bboxes"
    all_frames = []
    for i in range(start_scene_idx, end_scene_idx):
        yaml_path = os.path.join(path, "bbox_" + str(i) + ".yaml")
        if os.path.exists(yaml_path):
            all_frames.append(i)

    print("all_frames", all_frames)
    worker_per_gpu = 1
    num_gpus = 6  # 6
    gpu_start = 2  # 2
    workers = num_gpus * worker_per_gpu
    # all_frames = range(start_scene_idx, end_scene_idx)
    # print("workers", workers)
    frames_per_worker = math.ceil(len(all_frames) / workers)
    print("frames_per_worker", frames_per_worker)

    # processes = []
    log_dir = "./scripts/logs"
    for i in range(workers):
        curr_gpu = (i // worker_per_gpu) + gpu_start

        start = i * frames_per_worker
        end = start + frames_per_worker

        print(i, curr_gpu)
        print(all_frames[start:end])
        print("start, : end", start, end)

        my_env = os.environ.copy()
        frames_str = ",".join(str(f) for f in all_frames[start:end])
        my_env["CUDA_VISIBLE_DEVICES"] = str(curr_gpu)
        command = [
            "python",
            "distributed_worker.py",
            "--gpu",
            str(curr_gpu),
            "--frames",
            frames_str,
            # "--start",
            # str(start),
            # "--end",
            # str(end),
        ]
        log_file = os.path.join(log_dir, f"log_{i}.txt")
        log = open(log_file, "w")
        print(command)
        Popen(command, env=my_env, stderr=log, stdout=log)

    #     start = i * frames_per_worker
    #     end = start + frames_per_worker

    #     print(i, curr_gpu)
    #     print(all_frames[start:end])
    #     print("start, : end", start, end)

    #     # Select a subset of scene_idx and room_idx for this worker
    #     scene_room_subset = scene_lists[start:end]

    #     # Construct the command to run
    #     command = ["python", "cli.py", "run", "./scripts/render_scene.py"]
    #     for scene_room in scene_room_subset:
    #         command += [
    #             "-s",
    #             str(scene_room[0]),
    #             "-r",
    #             str(scene_room[1]),
    #             "--mode",
    #             "render",
    #             "--gpu",
    #             str(curr_gpu),
    #         ]

    #     # Spawn a process to run the command
    #     log_file = os.path.join(log_dir, f"log_{i}.txt")
    #     log = open(log_file, "w")
    #     env = os.environ.copy()
    #     env["CUDA_VISIBLE_DEVICES"] = str(curr_gpu)
    #     p = subprocess.Popen(command, env=env, stderr=log, stdout=log)
    #     processes.append(p)
    # # Wait for all the processes to finish
    # # for p in processes:
    # #     p.wait()

    # for p in processes:
    #     return_code = p.wait()
    #     if return_code != 0:
    #         print(
    #             f"process exited with non-zero status code: {return_code}, moving to next ..."
    #         )
    #         continue


if __name__ == "__main__":
    main()
