import argparse

import subprocess


def main(args):
    for scene_idx in args.frames:
        print("scene_idx", scene_idx)
        # cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} python cli.py run ./scripts/render_scene.py -s {scene_idx} --gpu {args.gpu}"
        # subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames", type=list, help="Is tested on validation data or not."
    )
    # parser.add_argument(
    #     "--end", default=0, type=int, help="Is tested on validation data or not."
    # )
    parser.add_argument(
        "--gpu", default=0, type=int, help="Is tested on validation data or not."
    )
    args = parser.parse_args()
    main(args)
