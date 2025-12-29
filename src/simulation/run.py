import argparse
import warp as wp


def main(device: str):
    wp.init()

    if device == "cuda" and wp.is_cuda_available():
        wp.set_device("cuda")
    else:
        wp.set_device("cpu")

    print("Running on device:", wp.get_device())
    print("Available devices:", wp.get_devices())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    main(args.device)
