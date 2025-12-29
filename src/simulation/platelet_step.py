import warp as wp
import numpy as np


@wp.kernel
def move_platelets(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt


def main():
    wp.init()
    wp.set_device("cpu")

    num_platelets = 10
    dt = 0.01

    positions = wp.array(
        np.random.rand(num_platelets, 3),
        dtype=wp.vec3,
        device="cpu",
    )

    velocities = wp.array(
        np.random.randn(num_platelets, 3) * 0.1,
        dtype=wp.vec3,
        device="cpu",
    )

    wp.launch(
        kernel=move_platelets,
        dim=num_platelets,
        inputs=[positions, velocities, dt],
    )

    print("Updated positions:")
    print(positions.numpy())


if __name__ == "__main__":
    main()
