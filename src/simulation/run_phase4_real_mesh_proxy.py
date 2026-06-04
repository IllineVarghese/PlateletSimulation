from pathlib import Path
import re

import numpy as np
import pandas as pd


def read_unity_mesh_bounds(mesh_path: Path):
    text = mesh_path.read_text(errors="ignore")

    center_match = re.search(
        r"m_Center:\s*\{x:\s*([\-\d.eE]+),\s*y:\s*([\-\d.eE]+),\s*z:\s*([\-\d.eE]+)\}",
        text,
    )
    extent_match = re.search(
        r"m_Extent:\s*\{x:\s*([\-\d.eE]+),\s*y:\s*([\-\d.eE]+),\s*z:\s*([\-\d.eE]+)\}",
        text,
    )
    vertex_match = re.search(r"vertexCount:\s*(\d+)", text)
    index_match = re.search(r"indexCount:\s*(\d+)", text)

    if not center_match or not extent_match:
        raise ValueError("Could not read Unity mesh bounds.")

    center = np.array([float(v) for v in center_match.groups()])
    extent = np.array([float(v) for v in extent_match.groups()])

    vertex_count = int(vertex_match.group(1)) if vertex_match else -1
    index_count = int(index_match.group(1)) if index_match else -1

    return center, extent, vertex_count, index_count


def main():
    mesh_path = Path("data/meshes/vessel/NurbsPath.mesh")
    out_dir = Path("results/phase4/real_mesh_proxy")
    out_dir.mkdir(parents=True, exist_ok=True)

    center, extent, vertex_count, index_count = read_unity_mesh_bounds(mesh_path)

    rng = np.random.default_rng(42)

    n_agents = 500
    n_frames = 60
    dt = 0.02

    length = float(extent[0])
    radius = float(min(extent[1], extent[2]) * 0.5)

    x_min = center[0] - extent[0] / 2
    x_max = center[0] + extent[0] / 2

    positions = np.zeros((n_frames, n_agents, 3))
    velocities = np.zeros((n_frames, n_agents, 3))
    shear_input = np.zeros((n_frames, n_agents))
    activation = np.zeros((n_frames, n_agents))
    stickiness = np.zeros((n_frames, n_agents))
    morphology = np.zeros((n_frames, n_agents))

    # spawn agents inside mesh-derived cylindrical proxy
    r = radius * np.sqrt(rng.random(n_agents))
    theta = rng.random(n_agents) * 2 * np.pi

    positions[0, :, 0] = rng.uniform(x_min, x_max, n_agents)
    positions[0, :, 1] = center[1] + r * np.cos(theta)
    positions[0, :, 2] = center[2] + r * np.sin(theta)

    vmax = 2.0

    for frame in range(n_frames):
        if frame > 0:
            positions[frame] = positions[frame - 1]

        dy = positions[frame, :, 1] - center[1]
        dz = positions[frame, :, 2] - center[2]
        radial_distance = np.sqrt(dy**2 + dz**2)

        radial_norm = np.clip(radial_distance / radius, 0.0, 1.0)

        axial_velocity = vmax * (1.0 - radial_norm**2)
        velocities[frame, :, 0] = axial_velocity

        shear_input[frame] = radial_norm

        if frame == 0:
            activation[frame] = shear_input[frame]
        else:
            activation[frame] = (
                0.85 * activation[frame - 1]
                + 0.15 * shear_input[frame]
            )

        stickiness[frame] = np.clip(0.8 * activation[frame], 0.0, 1.0)
        morphology[frame] = np.clip(1.2 * activation[frame], 0.0, 1.0)

        if frame < n_frames - 1:
            positions[frame + 1] = positions[frame]
            positions[frame + 1, :, 0] += axial_velocity * dt

            # wrap agents back to inlet
            over = positions[frame + 1, :, 0] > x_max
            positions[frame + 1, over, 0] = x_min

    np.save(out_dir / "real_mesh_proxy_positions.npy", positions)
    np.save(out_dir / "real_mesh_proxy_velocities.npy", velocities)
    np.save(out_dir / "real_mesh_proxy_shear_input.npy", shear_input)
    np.save(out_dir / "real_mesh_proxy_activation.npy", activation)
    np.save(out_dir / "real_mesh_proxy_stickiness.npy", stickiness)
    np.save(out_dir / "real_mesh_proxy_morphology.npy", morphology)

    summary = pd.DataFrame(
        {
            "mesh_file": [str(mesh_path)],
            "vertex_count": [vertex_count],
            "index_count": [index_count],
            "center_x": [center[0]],
            "center_y": [center[1]],
            "center_z": [center[2]],
            "extent_x": [extent[0]],
            "extent_y": [extent[1]],
            "extent_z": [extent[2]],
            "proxy_length": [length],
            "proxy_radius": [radius],
            "agents": [n_agents],
            "frames": [n_frames],
            "mean_final_activation": [float(activation[-1].mean())],
            "max_final_activation": [float(activation[-1].max())],
        }
    )

    summary.to_csv(out_dir / "real_mesh_proxy_summary.csv", index=False)

    print("Real vessel mesh proxy simulation complete.")
    print("Mesh:", mesh_path)
    print("Vertices:", vertex_count)
    print("Indices:", index_count)
    print("Center:", center)
    print("Extent:", extent)
    print("Proxy length:", length)
    print("Proxy radius:", radius)
    print("Agents:", n_agents)
    print("Frames:", n_frames)
    print("Mean final activation:", float(activation[-1].mean()))
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()