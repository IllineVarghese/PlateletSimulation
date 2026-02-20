import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from .sim_state import create_state
from .platelet_step import step_state


def run_simulation(config: dict):
    """
    Week 2 thesis runner:
    - reads parameters from config dict
    - creates persistent SimState ONCE
    - runs for N steps on chosen device
    - updates positions/velocities IN-PLACE each timestep
    - saves outputs into a fixed output folder
    - logs timing + FPS
    """

    sim_cfg = config.get("simulation", {})
    geom_cfg = config.get("geometry", {})  # kept for Week 3 (walls etc.)
    flow_cfg = config.get("flow", {})      # kept for Week 3 (Poiseuille etc.)
    out_cfg = config.get("output", {})

    # ---- simulation params ----
    seed = int(sim_cfg.get("seed", 42))
    steps = int(sim_cfg.get("steps", 100))
    num_agents = int(sim_cfg.get("num_agents", 1000))
    dt = float(sim_cfg.get("dt", 0.001))

    device = sim_cfg.get("device", "cpu")
    if device == "gpu":
        device = "cuda"  # allow "gpu" alias
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Invalid device '{device}'. Use 'cpu' or 'cuda' (or 'gpu').")

    base_dir = str(out_cfg.get("base_dir", "results/run"))
    save_every = int(out_cfg.get("save_every", 10))  # don't save every step for FPS tests

    # ---- reproducibility ----
    # NOTE: SimState uses its own NumPy RNG seeded from `seed` when initializing.
    # Keeping np.random.seed here is fine for other parts of the pipeline.
    np.random.seed(seed)

    # ---- output folder ----
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config copy (thesis reproducibility)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    print("\n=== Run (Persistent SimState) ===")
    print(f"device     : {device}")
    print(f"seed       : {seed}")
    print(f"steps      : {steps}")
    print(f"num_agents : {num_agents}")
    print(f"dt         : {dt}")
    print(f"output_dir : {out_dir}")
    print(f"save_every : {save_every}\n")

    # ---- timing logs ----
    timing_path = out_dir / "timing.csv"
    timing_file = open(timing_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(timing_file)
    writer.writerow(["step", "step_state_sec", "to_numpy_sec", "total_step_sec", "saved"])

    # ---- create persistent state ONCE (Week 2 core) ----
    # IMPORTANT: this allocates positions/velocities only once and keeps them across steps.
    state = create_state(config, device=device)

    # sanity print (safe)
    try:
        print(f"[init] state.positions.device = {state.positions.device}")
        print(f"[init] state.positions.shape  = {state.positions.shape}")
    except Exception:
        pass

    # Store only saved frames (keeps memory reasonable)
    saved_positions = []
    saved_steps = []
    saved_activation = []

    t_run0 = time.perf_counter()

    for i in range(steps):
        t_step0 = time.perf_counter()

        # 1) compute step (GPU/CPU) IN-PLACE
        t0 = time.perf_counter()
        step_state(state, dt=dt, cfg=config, debug_print=False)
        t_step_state = time.perf_counter() - t0

        # 2) only copy to numpy when saving (IMPORTANT for FPS)
        saved = 0
        t_to_numpy = 0.0

        # Keep your exact saving rule from Week 1:
        # save on first step, every save_every, and last step
        if save_every > 0 and ((i + 1) % save_every == 0 or i == 0 or i == steps - 1):
            t0 = time.perf_counter()

            # copy positions
            pos_np = state.positions.numpy()

            # copy activation
            act_np = state.activation.numpy()

            t_to_numpy = time.perf_counter() - t0

            saved_positions.append(pos_np)
            saved_activation.append(act_np)
            saved_steps.append(i)
            
            saved = 1  # <-- ADD THIS

            # lightweight checkpoint save (same as Week 1)
            np.save(out_dir / "positions_saved_steps.npy", np.array(saved_steps, dtype=np.int32))
            np.save(out_dir / "positions_saved.npy", np.stack(saved_positions, axis=0))
            np.save(out_dir / "activation_saved.npy", np.stack(saved_activation, axis=0))


        t_total_step = time.perf_counter() - t_step0
        writer.writerow([i, t_step_state, t_to_numpy, t_total_step, saved])

        # Print just once (safe)
        if i == 0:
            try:
                print("positions shape:", state.positions.shape)
            except Exception:
                pass

    total_runtime = time.perf_counter() - t_run0
    fps = steps / total_runtime if total_runtime > 0 else 0.0

    timing_file.close()

    # ---- final summary ----
    fps_path = out_dir / "fps.csv"
    with open(fps_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["steps", "runtime_sec", "fps", "device", "seed", "num_agents", "dt"])
        w.writerow([steps, total_runtime, fps, device, seed, num_agents, dt])

    print("\n--- Run complete ---")
    print(f"timing.csv : {timing_path}")
    print(f"fps.csv    : {fps_path}")
    print(f"runtime(s) : {total_runtime:.4f}")
    print(f"FPS        : {fps:.2f}")
    print(f"saved npy  : {out_dir / 'positions_saved.npy'} (only saved frames)\n")


def main():
    """
    CLI:
      python src/simulation/platelet_sim.py --config config.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_simulation(config)


if __name__ == "__main__":
    main()
