## Setup (Local - Windows)

- Python environment: venv (.venv)
- Install: pip install warp-lang
- Tested CPU: warp runs in CPU-only mode

## How to run


python src/simulation/platelet_step.py
python src/simulation/platelet_sim.py --steps 3 --device cpu
python src/simulation/platelet_sim.py --steps 3 --device cuda
python src/warp_playground/warp_hello_cpu.py
python src/warp_playground/warp_particles_cpu.py
python src/visualization/view_positions_pyvista.py

Outputs:
- results/positions_steps.npy (from platelet_sim.py)
- results/positions_steps_warp.npy (from warp_particles_cpu.py)
- results/pyvista_step_last.png (from view_positions_pyvista.py)
