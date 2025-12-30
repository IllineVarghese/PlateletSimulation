## Setup (Local - Windows)

- Python environment: venv (.venv)
- Install: pip install warp-lang
- Tested CPU: warp runs in CPU-only mode

## How to run

python src/simulation/run.py --device cpu
python src/simulation/platelet_step.py
python src/simulation/platelet_sim.py --steps 3
