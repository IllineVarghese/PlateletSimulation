# Pre-Practical GPU Baseline (Cylinder Poiseuille)

## What this is
Frozen baseline case for the pre-practical phase:
GPU-accelerated particle transport in a cylindrical vessel under a prescribed
Poiseuille velocity profile with wall collisions.

## Device
CUDA GPU (NVIDIA Warp).

## Inputs / Parameters
- Geometry: cylinder (constant radius)
- Flow: Poiseuille axial profile
- Collisions: particle–wall response
- Respawn: inlet respawn enabled (continuous flow)

## Outputs
- Trajectory data:
  results/positions_steps_warp_cyl_poiseuille.npy
- Visualization:
  results/baseline_cylinder_poiseuille_gpu.png

## Status
Baseline frozen. Future work will add quantitative analysis, animations,
and physical validation.
_Last updated before supervisor meeting._
  
## Video Outputs

- Cylinder Poiseuille flow (GPU, static coloring):
  results/cylinder_poiseuille_gpu.mp4

- Cylinder Poiseuille flow (GPU, velocity-colored):
  results/cylinder_poiseuille_gpu_colored.mp4

These videos demonstrate:
- GPU-based particle transport
- Parabolic Poiseuille velocity profile
- Correct boundary confinement
- Frame-consistent scalar visualization
