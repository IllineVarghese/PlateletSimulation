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
  