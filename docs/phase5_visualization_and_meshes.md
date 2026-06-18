# Phase 5: Visualization & Meshes

## Goal

The goal of Phase 5 was to improve the visual realism and export capability of the platelet simulation by replacing simple marker-based platelet rendering with real platelet mesh geometry. This phase focused on importing inactive and activated platelet meshes, switching the visible platelet morphology based on activation level, improving visual interpretation of activated platelets, optimizing rendering performance, and exporting the final scene into USD-compatible formats for professional visualization workflows.

Phase 5 corresponds to the Month 5 pipeline step:

- Import inactive and activated platelet meshes
- Implement mesh switching based on activation level
- Add optional activated-state deformation
- Set up Omniverse / USD export
- Perform performance optimization and scaling tests

---

## Input Mesh Assets

The following platelet mesh assets were available in the project:

- `data/meshes/platelet/inactive.obj`
- `data/meshes/platelet/activated.obj`
- `data/meshes/platelet/plateletDemo.fbx`

The inactive and activated OBJ meshes were used as the main platelet morphology assets. The FBX file was inspected separately, but direct FBX loading was not supported reliably by the current PyVista/VTK environment. Therefore, the OBJ-derived inactive and activated meshes were selected for the Phase 5 implementation.

---

## Week 1: Mesh Import and Inspection

The first step was to import and inspect the inactive and activated platelet meshes. The goal was to confirm that the meshes could be loaded, rendered, normalized, and compared visually before using them in the simulation.

The mesh inspection confirmed that both platelet states could be loaded and rendered successfully. The inactive mesh showed a more rounded resting morphology, while the activated mesh showed a more altered morphology suitable for representing activation-dependent visual state changes.

Generated Week 1 outputs included:

- `platelet_mesh_comparison.png`
- `platelet_mesh_view_xy.png`
- `platelet_mesh_view_xz.png`
- `platelet_mesh_view_yz.png`
- `platelet_mesh_view_iso.png`
- `platelet_mesh_state_scene.png`

These outputs confirmed that the platelet meshes were suitable for the next stage of activation-based rendering.

---

## Week 2: Activation-Based Mesh Switching

In Week 2, the platelet mesh state was linked to the activation value produced by the Phase 4 simulation. A threshold-based switching rule was implemented:

- Platelets with activation below 0.50 use the inactive platelet mesh.
- Platelets with activation greater than or equal to 0.50 use the activated platelet mesh.

This allowed the simulation to display different platelet morphologies depending on the activation state.

The mesh switching was first tested in a simplified vessel scene and then applied to real Phase 4 simulation data. The Phase 4 data included platelet positions, activation values, and shear values. The resulting visualizations showed inactive and activated platelets inside the vessel proxy, with activation values controlling the mesh state.

Important Week 2 outputs included:

- `mesh_switching_vessel_snapshot.png`
- `phase4_mesh_switching_snapshot.png`
- `phase4_mesh_switching_video.mp4`
- `phase4_mesh_switching_video_thesis.mp4`
- `phase5_thesis_activation_snapshot.png`

This completed the core requirement of activation-based mesh switching.

---

## Week 3: Activated Morphology, Mesh Optimization, and Performance

### Activation-Dependent Visual Deformation

Week 3 added a visual morphology model for activated platelets. The purpose was to make activated platelets easier to distinguish in dense visualizations.

The implemented deformation was visual-only. It included:

- activation-dependent scaling
- increased roughness or morphological exaggeration
- pseudo-filopodia / protrusion overlays for highly activated platelets
- blue-to-red activation color mapping

This should not be described as a biomechanical soft-body simulation. It is an activation-dependent visual deformation model used to improve readability.

The main output was:

- `advanced_activation_deformation_progression.png`

This figure shows how platelet morphology and color change with increasing activation.

### Mesh Decimation

The original platelet meshes were visually detailed but computationally expensive for dense scenes. Therefore, mesh decimation was implemented to reduce polygon complexity while preserving the recognizable platelet shape.

Measured mesh reduction results:

| Mesh | Original cells | Decimated cells | Cell reduction |
|---|---:|---:|---:|
| Inactive platelet | 225,296 | 17,671 | 92.2% |
| Activated platelet | 39,238 | 4,483 | 88.6% |

The decimated meshes were saved locally as:

- `results/phase5/week3/optimized_meshes/inactive_decimated.vtp`
- `results/phase5/week3/optimized_meshes/activated_decimated.vtp`

The main visualization output was:

- `mesh_decimation_comparison.png`

This figure demonstrated that the optimized meshes preserved scene-level platelet appearance while greatly reducing rendering cost.

### Rendering Performance Benchmark

The performance benchmark compared original and decimated platelet meshes using the same Phase 4 simulation frame and increasing platelet counts.

Measured render times:

| Rendered platelets | Original mesh time (s) | Decimated mesh time (s) | Approximate speedup |
|---:|---:|---:|---:|
| 25 | 1.595 | 0.469 | 3.4x |
| 50 | 1.757 | 0.738 | 2.4x |
| 100 | 3.413 | 1.161 | 2.9x |
| 150 | 5.403 | 1.829 | 3.0x |
| 200 | 6.935 | 2.229 | 3.1x |

The most important result was observed at 200 rendered platelets, where render time decreased from approximately 6.94 seconds to 2.23 seconds. This corresponds to an approximate 3.1x speedup.

The main performance output was:

- `original_vs_decimated_mesh_performance.png`

This result justified using decimated meshes for dense scenes, videos, scaling tests, and USD export.

### Optimized Thesis-Style Video

An optimized video was created using:

- real Phase 4 positions
- real activation values
- decimated inactive and activated platelet meshes
- activation-based mesh switching
- blue-to-red activation coloring
- visual protrusion overlays for highly activated platelets

The main video output was:

- `phase5_decimated_deformed_thesis_video.mp4`

This became the strongest Phase 5 dynamic visualization output.

---

## Week 4: USD / Omniverse-Compatible Export

### USD Environment Setup

The initial environment check showed that the Pixar USD Python modules were missing. The package `usd-core` was then installed successfully, which enabled the following modules:

- `pxr`
- `pxr.Usd`
- `pxr.UsdGeom`
- `pxr.Sdf`
- `pxr.Gf`

The Omniverse-specific module `omni.usd` was not available inside the normal VS Code Python environment. This is expected because `omni.usd` is usually part of the Omniverse Kit Python environment. However, standalone USD file generation was possible using `usd-core`.

Therefore, the project continued with direct USD export using Pixar USD Python tools.

### Static USD Export

A static USD scene was exported using one selected Phase 4 frame. The scene contained:

- vessel proxy mesh
- exported platelet meshes
- inactive and activated platelet states
- activation-based color values
- shear and activation metadata
- source platelet indices
- scale and mesh-state information

Generated static USD outputs:

- `phase5_platelet_static_scene.usda`
- `platelet_scene_metadata.csv`
- `usd_static_scene_export_summary.md`
- `phase5_static_usd_scene_preview.png`

The static export contained 80 platelet meshes:

| State | Count |
|---|---:|
| Inactive | 15 |
| Activated | 65 |
| Total | 80 |

The static USD file was approximately 19.6 MB because each platelet mesh was written directly into the USD scene.

### Animated USD Export

An animation-ready USD scene was then created using `UsdGeom.PointInstancer`. This is a more efficient USD design because it stores platelet mesh prototypes once and reuses them across many platelet instances.

The animated scene used:

- one inactive platelet prototype
- one activated platelet prototype
- time-sampled platelet positions
- time-sampled platelet scales
- time-sampled prototype indices for inactive/activated switching
- real Phase 4 activation and shear values

Generated animated USD outputs:

- `phase5_platelet_animated_scene.usda`
- `platelet_animation_metadata.csv`
- `animated_usd_export_summary.md`
- `phase5_animated_usd_final_preview.png`
- `activation_state_counts_over_time.png`

Animated USD export summary:

| Item | Value |
|---|---:|
| Source frames | 24 |
| Exported platelet IDs | 120 |
| Metadata rows | 2,880 |
| Start time code | 0 |
| End time code | 23 |
| Frames per second | 6 |
| Position time samples | 24 |
| Scale time samples | 24 |
| Prototype-index time samples | 24 |
| Prototype count | 2 |

The animated USD file was approximately 1.0 MB, which is much smaller than the static mesh-expanded USD scene. This demonstrates the advantage of using `PointInstancer` for animation-ready USD export.

---

## USD Export Validation

A validation script was implemented to check the static and animated USD exports. The validation confirmed that the files were generated correctly and could be opened by Pixar USD Python tools.

Validation results:

| Result type | Count |
|---|---:|
| PASS | 28 |
| WARN | 0 |
| FAIL | 0 |

The validation confirmed:

- static USD scene exists
- static USD scene opens successfully
- static metadata rows match the exported platelet mesh count
- animated USD scene exists
- animated USD scene opens successfully
- `PointInstancer` exists
- two prototypes exist: inactive and activated platelet
- 120 persistent platelet IDs exist
- positions are time-sampled across 24 frames
- scales are time-sampled across 24 frames
- prototype indices are time-sampled across 24 frames
- metadata row count matches `120 × 24 = 2880`

This validates the USD export package for Phase 5 documentation.

---

## Final Phase 5 Outputs

The most important final outputs from Phase 5 are:

| Output | Purpose |
|---|---|
| `platelet_mesh_comparison.png` | Shows imported inactive and activated platelet meshes |
| `platelet_mesh_state_scene.png` | Demonstrates initial mesh state rendering |
| `phase4_mesh_switching_snapshot.png` | Shows real Phase 4 data rendered with mesh switching |
| `phase5_thesis_activation_snapshot.png` | Thesis-style activation visualization |
| `advanced_activation_deformation_progression.png` | Explains activation-dependent visual deformation |
| `mesh_decimation_comparison.png` | Shows original vs decimated mesh comparison |
| `original_vs_decimated_mesh_performance.png` | Quantifies render-time improvement |
| `phase5_decimated_deformed_thesis_video.mp4` | Main optimized Phase 5 video |
| `phase5_platelet_static_scene.usda` | Static USD export |
| `phase5_platelet_animated_scene.usda` | Animation-ready USD export |
| `phase5_usd_export_validation.md` | USD export validation report |

---

## Scripts Added in Phase 5

The following scripts were added or used during Phase 5:

- `src/visualization/inspect_platelet_meshes.py`
- `src/visualization/inspect_platelet_mesh_orientation.py`
- `src/visualization/inspect_platelet_fbx.py`
- `src/visualization/mesh_utils.py`
- `src/visualization/test_mesh_utils.py`
- `src/visualization/platelet_mesh_state_scene.py`
- `src/visualization/phase5_mesh_switching_vessel_snapshot.py`
- `src/visualization/phase5_render_phase4_mesh_snapshot.py`
- `src/visualization/phase5_render_phase4_mesh_video.py`
- `src/visualization/phase5_thesis_visual_snapshot.py`
- `src/visualization/phase5_activated_deformation_test.py`
- `src/visualization/phase5_advanced_activation_deformation.py`
- `src/visualization/phase5_mesh_decimation_test.py`
- `src/visualization/phase5_compare_decimated_performance.py`
- `src/visualization/phase5_decimated_deformed_thesis_video.py`
- `src/visualization/phase5_week3_output_review.py`
- `src/visualization/phase5_usd_environment_check.py`
- `src/visualization/phase5_export_static_usd_scene.py`
- `src/visualization/phase5_export_animated_usd_scene.py`
- `src/visualization/phase5_validate_usd_exports.py`

---

## Limitations

Several limitations should be stated clearly.

First, the activated deformation is not a physical soft-body simulation. It is a visual morphology model used to improve readability of activation states. Therefore, the deformation and pseudo-filopodia overlays should be described as visual-only.

Second, the USD files were generated and validated using Pixar USD Python tools. Manual inspection inside NVIDIA Omniverse was not performed in this environment. Therefore, the exports should be described as USD-ready or Omniverse-compatible, not as Omniverse-rendered unless they are later opened and inspected in Omniverse.

Third, the visualization uses selected platelet subsets for readability and performance. This is appropriate for thesis figures and videos, but the subset selection should be stated when describing exported scenes.

Fourth, the vessel geometry used in Phase 5 export is a vessel proxy derived from the Phase 4 simulation coordinate range. It is suitable for visualization and export testing, but it is not a full biomechanical vessel-wall model.

---

## Conclusion

Phase 5 successfully upgraded the platelet simulation from simple particle-style visualization to mesh-based professional visualization. Inactive and activated platelet meshes were imported, inspected, normalized, and linked to the activation state of each platelet. Real Phase 4 simulation data were rendered using activation-based mesh switching, activation coloring, and visual deformation overlays.

Performance optimization was achieved through mesh decimation. The inactive platelet mesh cell count was reduced by 92.2%, and the activated platelet mesh cell count was reduced by 88.6%. Rendering benchmarks showed that decimated meshes reduced the 200-platelet render time from approximately 6.94 seconds to 2.23 seconds, corresponding to an approximate 3.1x speedup.

USD export was implemented successfully after enabling `usd-core`. Both static and animation-ready USD scenes were generated. The animated export used `UsdGeom.PointInstancer` with two platelet prototypes and time-sampled positions, scales, and prototype indices across 24 frames. Validation produced 28 PASS checks, 0 WARN checks, and 0 FAIL checks.

Overall, Phase 5 fulfilled the pipeline requirements for real platelet mesh visualization, activation-based mesh switching, visual activated-state deformation, performance optimization, and USD-ready professional visualization export.