# Phase 3: GRN-Driven Platelet Behavior under Shear Flow

## Objective

The objective of Phase 3 is to establish a fully functional **sensor–GRN–actuator feedback loop** governing platelet-like agents under flow conditions.

This phase represents the transition from:
- purely physical particle transport  
to  
- **behavior-driven, biologically inspired agents**

---

## System Architecture

Each agent follows a closed-loop pipeline:

**Sensors → GRN → Actuators → Motion**

### Sensors (Inputs)
- **Shear stress** (flow-dependent mechanical stimulus)
- **Collision impulse** (mechanical interaction)
- **Chemical concentration** (environmental signaling)

### GRN (Gene Regulatory Network)
- Integrates multi-modal inputs
- Produces continuous activation states ∈ [0, 1]
- Encodes simplified intracellular decision logic

### Actuators (Outputs)
- **Stickiness** → affects adhesion and velocity damping  
- **Morphology** → represents activation state (size scaling)  
- **Secretion** → represents signaling output (color encoding)

---

## Simulation Setup

- Geometry: Cylindrical vessel (3D)
- Flow: Simplified shear-driven transport (Poiseuille-inspired)
- Agents: ~20 independent platelets
- Time: Continuous simulation rendered as video

---

## Visual Encoding

| Property       | Representation              |
|---------------|----------------------------|
| Position      | 3D coordinates (x, y, z)   |
| Secretion     | Color (viridis colormap)   |
| Morphology    | Marker size                |
| Shear         | Spatial position (radial)  |
| Dynamics      | Particle trails            |

---

## Results: 3D Shear Behavior Analysis

A time-resolved 3D simulation was generated:

**Output:**  
`results/month3/month3_behavior_shear_analysis_3d.mp4`

This visualization demonstrates:

- Spatial distribution of agents under flow  
- Temporal evolution of GRN outputs  
- Emergent clustering patterns  

---

## Observations

### 1. Spatial Organization
- Agents naturally cluster in regions of similar shear exposure  
- No explicit clustering rule was implemented → **emergent behavior**

### 2. GRN Output Distribution
- Secretion and morphology vary across spatial regions  
- Clear heterogeneity between agents

### 3. Temporal Dynamics
- GRN states evolve smoothly over time  
- No oscillatory instability or divergence observed

---

## Shear Response Characteristics

The system exhibits **nonlinear response behavior**:

- Strong activation increase at low shear
- Saturation at higher shear levels
- Stickiness decreases with increasing shear

This confirms:

- Shear acts as a **trigger mechanism**
- GRN introduces **regulation and saturation**

---

## Interpretation

The simulation demonstrates successful integration of:

> mechanical stimuli → intracellular processing → behavioral response

Key insight:

- Behavior is **not hardcoded**
- It is **emergent from GRN dynamics**

This aligns with the core thesis goal:

> coupling intracellular regulatory networks with agent-based simulation :contentReference[oaicite:0]{index=0}  

---

## Validation Against Expected Behavior

Observed trends match known platelet characteristics:

- Activation increases under flow  
- Adhesion decreases at high shear  
- Signaling saturates due to regulation  

Thus, the system produces **biologically plausible qualitative behavior**.

---

## Limitations

- GRN model is simplified (not calibrated to experimental data)
- No explicit platelet–platelet adhesion yet
- Flow is approximated (no full fluid solver)
- Chemical field is static

---

## Key Contributions of Phase 3

- Implementation of a **closed-loop behavior system**
- Demonstration of **GRN-controlled agents**
- Stable **multi-agent simulation**
- Emergent behavior without explicit rules
- First **3D visualization of GRN-driven platelet dynamics**

---

## Next Steps (Phase 4)

- Introduce realistic shear computation
- Implement adhesion via stickiness output
- Compare **low vs high shear regimes**
- Validate spatial aggregation behavior

---

## Conclusion

Phase 3 successfully establishes a **behavior-driven simulation framework**, where platelet dynamics are governed by internal regulatory networks rather than predefined rules.

This provides the foundation for:
- biologically meaningful simulations  
- scalable multi-agent systems  
- and future validation studies