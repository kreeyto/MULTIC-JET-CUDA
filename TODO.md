## ✅ TODO / Roadmap

A set of key improvements and extensions planned for this project:

### 🔧 1. Memory Optimization and Performance

- [ ] **Implement shared memory usage** in key CUDA kernels:
  - Especially in `gpuComputeInterface` and the core LBM routines in `lbm.cu`.
  - Objective: reduce global memory bandwidth usage and improve data locality.

- [ ] **Reduce register pressure** in `gpuFusedCollisionStream` and `gpuEvolveScalarField`:
  - Currently limiting GPU occupancy.
  - Consider:
    - Kernel splitting (if registers cannot be reused efficiently)
    - Local reuse of intermediate results
    - Using `__launch_bounds__` to balance occupancy

### 🧩 2. Codebase Generalization and Modularity

- [ ] **Merge** this repository (`MULTIC-JET-CUDA`) with the `MULTIC-BUBBLE-CUDA` project.
  - Goal: make the codebase general for **multicomponent LBM flows**, regardless of geometry or injection scenario.

- [ ] **Refactor simulation logic** to support multiple case types via:
  - `#define` macros
  - Compilation flags (`-D<CASE>`): e.g., `-DJET_CASE`, `-DBUBBLE_CASE`, etc.
  - Encapsulation of case-specific setup (inflow/boundary conditions, initial fields, etc.)

### 🌊 3. Boundary Conditions

- [ ] **Implement boundary conditions** for a complete physical domain:
  - [ ] **Periodic** boundaries on lateral walls (`x` and `y` directions).
  - [ ] **Outflow** boundary at the domain exit (`z = NZ - 1`).
  - [x] **Inflow** already implemented at `z = 0`.

### 🔬 4. Physics Extensions

- [ ] **Introduce a thermal model** into the LBM core:
  - Purpose: enable simulation of **thermal effects** in multicomponent flows.
  - Strategy:
    - Add a new scalar distribution for temperature.
    - Couple viscosity/surface tension to temperature and component.

- [ ] **Associate physical properties** to each fluid component:
  - Assign **oil** properties to the injected jet and **water** to the background medium.

- [ ] **Allow dynamic oil properties**:
  - Parametrize oil characteristics (density, viscosity, surface tension) for multiple types or API grades.
  - Possibly via external config or compile-time macros.

### 📦 5. Code Usability

- [ ] Add configuration flexibility for:
  - Mesh size
  - Output frequency (`MACRO_SAVE`)
  - Number of steps (`NSTEPS`)
  - Jet velocity (`U_JET`)
  - Via external JSON or YAML file, or compile-time defines

- [ ] Improve post-processing abstraction and ease-of-use
  - Automate variable detection
  - Support additional fields (e.g., temperature if thermal model is added)

---

Feel free to contribute, discuss or pick up tasks from this list!
