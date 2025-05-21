## ✅ TODO / Roadmap

A set of key improvements and extensions planned for this project.

### 🔧 1. Memory Optimization and Performance

- [ ] **Implement shared memory usage** in key CUDA kernels:
  - Especially in interface kernels in `lbm_int.cu` and the core LBM routines in `lbm.cu`.
  - Objective: reduce global memory bandwidth usage and improve data locality.

- [x] **Reduced register pressure** in `gpuMomCollisionStream`:
  - Previously limiting GPU occupancy.
  - Implemented via:
    - Compilation flag `--maxrregcount` to limit register usage;
  - Still considering:
    - `__launch_bounds__` to fine-tune occupancy and scheduling.

### 🧩 2. Codebase Generalization and Modularity

- [ ] **Merge** this repository (`MULTIC-TS-CUDA`) with the `MULTIC-BUBBLE-CUDA` project.
  - Goal: make the codebase general for **multicomponent LBM flows**, regardless of geometry or injection scenario.

- [ ] **Refactor simulation logic** to support multiple case types via:
  - `#define` macros;
  - Encapsulation of case-specific setup (inflow/boundary conditions, initial fields, etc.).

### 🌊 3. Boundary Conditions

- [ ] **Implement boundary reconstruction** to enable physical boundary behavior:
  - The simulation operates strictly on the bulk domain.
  - Therefore, boundary types must be handled via reconstruction:
    - [ ] **Periodic** behavior on lateral faces (`x` and `y` directions);
    - [ ] **Outflow** at the domain exit (`z = NZ - 1`);
  - [x] **Inflow** already implemented explicitly at `z = 0`.

### 🔬 4. Physics Extensions

- [ ] **Introduce a thermal model** into the LBM core:
  - Purpose: enable simulation of **thermal effects** in multicomponent flows.
  - Strategy:
    - Add a new scalar distribution for temperature;
    - Couple viscosity/surface tension to temperature and component.

- [ ] **Associate physical properties** to each fluid component:
  - Assign **oil** properties to the injected jet and **water** to the background medium.

- [ ] **Allow dynamic oil properties**:
  - Parametrize oil characteristics (density, viscosity, surface tension) for multiple types or API grades;
  - Possibly via external config or compile-time macros.

### 📦 5. Code Usability

- [ ] Move post-processing to GPU for better performance and integration:
  - Implement CUDA kernels for derived quantities (e.g., vorticity, curvature, phase gradients);
  - Reduce host-device transfers by avoiding CPU-side analysis;
  - Automate variable detection;
  - Maintain flexibility for future extensions (e.g., temperature field in thermal models).

---

Feel free to contribute, discuss or pick up tasks from this list!
