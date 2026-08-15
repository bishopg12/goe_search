# Game of Life - Garden of Eden (GOE) Search

This repository contains the source code for the highly-optimized **Conway's Game of Life - Garden of Eden (GOE)** state space search project. The goal of this project is to exhaustively map and evaluate the massively complex state spaces of grid-based environments (specifically the 7x7 and 7x8 grids) using advanced algorithmic pruning and High-Performance Computing (HPC) clusters to find Garden of Eden patterns (states with no predecessors).

## 🚀 Overview

The state space for a 7x7 GOE grid contains roughly 560 Trillion ($5.6 \times 10^{14}$) reachable states, while the 7x8 grid expands to an astronomical 72 Quadrillion ($7.2 \times 10^{16}$) states. Evaluating these states requires traversing a graph via Breadth-First Search (BFS) and identifying unique canonical forms to prevent combinatorial explosion.

To achieve this, the engine is heavily parallelized across three dimensions:
- **MPI (Message Passing Interface):** Distributes the global hash space across multiple discrete compute nodes, allowing linear horizontal scaling without duplicating work.
- **OpenMP:** Fully saturates local CPU cores with shared-memory multi-threading for rapid state evaluations and bitwise operations.
- **AMD HIP (ROCm):** Offloads the heaviest BFS evaluation kernels to AMD Instinct GPUs, achieving massive throughput by processing millions of states concurrently.

## 📂 Project Structure

### Search Engines
- **`goe_search_7x7.c` / `.cpp`:** The core CPU-based search engine for the 7x7 grid. Utilizes MPI for distribution and OpenMP for local thread pools.
- **`goe_search_7x7.hip` (v2-v7):** GPU-accelerated iterations of the 7x7 search. Features various levels of optimization including shared memory caching and warp-level primitives.
- **`goe_search_7x8.hip`:** The GPU-accelerated engine designed specifically to tackle the larger 7x8 grid constraints.

### Utilities and Testing
- **`generate_cnf.py`:** A Python utility for generating Conjunctive Normal Form (CNF) constraints for SAT-based logic evaluations.
- **`verify_compression.py`:** Validates state compression logic and serialization integrity.
- **`count_unique.c` / `count_uniq`:** Fast C utilities for analyzing and deduping output state files.
- **`test_*.c / .cpp / .hip`:** Micro-benchmarks and unit tests validating canonical representation logic, distribution integrity, and shared memory access speeds.

## 🛠 Prerequisites

To compile and run this project, you will need access to an HPC environment with the following:
- **MPI Implementation:** OpenMPI or MPICH with `mpicc` / `mpicxx`.
- **Compiler:** GCC 9.0+ (with OpenMP support).
- **GPU Toolkit:** AMD ROCm toolkit (`hipcc`) for compiling `.hip` files.
- **Resource Manager (Optional):** Slurm with PMIX support is recommended for multi-node deployments.

## 🏗 Compilation

### CPU Only (7x7)
Compile using `mpicc`, ensuring OpenMP is enabled and the architecture is optimized for the host machine.
```bash
mpicc -O3 -fopenmp -march=native goe_search/goe_search_7x7.c -o goe_search_7x7_cpu
```

### GPU / Heterogeneous (HIP)
Compile the HIP kernels using `hipcc`. You must link MPI libraries manually depending on your local MPI installation paths.
```bash
hipcc -O3 -I/usr/lib/x86_64-linux-gnu/openmpi/include \
      -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi \
      goe_search/goe_search_7x7_v7.hip -o goe_search_7x7_hip
```

## 🏃 Execution

### Multi-Node CPU Search
Run across multiple nodes using `srun` (Slurm) or `mpirun`. Ensure CPU binding is disabled so OpenMP can freely spawn threads across all allocated cores.
```bash
# Example Slurm execution (8 nodes, 16 cores per node)
export OMP_NUM_THREADS=16
srun -n 8 --cpus-per-task=16 --cpu-bind=none --mpi=pmix ./goe_search_7x7_cpu
```

### Heterogeneous Deployment (CPU + GPU)
You can deploy a heterogeneous search where standard compute nodes run the OpenMP CPU executable, and GPU-equipped nodes run the HIP executable. This is achieved using Slurm's Heterogeneous Job feature (`pack-group`).
```bash
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_btl_tcp_if_include="<YOUR_SUBNET_IP/MASK>" # Ensure MPI traffic stays on internal fast fabric

srun --mpi=pmix \
     --pack-group=0 ./goe_search_7x7_cpu : \
     --pack-group=1 ./goe_search_7x7_v7_hip
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
