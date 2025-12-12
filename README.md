# 2D Heat Equation Parallel Solver

**Parallel Computing Project**  
*Saint Joseph University of Beirut - Department of Informatics Engineering and Communications*

[![MPI](https://img.shields.io/badge/MPI-MPICH-blue.svg)](https://www.mpich.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-Parallel-green.svg)](https://www.openmp.org/)
[![C](https://img.shields.io/badge/Language-C-brightgreen.svg)](https://en.wikipedia.org/wiki/C_(programming_language))

---

## 📋 Overview

This project implements **parallel solvers for the 2D transient heat conduction equation** using both **MPI** (distributed memory) and **OpenMP** (shared memory) approaches. The implementation demonstrates advanced parallel computing techniques including domain decomposition, non-blocking communication, and performance optimization.

### Key Features

- ✅ **Dual implementations**: MPI for distributed systems + OpenMP for shared memory
- ✅ **2D Cartesian topology** with automatic domain decomposition
- ✅ **Non-blocking MPI communication** for optimal performance
- ✅ **Halo exchange** for boundary synchronization
- ✅ **Comprehensive performance analysis** with detailed timing
- ✅ **Scientific visualization** with Python/Matplotlib
- ✅ **Tested on multi-machine cluster** (3 laptops)

---

## 🔬 Problem Description

We solve the **2D transient heat conduction equation**:

```
∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
```

**Where:**
- `T(x,y,t)` - Temperature field
- `α = 0.1 m²/s` - Thermal diffusivity
- `(x, y)` - Spatial coordinates  
- `t` - Time

### Configuration

| Parameter | Value |
|-----------|-------|
| Grid size | 200 × 200 |
| Time steps | 10,000 |
| Snapshot interval | 200 steps |
| dx, dy | 1.0 m |
| CFL factor | 0.2 |
| Initial condition | Gaussian hot spot at center |
| Boundary conditions | Dirichlet (T=0 on all edges) |

---

## 📐 Mathematical Background

### Discretization

**5-point stencil (centered finite differences)**:

```
T[i,j]^(n+1) = T[i,j]^n + α·Δt·(
    (T[i-1,j]^n - 2T[i,j]^n + T[i+1,j]^n)/Δx² +
    (T[i,j-1]^n - 2T[i,j]^n + T[i,j+1]^n)/Δy²
)
```

### Stability Condition

**Von Neumann stability analysis** for explicit 2D diffusion:

```
Δt ≤ Δx²/(4α)
```

**Implementation**: `dt = 0.2 × (dx²/4α)` for safety margin

---

## 🚀 Implementation Details

### MPI Version (`heat_mpi.c`)

**Features:**
- 2D Cartesian process topology (`MPI_Cart_create`)
- Automatic grid partitioning (`MPI_Dims_create`)
- Halo cells for boundary data
- Non-blocking communication (`MPI_Isend`, `MPI_Irecv`)
- Overlap of computation and communication
- Rank 0 handles all I/O operations

**Communication Pattern:**
```
Each process exchanges halos with 4 neighbors (North, South, East, West):

┌─────────┬─────────┬─────────┐
│ Rank 0  │ Rank 1  │ Rank 2  │  Example: 3×3 grid
├─────────┼─────────┼─────────┤  (9 processes)
│ Rank 3  │ Rank 4  │ Rank 5  │
├─────────┼─────────┼─────────┤
│ Rank 6  │ Rank 7  │ Rank 8  │
└─────────┴─────────┴─────────┘
```

**Algorithm:**
1. Apply boundary conditions on domain edges
2. Start non-blocking halo exchange
3. Update interior cells (independent of halos)
4. Wait for communication completion
5. Update boundary cells (dependent on halos)
6. Gather and save snapshots (rank 0 only)

### OpenMP Version (`heat_parallel.c`)

**Features:**
- Shared-memory parallelization
- `#pragma omp parallel for collapse(2)` for 2D loops
- Thread-safe I/O operations
- Detailed performance instrumentation
- Automatic thread count detection

**Performance Metrics:**
- Total execution time
- Computation time breakdown
- Boundary update time
- I/O time
- GFLOPS calculation
- Memory bandwidth estimation

---

## 🛠️ Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gcc mpich make python3 python3-pip

# Python dependencies for visualization
pip3 install numpy matplotlib
```

### Compilation

```bash
# MPI version
mpicc -O3 -o heat_mpi heat_mpi.c -lm

# OpenMP version
gcc -fopenmp -O3 -o heat_omp heat_parallel.c -lm
```

---

## 📖 Usage

### OpenMP Version (Single Machine)

```bash
# Use all available cores
./heat_omp

# Specify thread count
./heat_omp 4

# Or via environment variable
export OMP_NUM_THREADS=8
./heat_omp
```

### MPI Version (Single Machine)

```bash
# Run with 4 processes
mpirun -np 4 ./heat_mpi
```

### MPI Version (Multi-Machine Cluster)

**1. Setup hostfile** (`hostfile.txt`):
```
10.100.100.100 slots=2
10.100.100.101 slots=2
10.100.100.102 slots=2
```

**2. Configure passwordless SSH** between all machines:
```bash
ssh-keygen -t rsa
ssh-copy-id user@10.100.100.100
ssh-copy-id user@10.100.100.101
ssh-copy-id user@10.100.100.102
```

**3. Run the simulation**:
```bash
# Disable X11 display (important!)
export DISPLAY=

# Run with 4 processes (2×2 grid)
mpirun -np 4 -f hostfile.txt ./heat_mpi

# Or with 8 processes (2×4 grid)
mpirun -np 8 -f hostfile.txt ./heat_mpi
```

**Note:** Grid size (200×200) must be divisible by process grid dimensions.  
Valid process counts: 1, 2, 4, 8, 10, 16, 20, 25, 40, 50, 100, 200

---

## 📊 Performance Results

### Test Configuration

- **Hardware**: 3 laptops
- **Network**: 1 Gbps Ethernet
- **MPI**: MPICH 3.x
- **Grid**: 200 × 200 points
- **Time steps**: 10,000
- **Snapshots**: 51 CSV files

### MPI Results (4 Processes, Multi-Machine)

| Component | Time (s) | Percentage |
|-----------|----------|------------|
| **Total** | 258.48 | 100.0% |
| Computation | 0.28 | 0.1% |
| Communication | 255.05 | 98.7% |
| I/O | 19.70 | 7.6% |

**Performance Metrics:**
- Grid updates per second: **1.55 × 10⁶**
- Time per iteration: **0.026 seconds**

### Analysis

**High communication overhead** (98.7%) indicates:
- Network latency dominates for small grids
- Communication-to-computation ratio unfavorable
- Ethernet network bandwidth limitation

### OpenMP Results (Example: 4 Threads)

*Run locally to compare and add results here*

---

## 🎨 Visualization

### Generate Visualizations

```bash
cd /path/to/output/directory

# Run visualization script
python3 visualize.py
```

**Options:**
1. **View final output** - Single heatmap
2. **Create animation** - GIF/MP4 animation
3. **Analyze final state** - 3-panel analysis
4. **Compare timesteps** - Side-by-side comparison

### Output Files

The simulation generates:
- **51 CSV snapshots**: `output_step_00000.csv` to `output_step_10000.csv`
- **1 final output**: `final_output.csv`
- **File size**: ~320 KB per 200×200 grid

### Visualization Examples

**Final Temperature Distribution:**
- Hot center (initial Gaussian)
- Cool boundaries (T=0)
- Smooth radial gradient
- Symmetric circular pattern

---

## 📁 Project Structure

```
heat-equation-parallel/
├── heat_mpi.c                 # MPI implementation
├── heat_parallel.c            # OpenMP implementation
├── visualize.py               # Visualization script
├── README.md                  # This file

---

## 🔧 Troubleshooting

### Common Issues

#### 1. X11 Authorization Error (MPI)

**Error:** `Authorization required, but no authorization protocol specified`

**Fix:**
```bash
export DISPLAY=
mpirun -np 4 -f hostfile.txt ./heat_mpi
```

#### 2. Grid Divisibility Error

**Error:** `NX=200 must be divisible by Px=3`

**Fix:** Use process counts that divide 200 evenly (1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200)

```bash
# Good (2×2 = 4 processes)
mpirun -np 4 -f hostfile.txt ./heat_mpi

# Bad (3 processes creates 3×1 or 1×3 grid)
mpirun -np 3 -f hostfile.txt ./heat_mpi  # ERROR!
```

#### 3. SSH Password Prompt

**Fix:** Setup passwordless SSH (see Installation section)

#### 4. No CSV Files Generated

**Check:**
```bash
# Where are the files?
find ~ -name "*.csv" -type f -mmin -30

# Check working directory
pwd
ls -lh *.csv
```

**Fix:** Use `--wdir` flag:
```bash
mpirun -np 4 -f hostfile.txt --wdir ~/mpi_final ./heat_mpi
```

---

## 🧪 Validation

### Correctness Checks

```python
import numpy as np

# Load results
T = np.loadtxt('final_output.csv', delimiter=',')

# 1. Check boundaries (should be ~0)
print(f"Max edge value: {max(np.max(np.abs(T[0,:])), np.max(np.abs(T[-1,:])))}")

# 2. Check symmetry
print(f"Horizontal symmetry: {np.allclose(T, np.fliplr(T), atol=1e-6)}")
print(f"Vertical symmetry: {np.allclose(T, np.flipud(T), atol=1e-6)}")

# 3. Energy conservation (should decrease)
T0 = np.loadtxt('output_step_00000.csv', delimiter=',')
print(f"Initial energy: {np.sum(T0):.2f}")
print(f"Final energy: {np.sum(T):.2f}")
print(f"Ratio: {np.sum(T)/np.sum(T0):.4f}")
```

---

## 👥 Team

**Team Members:**
- Christian el helou - MPI implementation
- Charbel el khoury - OpenMP implementation  
- Elias chalhoub - Visualization & testing
- Emile Bou Faissal - Performance analysis & documentation

**Institution:** Saint Joseph University of Beirut  
**Course:** Parallel Computing  
**Semester:** Fall 2025 
**Instructor:** DR Maroun Ayli

---

## 📚 References

1. Chapra, S. C., & Canale, R. P. (2015). *Numerical Methods for Engineers* (7th ed.). McGraw-Hill.
2. Gropp, W., Lusk, E., & Skjellum, A. (2014). *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (3rd ed.). MIT Press.
3. MPI Forum. (2021). *MPI: A Message-Passing Interface Standard Version 4.0*.
4. OpenMP Architecture Review Board. (2021). *OpenMP Application Programming Interface Version 5.2*.


## 🙏 Acknowledgments

- **Professor [DR Maroun Ayli]** for project guidance
- **Saint Joseph University** for computational resources
- **MPICH & OpenMP communities** for excellent tools and documentation
- **NumPy & Matplotlib teams** for visualization tools