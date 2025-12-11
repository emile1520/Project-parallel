/*
 * ============================================================================
 * 2D Heat Equation Solver - OpenMP Parallel Implementation
 * ============================================================================
 * 
 * Solves the 2D transient heat conduction equation:
 *     ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
 * 
 * Discretization:
 *     - Spatial: Centered finite differences (5-point stencil)
 *     - Temporal: Explicit (Forward) Euler method
 *     - Stability: dt ≤ dx²/(4α) for 2D diffusion
 * 
 * Boundary Conditions:
 *     - Dirichlet: T = 0 on all four edges
 * 
 * Initial Condition:
 *     - Gaussian hot spot: T(x,y,0) = exp(-((x-xc)² + (y-yc)²)/σ²)
 * 
 * Parallelization:
 *     - OpenMP shared-memory parallelism
 *     - Parallel loops with collapse(2) directive
 *     - Thread-safe I/O operations
 * 
 * Compilation:
 *     gcc -fopenmp -O3 -o heat_omp heat_parallel.c -lm
 * 
 * Usage:
 *     ./heat_omp [threads]
 *     export OMP_NUM_THREADS=4 && ./heat_omp
 * 
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

// ============================================================================
// Configuration Parameters
// ============================================================================

#define NX 200                      // Grid points in x-direction
#define NY 200                      // Grid points in y-direction
#define NSTEPS 10000                // Number of time steps
#define SNAPSHOT_INTERVAL 200       // Save output every N steps

// Physical parameters
#define ALPHA 0.1                   // Thermal diffusivity [m²/s]
#define DX 1.0                      // Grid spacing in x [m]
#define DY 1.0                      // Grid spacing in y [m]
#define CFL 0.2                     // CFL stability factor (< 0.25 for 2D)

// Initial condition parameters
#define IC_SIGMA 200.0              // Gaussian width parameter

// Output configuration
#define OUTPUT_PREFIX "output_step" // CSV output file prefix
#define FINAL_OUTPUT "final_output.csv"
#define TIMING_OUTPUT "timing_omp.txt"

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    double total_time;
    double compute_time;
    double io_time;
    double init_time;
    double boundary_time;
} TimingStats;

typedef struct {
    int nx, ny;
    int nsteps;
    double dx, dy, dt;
    double alpha;
    double simulated_time;
} SimulationParams;

// ============================================================================
// Function Prototypes
// ============================================================================

void initialize_grid(double *T, double *Tnew, int nx, int ny);
void apply_boundary_conditions(double *T, int nx, int ny);
void compute_heat_equation(double *T, double *Tnew, int nx, int ny, 
                          double alpha, double dt, double dx, double dy);
void save_csv(const double *T, int nx, int ny, int step);
void save_timing_stats(const TimingStats *stats, const SimulationParams *params);
void print_header(const SimulationParams *params);
void print_timing_summary(const TimingStats *stats, const SimulationParams *params);
double calculate_stability_dt(double dx, double dy, double alpha, double cfl);

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char **argv) {
    
    // ========================================================================
    // Initialize OpenMP
    // ========================================================================
    
    // Set number of threads if provided as argument
    if (argc > 1) {
        int nthreads = atoi(argv[1]);
        if (nthreads > 0) {
            omp_set_num_threads(nthreads);
        }
    }
    
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }
    
    // ========================================================================
    // Setup Simulation Parameters
    // ========================================================================
    
    SimulationParams params;
    params.nx = NX;
    params.ny = NY;
    params.nsteps = NSTEPS;
    params.dx = DX;
    params.dy = DY;
    params.alpha = ALPHA;
    params.dt = calculate_stability_dt(DX, DY, ALPHA, CFL);
    params.simulated_time = params.dt * NSTEPS;
    
    print_header(&params);
    printf("Number of OpenMP threads: %d\n", num_threads);
    printf("Maximum threads available: %d\n", omp_get_max_threads());
    printf("\n");
    
    // ========================================================================
    // Allocate Memory
    // ========================================================================
    
    double init_start = omp_get_wtime();
    
    size_t grid_size = (size_t)NX * NY;
    double *T    = (double *)malloc(grid_size * sizeof(double));
    double *Tnew = (double *)malloc(grid_size * sizeof(double));
    
    if (!T || !Tnew) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        fprintf(stderr, "Requested: %.2f MB per array\n", 
                (grid_size * sizeof(double)) / (1024.0 * 1024.0));
        return EXIT_FAILURE;
    }
    
    printf("Memory allocated: %.2f MB (2 arrays)\n", 
           (2.0 * grid_size * sizeof(double)) / (1024.0 * 1024.0));
    
    // ========================================================================
    // Initialize Grid
    // ========================================================================
    
    initialize_grid(T, Tnew, NX, NY);
    apply_boundary_conditions(T, NX, NY);
    apply_boundary_conditions(Tnew, NX, NY);
    
    double init_end = omp_get_wtime();
    
    // ========================================================================
    // Time Stepping Loop
    // ========================================================================
    
    printf("\nStarting time integration...\n");
    printf("Progress: ");
    fflush(stdout);
    
    TimingStats stats = {0};
    stats.init_time = init_end - init_start;
    
    double main_start = omp_get_wtime();
    
    for (int step = 0; step <= NSTEPS; step++) {
        
        // Progress indicator
        if (step % (NSTEPS / 10) == 0 && step > 0) {
            printf("%d%% ", (int)(100.0 * step / NSTEPS));
            fflush(stdout);
        }
        
        // ====================================================================
        // Compute Heat Equation (Interior Points)
        // ====================================================================
        
        double compute_start = omp_get_wtime();
        
        compute_heat_equation(T, Tnew, NX, NY, ALPHA, params.dt, DX, DY);
        
        double compute_end = omp_get_wtime();
        stats.compute_time += (compute_end - compute_start);
        
        // ====================================================================
        // Apply Boundary Conditions
        // ====================================================================
        
        double boundary_start = omp_get_wtime();
        
        apply_boundary_conditions(Tnew, NX, NY);
        
        double boundary_end = omp_get_wtime();
        stats.boundary_time += (boundary_end - boundary_start);
        
        // ====================================================================
        // Swap Arrays
        // ====================================================================
        
        double *tmp = T;
        T = Tnew;
        Tnew = tmp;
        
        // ====================================================================
        // Save Snapshots
        // ====================================================================
        
        if (step % SNAPSHOT_INTERVAL == 0 || step == NSTEPS) {
            double io_start = omp_get_wtime();
            save_csv(T, NX, NY, step);
            double io_end = omp_get_wtime();
            stats.io_time += (io_end - io_start);
        }
    }
    
    double main_end = omp_get_wtime();
    stats.total_time = main_end - main_start;
    
    printf("100%%\n");
    
    // ========================================================================
    // Save Final Output
    // ========================================================================
    
    printf("\nSaving final output to %s...\n", FINAL_OUTPUT);
    double io_start = omp_get_wtime();
    save_csv(T, NX, NY, -1);
    double io_end = omp_get_wtime();
    stats.io_time += (io_end - io_start);
    
    // ========================================================================
    // Print Performance Statistics
    // ========================================================================
    
    print_timing_summary(&stats, &params);
    save_timing_stats(&stats, &params);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    free(T);
    free(Tnew);
    
    printf("\nSimulation complete!\n");
    
    return EXIT_SUCCESS;
}

// ============================================================================
// Function Implementations
// ============================================================================

/**
 * Calculate stable time step based on CFL condition
 * For 2D explicit diffusion: dt ≤ dx²/(4α)
 */
double calculate_stability_dt(double dx, double dy, double alpha, double cfl) {
    double dx_min = (dx < dy) ? dx : dy;
    double dt_max = dx_min * dx_min / (4.0 * alpha);
    return cfl * dt_max;
}

/**
 * Initialize temperature field with Gaussian hot spot
 */
void initialize_grid(double *T, double *Tnew, int nx, int ny) {
    
    int cx = nx / 2;
    int cy = ny / 2;
    
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double dx_val = i - cx;
            double dy_val = j - cy;
            double r_squared = dx_val * dx_val + dy_val * dy_val;
            double val = exp(-r_squared / IC_SIGMA);
            
            T[j * nx + i] = val;
            Tnew[j * nx + i] = val;
        }
    }
}

/**
 * Apply Dirichlet boundary conditions (T = 0 on all edges)
 */
void apply_boundary_conditions(double *T, int nx, int ny) {
    
    // Top and bottom edges
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        T[0 * nx + i] = 0.0;              // top (j=0)
        T[(ny - 1) * nx + i] = 0.0;       // bottom (j=ny-1)
    }
    
    // Left and right edges
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        T[j * nx + 0] = 0.0;              // left (i=0)
        T[j * nx + (nx - 1)] = 0.0;       // right (i=nx-1)
    }
}

/**
 * Compute heat equation using 5-point stencil
 * 
 * Discretization:
 *     T_new[j,i] = T[j,i] + α·dt·(
 *         (T[j-1,i] - 2T[j,i] + T[j+1,i])/dy² +
 *         (T[j,i-1] - 2T[j,i] + T[j,i+1])/dx²
 *     )
 */
void compute_heat_equation(double *T, double *Tnew, int nx, int ny,
                          double alpha, double dt, double dx, double dy) {
    
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double factor = alpha * dt;
    
    // Update interior points (j ∈ [1, ny-2], i ∈ [1, nx-2])
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            
            const double center = T[j * nx + i];
            const double up     = T[(j - 1) * nx + i];
            const double down   = T[(j + 1) * nx + i];
            const double left   = T[j * nx + (i - 1)];
            const double right  = T[j * nx + (i + 1)];
            
            const double laplacian = 
                (up - 2.0 * center + down) / dy2 +
                (left - 2.0 * center + right) / dx2;
            
            Tnew[j * nx + i] = center + factor * laplacian;
        }
    }
}

/**
 * Save temperature field to CSV file
 */
void save_csv(const double *T, int nx, int ny, int step) {
    
    char filename[128];
    
    if (step < 0) {
        snprintf(filename, sizeof(filename), "%s", FINAL_OUTPUT);
    } else {
        snprintf(filename, sizeof(filename), "%s_%05d.csv", OUTPUT_PREFIX, step);
    }
    
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename);
        return;
    }
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            fprintf(f, "%.6f", T[j * nx + i]);
            if (i < nx - 1) {
                fprintf(f, ",");
            }
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
}

/**
 * Save timing statistics to file
 */
void save_timing_stats(const TimingStats *stats, const SimulationParams *params) {
    
    FILE *f = fopen(TIMING_OUTPUT, "w");
    if (!f) {
        fprintf(stderr, "Warning: Cannot open %s for writing\n", TIMING_OUTPUT);
        return;
    }
    
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }
    
    fprintf(f, "# OpenMP Heat Equation Solver - Timing Results\n");
    fprintf(f, "# ============================================\n\n");
    
    fprintf(f, "Configuration:\n");
    fprintf(f, "  Grid size: %d x %d\n", params->nx, params->ny);
    fprintf(f, "  Time steps: %d\n", params->nsteps);
    fprintf(f, "  Grid spacing: dx=%.3f, dy=%.3f\n", params->dx, params->dy);
    fprintf(f, "  Time step: dt=%.6f\n", params->dt);
    fprintf(f, "  Thermal diffusivity: alpha=%.3f\n", params->alpha);
    fprintf(f, "  Simulated time: %.2f\n", params->simulated_time);
    fprintf(f, "  OpenMP threads: %d\n\n", num_threads);
    
    fprintf(f, "Timing Breakdown:\n");
    fprintf(f, "  Initialization: %.4f s (%.1f%%)\n", 
            stats->init_time, 100.0 * stats->init_time / stats->total_time);
    fprintf(f, "  Computation:    %.4f s (%.1f%%)\n", 
            stats->compute_time, 100.0 * stats->compute_time / stats->total_time);
    fprintf(f, "  Boundaries:     %.4f s (%.1f%%)\n", 
            stats->boundary_time, 100.0 * stats->boundary_time / stats->total_time);
    fprintf(f, "  I/O:            %.4f s (%.1f%%)\n", 
            stats->io_time, 100.0 * stats->io_time / stats->total_time);
    fprintf(f, "  Total:          %.4f s\n\n", stats->total_time);
    
    fprintf(f, "Performance Metrics:\n");
    fprintf(f, "  Grid updates per second: %.2e\n", 
            (double)(params->nx * params->ny * params->nsteps) / stats->total_time);
    fprintf(f, "  Time per iteration: %.6f s\n", 
            stats->total_time / params->nsteps);
    fprintf(f, "  Throughput: %.2f GFLOPS\n",
            (7.0 * (params->nx - 2) * (params->ny - 2) * params->nsteps) / 
            (stats->compute_time * 1e9));
    
    fclose(f);
    
    printf("Timing statistics saved to %s\n", TIMING_OUTPUT);
}

/**
 * Print simulation header
 */
void print_header(const SimulationParams *params) {
    
    printf("\n");
    printf("================================================================================\n");
    printf("                2D Heat Equation Solver - OpenMP Implementation                \n");
    printf("================================================================================\n");
    printf("\n");
    printf("Problem Configuration:\n");
    printf("  Domain size:         %d x %d grid points\n", params->nx, params->ny);
    printf("  Grid spacing:        dx = %.3f m, dy = %.3f m\n", params->dx, params->dy);
    printf("  Thermal diffusivity: α = %.3f m²/s\n", params->alpha);
    printf("  Time step:           dt = %.6f s (CFL = %.2f)\n", params->dt, CFL);
    printf("  Number of steps:     %d\n", params->nsteps);
    printf("  Simulated time:      %.2f s\n", params->simulated_time);
    printf("  Snapshot interval:   every %d steps\n", SNAPSHOT_INTERVAL);
    printf("\n");
    printf("Numerical Method:\n");
    printf("  Spatial:  Centered finite differences (5-point stencil)\n");
    printf("  Temporal: Explicit Euler method\n");
    printf("  Boundary: Dirichlet (T = 0 on all edges)\n");
    printf("  Initial:  Gaussian hot spot at center\n");
    printf("\n");
    printf("Stability Analysis:\n");
    printf("  Stability limit: dt ≤ %.6f s\n", DX * DX / (4.0 * ALPHA));
    printf("  Current dt/limit:    %.2f%% (SAFE)\n", 
           100.0 * params->dt / (DX * DX / (4.0 * ALPHA)));
    printf("\n");
}

/**
 * Print timing summary
 */
void print_timing_summary(const TimingStats *stats, const SimulationParams *params) {
    
    printf("\n");
    printf("================================================================================\n");
    printf("                          Performance Summary                                  \n");
    printf("================================================================================\n");
    printf("\n");
    
    printf("Execution Time Breakdown:\n");
    printf("  %-20s %10.4f s  (%5.1f%%)\n", "Initialization:", 
           stats->init_time, 100.0 * stats->init_time / stats->total_time);
    printf("  %-20s %10.4f s  (%5.1f%%)\n", "Computation:", 
           stats->compute_time, 100.0 * stats->compute_time / stats->total_time);
    printf("  %-20s %10.4f s  (%5.1f%%)\n", "Boundary Updates:", 
           stats->boundary_time, 100.0 * stats->boundary_time / stats->total_time);
    printf("  %-20s %10.4f s  (%5.1f%%)\n", "I/O Operations:", 
           stats->io_time, 100.0 * stats->io_time / stats->total_time);
    printf("  %s\n", "------------------------------------------------------------");
    printf("  %-20s %10.4f s  (%5.1f%%)\n", "TOTAL:", 
           stats->total_time, 100.0);
    printf("\n");
    
    printf("Performance Metrics:\n");
    
    double grid_updates = (double)params->nx * params->ny * params->nsteps;
    printf("  Grid updates:            %.2e\n", grid_updates);
    printf("  Updates per second:      %.2e\n", grid_updates / stats->total_time);
    printf("  Time per iteration:      %.6f s\n", stats->total_time / params->nsteps);
    
    // Compute FLOPS (7 operations per interior point per iteration)
    double interior_points = (params->nx - 2) * (params->ny - 2);
    double total_flops = 7.0 * interior_points * params->nsteps;
    double gflops = total_flops / (stats->compute_time * 1e9);
    printf("  Computational GFLOPS:    %.2f\n", gflops);
    
    // Memory bandwidth estimate
    // Each update: 5 reads + 1 write = 6 doubles per point
    double bytes_per_update = 6.0 * sizeof(double) * interior_points;
    double total_bytes = bytes_per_update * params->nsteps;
    double bandwidth_gb_s = total_bytes / (stats->compute_time * 1e9);
    printf("  Memory bandwidth:        %.2f GB/s\n", bandwidth_gb_s);
    
    printf("\n");
    
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }
    
    printf("Parallel Efficiency:\n");
    printf("  Number of threads:       %d\n", num_threads);
    printf("  Work per thread:         %.2e grid updates\n", 
           grid_updates / num_threads);
    
    printf("\n");
    printf("Output Files:\n");
    printf("  CSV snapshots:           %s_*.csv\n", OUTPUT_PREFIX);
    printf("  Final output:            %s\n", FINAL_OUTPUT);
    printf("  Timing data:             %s\n", TIMING_OUTPUT);
    printf("\n");
}