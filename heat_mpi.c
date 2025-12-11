#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NX 200              // global number of points in x
#define NY 200              // global number of points in y
#define NSTEPS 10000        // Reduced for faster testing
#define SNAPSHOT_INTERVAL 200  // More snapshots for better animation

// Physical parameters
const double alpha = 0.1;
const double dx = 1.0;
const double dy = 1.0;
const double CFL = 0.2;      // stability safety factor for explicit 2D diffusion

// Macro for indexing local 2D array with halos
// j = 0..local_ny+1, i = 0..local_nx+1
#define IDX(j,i,local_nx) ((j) * ((local_nx) + 2) + (i))

void save_csv(double *T, int step) {
    char filename[64];
    if (step < 0) sprintf(filename, "final_output.csv");
    else          sprintf(filename, "output_step_%05d.csv", step);

    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening %s for writing\n", filename);
        return;
    }

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            fprintf(f, "%.6f", T[j * NX + i]);
            if (i < NX - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double start_time = MPI_Wtime(); // Start timing

    // ------------------------------------------------------------
    // 1) Create 2D Cartesian topology
    // ------------------------------------------------------------
    int dims[2] = {0, 0};      // dims[0] = Py (rows), dims[1] = Px (cols)
    MPI_Dims_create(world_size, 2, dims);

    int periods[2] = {0, 0};   // non-periodic
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);

    int rank, size;
    MPI_Comm_rank(cart, &rank);
    MPI_Comm_size(cart, &size);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int py = coords[0];        // row index in process grid
    int px = coords[1];        // col index in process grid

    int Py = dims[0];
    int Px = dims[1];

    if (rank == 0) {
        printf("=== MPI 2D Heat Equation Solver ===\n");
        printf("Process grid: Py = %d, Px = %d (total: %d)\n", Py, Px, size);
        printf("Global domain: NX = %d, NY = %d\n", NX, NY);
    }

    // Check divisibility for simple block decomposition
    if (NX % Px != 0 || NY % Py != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: NX=%d must be divisible by Px=%d, and NY=%d by Py=%d\n",
                    NX, Px, NY, Py);
        }
        MPI_Abort(cart, 1);
    }

    // Local interior sizes
    int local_nx = NX / Px;
    int local_ny = NY / Py;

    if (rank == 0) {
        printf("Local subdomain per process: %d x %d\n", local_nx, local_ny);
    }

    // Global start indices of this block (interior, no halos)
    int global_i_start = px * local_nx;
    int global_j_start = py * local_ny;

    // ------------------------------------------------------------
    // 2) Allocate local arrays with halo cells
    // ------------------------------------------------------------
    int local_size = (local_nx + 2) * (local_ny + 2);
    double *T    = (double *)calloc(local_size, sizeof(double));
    double *Tnew = (double *)calloc(local_size, sizeof(double));

    if (!T || !Tnew) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        MPI_Abort(cart, 2);
    }

    // Time step from stability condition for explicit 2D heat equation
    double dt = CFL * dx * dx / (4.0 * alpha);
    if (rank == 0) {
        printf("Time step dt = %.6f (CFL = %.2f, alpha = %.2f)\n", dt, CFL, alpha);
        printf("Total time steps: %d, snapshots every %d steps\n", NSTEPS, SNAPSHOT_INTERVAL);
        printf("Simulated time: %.2f\n\n", NSTEPS * dt);
    }

    // ------------------------------------------------------------
    // 3) Initial condition: Gaussian hot spot at global center
    // ------------------------------------------------------------
    int cx = NX / 2;
    int cy = NY / 2;

    for (int j = 1; j <= local_ny; j++) {
        int gj = global_j_start + (j - 1);   // global y index
        for (int i = 1; i <= local_nx; i++) {
            int gi = global_i_start + (i - 1); // global x index
            double dxg = gi - cx;
            double dyg = gj - cy;
            double val = exp(-(dxg*dxg + dyg*dyg) / 200.0);

            T[IDX(j,i,local_nx)]    = val;
            Tnew[IDX(j,i,local_nx)] = val;
        }
    }

    // ------------------------------------------------------------
    // 4) Neighbor ranks in 2D grid: N, S, W, E
    // ------------------------------------------------------------
    int north, south, west, east;
    MPI_Cart_shift(cart, 0, 1, &north, &south); // along y (rows)
    MPI_Cart_shift(cart, 1, 1, &west,  &east);  // along x (cols)

    // ------------------------------------------------------------
    // 5) Buffers for column halo exchange and gathering
    // ------------------------------------------------------------
    double *send_west  = (double *)malloc(local_ny * sizeof(double));
    double *send_east  = (double *)malloc(local_ny * sizeof(double));
    double *recv_west  = (double *)malloc(local_ny * sizeof(double));
    double *recv_east  = (double *)malloc(local_ny * sizeof(double));

    if (!send_west || !send_east || !recv_west || !recv_east) {
        fprintf(stderr, "Rank %d: column buffer allocation failed\n", rank);
        MPI_Abort(cart, 3);
    }

    // Buffer for gathering interior blocks
    double *sendbuf = (double *)malloc(local_nx * local_ny * sizeof(double));
    double *gatherbuf = NULL;
    if (rank == 0) {
        gatherbuf = (double *)malloc(local_nx * local_ny * size * sizeof(double));
        if (!gatherbuf) {
            fprintf(stderr, "Root: gather buffer allocation failed\n");
            MPI_Abort(cart, 4);
        }
    }

    MPI_Request reqs[8]; // up to 8 non-blocking comms (4 sends + 4 recvs)

    double compute_time = 0.0;
    double comm_time = 0.0;
    double io_time = 0.0;

    // ------------------------------------------------------------
    // 6) Time stepping
    // ------------------------------------------------------------
    for (int step = 0; step <= NSTEPS; step++) {

        int nreq = 0;
        double step_comm_start = MPI_Wtime();

        // --------------------------------------------------------
        // 6.1: Apply Dirichlet BC (T=0) on ghost layers at global edges
        //      This sets boundary CONDITIONS, not interior values!
        // --------------------------------------------------------
        if (north == MPI_PROC_NULL) {
            for (int i = 0; i <= local_nx + 1; i++) {
                T[IDX(0, i, local_nx)] = 0.0;
            }
        }
        if (south == MPI_PROC_NULL) {
            for (int i = 0; i <= local_nx + 1; i++) {
                T[IDX(local_ny + 1, i, local_nx)] = 0.0;
            }
        }
        if (west == MPI_PROC_NULL) {
            for (int j = 0; j <= local_ny + 1; j++) {
                T[IDX(j, 0, local_nx)] = 0.0;
            }
        }
        if (east == MPI_PROC_NULL) {
            for (int j = 0; j <= local_ny + 1; j++) {
                T[IDX(j, local_nx + 1, local_nx)] = 0.0;
            }
        }

        // --------------------------------------------------------
        // 6.2: Start non-blocking halo exchange (rows + columns)
        // --------------------------------------------------------
        if (north != MPI_PROC_NULL) {
            MPI_Isend(&T[IDX(1, 1, local_nx)], local_nx, MPI_DOUBLE,
                      north, 0, cart, &reqs[nreq++]);
            MPI_Irecv(&T[IDX(0, 1, local_nx)], local_nx, MPI_DOUBLE,
                      north, 1, cart, &reqs[nreq++]);
        }

        if (south != MPI_PROC_NULL) {
            MPI_Isend(&T[IDX(local_ny, 1, local_nx)], local_nx, MPI_DOUBLE,
                      south, 1, cart, &reqs[nreq++]);
            MPI_Irecv(&T[IDX(local_ny + 1, 1, local_nx)], local_nx, MPI_DOUBLE,
                      south, 0, cart, &reqs[nreq++]);
        }

        if (west != MPI_PROC_NULL) {
            for (int j = 1; j <= local_ny; j++) {
                send_west[j - 1] = T[IDX(j, 1, local_nx)];
            }
            MPI_Isend(send_west, local_ny, MPI_DOUBLE,
                      west, 2, cart, &reqs[nreq++]);
            MPI_Irecv(recv_west, local_ny, MPI_DOUBLE,
                      west, 3, cart, &reqs[nreq++]);
        }

        if (east != MPI_PROC_NULL) {
            for (int j = 1; j <= local_ny; j++) {
                send_east[j - 1] = T[IDX(j, local_nx, local_nx)];
            }
            MPI_Isend(send_east, local_ny, MPI_DOUBLE,
                      east, 3, cart, &reqs[nreq++]);
            MPI_Irecv(recv_east, local_ny, MPI_DOUBLE,
                      east, 2, cart, &reqs[nreq++]);
        }

        double step_compute_start = MPI_Wtime();
        comm_time += (step_compute_start - step_comm_start);

        // --------------------------------------------------------
        // 6.3: Compute interior (independent of halo data)
        // --------------------------------------------------------
        for (int j = 2; j <= local_ny - 1; j++) {
            for (int i = 2; i <= local_nx - 1; i++) {
                double center = T[IDX(j, i, local_nx)];
                double up     = T[IDX(j - 1, i, local_nx)];
                double down   = T[IDX(j + 1, i, local_nx)];
                double left   = T[IDX(j, i - 1, local_nx)];
                double right  = T[IDX(j, i + 1, local_nx)];

                Tnew[IDX(j, i, local_nx)] =
                    center + alpha * dt * (
                        (up    - 2.0 * center + down)  / (dy * dy) +
                        (left  - 2.0 * center + right) / (dx * dx)
                    );
            }
        }

        // --------------------------------------------------------
        // 6.4: Wait for halo communications
        // --------------------------------------------------------
        double wait_start = MPI_Wtime();
        if (nreq > 0) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // Unpack column data
        if (west != MPI_PROC_NULL) {
            for (int j = 1; j <= local_ny; j++) {
                T[IDX(j, 0, local_nx)] = recv_west[j - 1];
            }
        }
        if (east != MPI_PROC_NULL) {
            for (int j = 1; j <= local_ny; j++) {
                T[IDX(j, local_nx + 1, local_nx)] = recv_east[j - 1];
            }
        }

        double boundary_compute_start = MPI_Wtime();
        comm_time += (boundary_compute_start - wait_start);

        // --------------------------------------------------------
        // 6.5: Compute boundary interior cells (depend on halo)
        // --------------------------------------------------------
        // Top row (j=1)
        if (local_ny >= 1) {
            int j = 1;
            for (int i = 1; i <= local_nx; i++) {
                double center = T[IDX(j, i, local_nx)];
                double up     = T[IDX(j - 1, i, local_nx)];
                double down   = T[IDX(j + 1, i, local_nx)];
                double left   = T[IDX(j, i - 1, local_nx)];
                double right  = T[IDX(j, i + 1, local_nx)];

                Tnew[IDX(j, i, local_nx)] =
                    center + alpha * dt * (
                        (up    - 2.0 * center + down)  / (dy * dy) +
                        (left  - 2.0 * center + right) / (dx * dx)
                    );
            }
        }

        // Bottom row (j=local_ny)
        if (local_ny > 1) {
            int j = local_ny;
            for (int i = 1; i <= local_nx; i++) {
                double center = T[IDX(j, i, local_nx)];
                double up     = T[IDX(j - 1, i, local_nx)];
                double down   = T[IDX(j + 1, i, local_nx)];
                double left   = T[IDX(j, i - 1, local_nx)];
                double right  = T[IDX(j, i + 1, local_nx)];

                Tnew[IDX(j, i, local_nx)] =
                    center + alpha * dt * (
                        (up    - 2.0 * center + down)  / (dy * dy) +
                        (left  - 2.0 * center + right) / (dx * dx)
                    );
            }
        }

        // Left column (i=1), excluding corners
        if (local_nx >= 1) {
            int i = 1;
            for (int j = 2; j <= local_ny - 1; j++) {
                double center = T[IDX(j, i, local_nx)];
                double up     = T[IDX(j - 1, i, local_nx)];
                double down   = T[IDX(j + 1, i, local_nx)];
                double left   = T[IDX(j, i - 1, local_nx)];
                double right  = T[IDX(j, i + 1, local_nx)];

                Tnew[IDX(j, i, local_nx)] =
                    center + alpha * dt * (
                        (up    - 2.0 * center + down)  / (dy * dy) +
                        (left  - 2.0 * center + right) / (dx * dx)
                    );
            }
        }

        // Right column (i=local_nx), excluding corners
        if (local_nx > 1) {
            int i = local_nx;
            for (int j = 2; j <= local_ny - 1; j++) {
                double center = T[IDX(j, i, local_nx)];
                double up     = T[IDX(j - 1, i, local_nx)];
                double down   = T[IDX(j + 1, i, local_nx)];
                double left   = T[IDX(j, i - 1, local_nx)];
                double right  = T[IDX(j, i + 1, local_nx)];

                Tnew[IDX(j, i, local_nx)] =
                    center + alpha * dt * (
                        (up    - 2.0 * center + down)  / (dy * dy) +
                        (left  - 2.0 * center + right) / (dx * dx)
                    );
            }
        }

        double step_compute_end = MPI_Wtime();
        compute_time += (step_compute_end - boundary_compute_start);

        // --------------------------------------------------------
        // 6.6: Swap arrays
        // --------------------------------------------------------
        double *tmp = T;
        T = Tnew;
        Tnew = tmp;

        // --------------------------------------------------------
        // 6.7: Gather and save snapshots
        // --------------------------------------------------------
        if (step % SNAPSHOT_INTERVAL == 0 || step == NSTEPS) {
            double io_start = MPI_Wtime();

            // Pack interior data (no halos)
            for (int j = 0; j < local_ny; j++) {
                for (int i = 0; i < local_nx; i++) {
                    sendbuf[j * local_nx + i] = T[IDX(j + 1, i + 1, local_nx)];
                }
            }

            MPI_Gather(sendbuf, local_nx * local_ny, MPI_DOUBLE,
                       gatherbuf, local_nx * local_ny, MPI_DOUBLE,
                       0, cart);

            if (rank == 0) {
                double *full = (double *)malloc(NX * NY * sizeof(double));
                if (!full) {
                    fprintf(stderr, "Root: allocation for full field failed\n");
                    MPI_Abort(cart, 5);
                }

                // Reconstruct global field
                for (int r = 0; r < size; r++) {
                    int rc[2];
                    MPI_Cart_coords(cart, r, 2, rc);
                    int r_py = rc[0];
                    int r_px = rc[1];

                    int gj_start = r_py * local_ny;
                    int gi_start = r_px * local_nx;

                    double *block = &gatherbuf[r * local_nx * local_ny];

                    for (int jj = 0; jj < local_ny; jj++) {
                        for (int ii = 0; ii < local_nx; ii++) {
                            full[(gj_start + jj) * NX + (gi_start + ii)] = 
                                block[jj * local_nx + ii];
                        }
                    }
                }

                save_csv(full, (step == NSTEPS) ? -1 : step);
                printf("Step %5d/%d saved (time: %.2f)\n", step, NSTEPS, step * dt);
                free(full);
            }

            io_time += (MPI_Wtime() - io_start);
        }
    } // end time loop

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    // ------------------------------------------------------------
    // 7) Performance statistics
    // ------------------------------------------------------------
    double max_compute, max_comm, max_io, max_total;
    MPI_Reduce(&compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&io_time, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        printf("\n=== Performance Statistics ===\n");
        printf("Total execution time:  %.4f seconds\n", max_total);
        printf("Computation time:      %.4f seconds (%.1f%%)\n", 
               max_compute, 100.0 * max_compute / max_total);
        printf("Communication time:    %.4f seconds (%.1f%%)\n", 
               max_comm, 100.0 * max_comm / max_total);
        printf("I/O time:              %.4f seconds (%.1f%%)\n", 
               max_io, 100.0 * max_io / max_total);
        printf("\nGrid updates per second: %.2e\n", 
               (double)(NX * NY * NSTEPS) / max_total);
        printf("Time per iteration:    %.6f seconds\n", max_total / NSTEPS);
    }

    // ------------------------------------------------------------
    // 8) Cleanup
    // ------------------------------------------------------------
    free(T);
    free(Tnew);
    free(send_west);
    free(send_east);
    free(recv_west);
    free(recv_east);
    free(sendbuf);
    if (rank == 0 && gatherbuf) free(gatherbuf);

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}