/**
 * @file goe_search_7x7.c
 * @brief Game of Life 7x7 Garden of Eden (GOE) Prover
 *
 * Distributed brute-force search for Garden of Eden states on a 7x7 grid.
 * Utilizes MPI for distributed computing and OpenMP for multi-threading.
 * Incorporates dynamic load balancing based on node performance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/**
 * @brief Precomputed lookup tables for valid predecessor transitions.
 * Since 7x7 is larger, we split row matching into left and right halves.
 */
uint64_t left_p3[16][64][64];
uint32_t right_p3[8][32][32];
uint32_t overlap_mask[4];

/**
 * @brief Reverses the bits of a 7-bit integer.
 * Used for identifying symmetric configurations.
 *
 * @param row The 7-bit integer representing a row.
 * @return The reversed 7-bit integer.
 */
int rev7(int row) {
    int rev = 0;
    for (int i = 0; i < 7; i++) {
        if ((row >> i) & 1) rev |= (1 << (6 - i));
    }
    return rev;
}

/**
 * @brief Initializes the transition lookup tables.
 * Computes all possible left and right half configurations for valid predecessors.
 */
void init_tables() {
    // Generate valid overlap masks to correlate left and right halves
    for (int i = 0; i < 4; i++) {
        overlap_mask[i] = 0;
        for (int p3_r = 0; p3_r < 32; p3_r++) {
            if ((p3_r & 3) == i) overlap_mask[i] |= (1U << p3_r);
        }
    }
    
    // Evaluate Game of Life rules for 9-cell neighborhoods
    for (int p1 = 0; p1 < 512; p1++) {
        for (int p2 = 0; p2 < 512; p2++) {
            for (int p3 = 0; p3 < 512; p3++) {
                int row_val = 0;
                for (int c = 1; c <= 7; c++) {
                    int c1 = (p1 >> (c - 1)) & 7, c2 = (p2 >> (c - 1)) & 7, c3 = (p3 >> (c - 1)) & 7;
                    int live = (c1 & 1) + ((c1 >> 1) & 1) + ((c1 >> 2) & 1) +
                               (c2 & 1) + ((c2 >> 2) & 1) +
                               (c3 & 1) + ((c3 >> 1) & 1) + ((c3 >> 2) & 1);
                    int center = (c2 >> 1) & 1;
                    if (live == 3 || (center && live == 2)) row_val |= (1 << (c - 1));
                }
                
                // Populate left transition table
                int T_left = row_val & 15, p1_l = p1 & 63, p2_l = p2 & 63, p3_l = p3 & 63;
                left_p3[T_left][p1_l][p2_l] |= (1ULL << p3_l);

                // Populate right transition table
                int T_right = (row_val >> 4) & 7, p1_r = (p1 >> 4) & 31, p2_r = (p2 >> 4) & 31, p3_r = (p3 >> 4) & 31;
                right_p3[T_right][p1_r][p2_r] |= (1U << p3_r);
            }
        }
    }
}

/**
 * @brief Checks if a pattern is the canonical (lexicographically smallest) form.
 * Tests horizontal flip, vertical flip, and 180-degree rotation.
 *
 * @param T The 49-bit integer representing the 7x7 grid.
 * @return true if canonical, false otherwise.
 */
bool is_canonical(uint64_t T) {
    uint64_t href = 0, vref = 0, rot180 = 0;
    for (int r = 0; r < 7; r++) {
        int row = (T >> (r * 7)) & 127;
        int rev = rev7(row);
        href |= ((uint64_t)rev) << (r * 7);
        vref |= ((uint64_t)row) << ((6 - r) * 7);
        rot180 |= ((uint64_t)rev) << ((6 - r) * 7);
    }
    return !(href < T || vref < T || rot180 < T);
}

/**
 * @brief Recursively attempts to find a predecessor for the given pattern.
 * Uses split-half lookup tables and memoization for efficiency.
 *
 * @param row The current row depth.
 * @param p1 Predecessor row n-2.
 * @param p2 Predecessor row n-1.
 * @param T Target 7x7 pattern.
 * @param current_id The memoization clearance ID.
 * @param failed Memoization table recording failed subpaths.
 * @return true if a predecessor exists, false otherwise.
 */
bool has_predecessor(int row, int p1, int p2, uint64_t T, uint16_t current_id, uint16_t (*failed)[512][512]) {
    if (row == 7) return true;
    if (failed[row][p1][p2] == current_id) return false;

    // Extract relevant row segments for table lookup
    int T_row = (T >> (row * 7)) & 127;
    int T_left = T_row & 15, T_right = (T_row >> 4) & 7;
    int p1_l = p1 & 63, p1_r = (p1 >> 4) & 31;
    int p2_l = p2 & 63, p2_r = (p2 >> 4) & 31;

    // Get bitsets of valid next predecessor rows (p3)
    uint64_t l_mask = left_p3[T_left][p1_l][p2_l];
    uint32_t r_mask = right_p3[T_right][p1_r][p2_r];

    // Iterate through overlapping valid left and right halves
    while (l_mask) {
        int p3_l = __builtin_ctzll(l_mask);
        l_mask &= l_mask - 1;
        uint32_t valid_r = r_mask & overlap_mask[p3_l >> 4];
        
        while (valid_r) {
            int p3_r = __builtin_ctz(valid_r);
            valid_r &= valid_r - 1;
            int p3 = p3_l | (p3_r << 4);
            if (has_predecessor(row + 1, p2, p3, T, current_id, failed)) return true;
        }
    }
    
    // Mark this path as failed in the memoization table
    failed[row][p1][p2] = current_id;
    return false;
}

/**
 * @brief Rapid bit-mixing hash function for stochastic load balancing.
 *
 * @param x Input value.
 * @return Hashed value.
 */
static inline uint64_t mix_hash(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

/**
 * @brief Main function. Distributes search over MPI nodes and handles load balancing.
 */
int main(int argc, char** argv) {
    int provided;
    // Require MPI thread support for OpenMP interoperability
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    omp_set_num_threads(16);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) printf("Initializing L1-Cache Brute Force GOE Search...\n");
    init_tables();
    
    // Total state space is 2^49 for a 7x7 grid
    uint64_t total_T = 1ULL << 49;
    uint64_t step_size = num_procs * 5000000000ULL;
    uint64_t T_current = 0, total_orphans = 0;

    // Rank 0 handles checkpoint loading
    if (rank == 0) {
        FILE* ckpt = fopen("checkpoint.dat", "r");
        if (ckpt) {
            fscanf(ckpt, "%llu", (unsigned long long*)&T_current);
            fscanf(ckpt, "%llu", (unsigned long long*)&total_orphans);
            fclose(ckpt);
            printf("Resuming from T %llu\n", (unsigned long long)T_current);
        } else {
            T_current = 0;
            printf("Starting fresh from T 0\n");
        }
    }
    
    // Broadcast state to all MPI ranks
    MPI_Bcast(&T_current, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_orphans, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Initial equal weights for dynamic load balancing
    double *node_weights = malloc(num_procs * sizeof(double));
    for (int i = 0; i < num_procs; i++) node_weights[i] = 100.0 / num_procs;

    double start_time = MPI_Wtime(), last_checkpoint_time = MPI_Wtime(), my_avg_compute_time = -1.0;

    // Main search loop iterating in steps to allow load balancing updates
    while (T_current < total_T) {
        uint64_t my_orphans = 0;
        uint64_t T_end = T_current + step_size;
        if (T_end > total_T) T_end = total_T;

        double total_weight = 0;
        for (int i = 0; i < num_procs; i++) total_weight += node_weights[i];

        // Assign task slices using a hash range distributed by node weight
        uint64_t HASH_RANGE = 10000000;
        double my_start_frac = 0;
        for (int i = 0; i < rank; i++) my_start_frac += node_weights[i] / total_weight;
        uint64_t my_start_hash = (uint64_t)(my_start_frac * HASH_RANGE);
        uint64_t my_end_hash = (uint64_t)((my_start_frac + (node_weights[rank] / total_weight)) * HASH_RANGE);
        if (rank == num_procs - 1) my_end_hash = HASH_RANGE;

        double compute_start = MPI_Wtime();
        printf("T_current=%llu, rank=%d, weight=%f, my_start_hash=%llu, my_end_hash=%llu\n", 
               (unsigned long long)T_current, rank, node_weights[rank], 
               (unsigned long long)my_start_hash, (unsigned long long)my_end_hash);
        fflush(stdout);
        
        if (node_weights[rank] > 0.0) {
            #pragma omp parallel
            {
                // Thread-local memoization state
                uint16_t (*failed)[512][512] = calloc(7, sizeof(*failed));
                uint16_t current_id = 1;
                unsigned long long local_orphans = 0;

                #pragma omp for schedule(dynamic, 10000)
                for (uint64_t p = T_current; p < T_end; p++) {
                    if (p % 100000 == 0) {
                        printf("Reached %llu\n", (unsigned long long)p);
                        fflush(stdout);
                    }
                    
                    // Filter tasks based on hash assignment
                    if (mix_hash(p) % HASH_RANGE < my_start_hash || mix_hash(p) % HASH_RANGE >= my_end_hash) continue;
                    
                    // Skip non-canonical patterns
                    if (!is_canonical(p)) continue;
                    
                    // Fast memoization clear via ID increment
                    current_id++;
                    if (current_id == 65535) {
                        memset(failed, 0, 7 * 512 * 512 * sizeof(uint16_t));
                        current_id = 1;
                    }

                    bool has_pred = false;
                    for (int p1 = 0; p1 < 512 && !has_pred; p1++) {
                        for (int p2 = 0; p2 < 512 && !has_pred; p2++) {
                            if (has_predecessor(0, p1, p2, p, current_id, failed)) has_pred = true;
                        }
                    }
                    if (!has_pred) local_orphans++;
                }

                #pragma omp atomic
                my_orphans += local_orphans;
                free(failed);
            }
        }

        double compute_end = MPI_Wtime(), my_compute_time = compute_end - compute_start;
        if (my_compute_time < 0.001) my_compute_time = 0.001;
        
        // Calculate throughput and update exponential moving average for load balancing
        double my_throughput = node_weights[rank] / my_compute_time;
        if (my_avg_compute_time < 0.0) my_avg_compute_time = my_throughput; 
        else my_avg_compute_time = 0.8 * my_avg_compute_time + 0.2 * my_throughput;

        double my_new_weight = my_avg_compute_time;
        
        // Share new performance metrics across all nodes
        MPI_Allgather(&my_new_weight, 1, MPI_DOUBLE, node_weights, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Normalize node weights
        double total_new_weight = 0;
        for(int i = 0; i < num_procs; i++) {
            if (node_weights[i] < 0.0) node_weights[i] = 0.0;
            total_new_weight += node_weights[i];
        }
        for(int i = 0; i < num_procs; i++) node_weights[i] = (node_weights[i] / total_new_weight) * 100.0;

        uint64_t global_orphans = 0;
        
        // Sum up found orphan states across all nodes
        MPI_Reduce(&my_orphans, &global_orphans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            total_orphans += global_orphans;
            T_current = T_end;
            
            // Print progress and estimated time remaining
            double elapsed_time = MPI_Wtime() - start_time;
            double speed = T_current / elapsed_time;
            double remaining = (total_T - T_current) / speed;
            int hours = (int)(remaining / 3600), minutes = (int)(remaining / 60) % 60;
            
            printf("goe_search_7x7 %30.4f%% %lluKB %7.1f M-States/sec %02d:%02d ETA\n",
                   ((double)T_current / total_T) * 100.0, total_orphans * sizeof(uint64_t) / 1024,
                   speed / 1000000.0, hours, minutes);
            fflush(stdout);

            // Periodic checkpointing
            if (MPI_Wtime() - last_checkpoint_time > 300.0) {
                FILE *chk = fopen("checkpoint.dat", "w");
                if (chk) {
                    fprintf(chk, "%llu\n", (unsigned long long)T_current);
                    fprintf(chk, "%llu\n", (unsigned long long)total_orphans);
                    fclose(chk);
                }
                last_checkpoint_time = MPI_Wtime();
            }
        }
        
        // Broadcast the updated target block for the next iteration
        MPI_Bcast(&T_current, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}
