/**
 * @file goe_search.c
 * @brief Game of Life 6x6 Garden of Eden (GOE) Prover
 * 
 * This program searches for Garden of Eden states (patterns with no predecessors)
 * in Conway's Game of Life on a 6x6 grid. It uses a distributed approach with MPI
 * and OpenMP, employing Top-Down Memoized DFS, ID-based Zero-Clear, and Full Symmetry checks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <string.h>

/**
 * @brief Lookup table for valid predecessor configurations (p3).
 * Indexed by [target pattern][predecessor 1][predecessor 2][4].
 * The 4 uint64_t values form a 256-bit bitset of valid p3 states.
 */
// [T_k][p1][p2][4]
uint64_t valid_p3[64][256][256][4] __attribute__((aligned(32)));

/**
 * @brief Determines the next state of a cell given its neighborhood.
 * 
 * @param p0 Previous row pattern.
 * @param p1 Current row pattern (contains the cell to check).
 * @param p2 Next row pattern.
 * @param c The column index of the cell.
 * @return 1 if the cell will be alive in the next generation, 0 otherwise.
 */
int cell_next(int p0, int p1, int p2, int c) {
    int sum = 0;
    for (int i = c; i <= c + 2; i++) {
        sum += (p0 >> i) & 1;
        if (i != c + 1) sum += (p1 >> i) & 1;
        sum += (p2 >> i) & 1;
    }
    int alive = (p1 >> (c + 1)) & 1;
    if (alive && (sum == 2 || sum == 3)) return 1;
    if (!alive && sum == 3) return 1;
    return 0;
}

/**
 * @brief Initializes the predecessor lookup tables.
 * Precomputes valid transitions to optimize the search space.
 */
void init_tables() {
    memset(valid_p3, 0, sizeof(valid_p3));
    for (int p0 = 0; p0 < 256; p0++) {
        for (int p1 = 0; p1 < 256; p1++) {
            for (int p2 = 0; p2 < 256; p2++) {
                int T = 0;
                for (int c = 0; c < 6; c++) {
                    T |= (cell_next(p0, p1, p2, c) << c);
                }
                valid_p3[T][p0][p1][p2 / 64] |= (1ULL << (p2 % 64));
            }
        }
    }
}

/**
 * @brief Reverses the bits of a 6-bit integer.
 * Used for symmetry checking.
 * 
 * @param x The 6-bit integer to reverse.
 * @return The reversed 6-bit integer.
 */
int rev6(int x) {
    int r = 0;
    for (int i = 0; i < 6; i++) {
        if ((x >> i) & 1) r |= (1 << (5 - i));
    }
    return r;
}

/**
 * @brief Full Symmetry Check (Horizontal, Vertical, 180-Rot)
 * Checks if the given 6x6 pattern is the canonical (lexicographically smallest)
 * representation among its symmetric equivalents.
 * 
 * @param T The 36-bit pattern representing the 6x6 grid.
 * @return true if T is canonical, false otherwise.
 */
bool is_canonical(uint64_t T) {
    uint64_t href = 0, vref = 0, rot180 = 0;
    for (int r = 0; r < 6; r++) {
        int row = (T >> (r * 6)) & 63;
        int rev = rev6(row);
        href |= ((uint64_t)rev) << (r * 6);
        vref |= ((uint64_t)row) << ((5 - r) * 6);
        rot180 |= ((uint64_t)rev) << ((5 - r) * 6);
    }
    if (href < T || vref < T || rot180 < T) return false;
    return true;
}

/**
 * @brief Depth-First Search to determine if a pattern has a predecessor.
 * Uses memoization to avoid redundant checks.
 * 
 * @param row The current row being processed.
 * @param p1 Predecessor row 1.
 * @param p2 Predecessor row 2.
 * @param T The target pattern.
 * @param current_id The current ID used for O(1) clearing of the memoization table.
 * @param failed The memoization table tracking failed paths.
 * @return true if a predecessor is found, false otherwise.
 */
bool has_predecessor(int row, int p1, int p2, uint64_t T, int current_id, int failed[6][256][256]) {
    // If all rows have been successfully processed, a predecessor exists
    if (row == 6) return true;
    
    // Check memoization table
    if (failed[row][p1][p2] == current_id) return false;

    // Extract the target row state
    int T_row = (T >> (row * 6)) & 63;
    
    // Check valid predecessor states using bitwise operations and __builtin_ctzll
    uint64_t m0 = valid_p3[T_row][p1][p2][0];
    while (m0) {
        int p3 = __builtin_ctzll(m0);
        if (has_predecessor(row + 1, p2, p3, T, current_id, failed)) return true;
        m0 &= m0 - 1;
    }
    uint64_t m1 = valid_p3[T_row][p1][p2][1];
    while (m1) {
        int p3 = 64 + __builtin_ctzll(m1);
        if (has_predecessor(row + 1, p2, p3, T, current_id, failed)) return true;
        m1 &= m1 - 1;
    }
    uint64_t m2 = valid_p3[T_row][p1][p2][2];
    while (m2) {
        int p3 = 128 + __builtin_ctzll(m2);
        if (has_predecessor(row + 1, p2, p3, T, current_id, failed)) return true;
        m2 &= m2 - 1;
    }
    uint64_t m3 = valid_p3[T_row][p1][p2][3];
    while (m3) {
        int p3 = 192 + __builtin_ctzll(m3);
        if (has_predecessor(row + 1, p2, p3, T, current_id, failed)) return true;
        m3 &= m3 - 1;
    }

    // Mark as failed in the memoization table
    failed[row][p1][p2] = current_id;
    return false;
}

/**
 * @brief Main entry point. Sets up MPI, initializes tables, and partitions work.
 */
int main(int argc, char** argv) {
    int provided;
    // Initialize MPI with thread support required for OpenMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        printf("Initializing Game of Life 6x6 GOE Prover on %d MPI nodes...\n", num_procs);
        printf("Optimizations: Top-Down Memoized DFS, ID-based Zero-Clear, Full Symmetry, OpenMP\n");
    }
    
    init_tables();
    
    // Total state space for 6x6 grid is 2^36
    uint64_t total_T = (1ULL << 36);
    uint64_t num_tasks_per_node = total_T / num_procs;
    
    if (rank == 0) {
        printf("Starting distributed search of 68.7 Billion states (Cyclic Distribution)...\n");
    }

    double start_time = MPI_Wtime();
    uint64_t local_orphans = 0;

    int progress_counter = 0;
    uint64_t progress_step = num_tasks_per_node / 1000;
    if (progress_step == 0) progress_step = 1;

    // Parallelize task processing using OpenMP with dynamic scheduling
    #pragma omp parallel reduction(+:local_orphans)
    {
        // Thread-local memoization table for DFS
        int failed[6][256][256] = {0};
        int current_id = 0;

        #pragma omp for schedule(dynamic, 1000000)
        for (uint64_t i = 0; i < num_tasks_per_node; i++) {
            // Distribute tasks cyclically to balance load
            uint64_t T = i * num_procs + rank;
            
            if (rank == 0 && omp_get_thread_num() == 0 && i % progress_step == 0) {
                printf("Progress: %.1f%%\n", 100.0 * i / num_tasks_per_node);
                fflush(stdout);
            }

            // Only process canonical patterns to avoid duplicate effort
            if (!is_canonical(T)) continue;
            
            // Increment ID to "clear" the memoization array in O(1) time
            current_id++;
            if (current_id > 1000000000) {
                memset(failed, 0, sizeof(failed));
                current_id = 1;
            }

            bool is_orphan = true;
            // Iterate over all possible first two rows of predecessors
            for (int p1 = 0; p1 < 256 && is_orphan; p1++) {
                for (int p2 = 0; p2 < 256 && is_orphan; p2++) {
                    if (has_predecessor(0, p1, p2, T, current_id, failed)) {
                        is_orphan = false;
                    }
                }
            }

            if (is_orphan) {
                local_orphans++;
            }
        }
    }

    // Aggregate orphan state counts from all MPI nodes
    uint64_t total_orphans = 0;
    MPI_Reduce(&local_orphans, &total_orphans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Progress: 100.0%%\n");
        printf("==================================================\n");
        printf("Search Completed in %.2f seconds.\n", end_time - start_time);
        if (total_orphans == 0) {
            printf("PROOF SUCCESSFUL: There are NO orphan states in a 6x6 grid.\n");
        } else {
            printf("FOUND %llu orphan states (modulo symmetry) in a 6x6 grid.\n", total_orphans);
        }
        printf("==================================================\n");
    }

    MPI_Finalize();
    return 0;
}
