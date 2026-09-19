#include <math.h>
/**
 * @file goe_search_7x7.hip
 * @brief Game of Life orphan search implementation for 7x7 bounding boxes.
 *
 * This file implements an optimized search algorithm to find Garden of Eden (orphan) states
 * in Conway's Game of Life within a 7x7 grid. It leverages HIP for execution on AMD GPUs,
 * MPI for distributed computing, and heavily utilizes bitwise logic and shared memory caching
 * to maximize throughput.
 */

#include <stdio.h>
#include <thread>
#include <atomic>
#include <vector>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#ifndef __HIP_DEVICE_COMPILE__
#endif
#include <hip/hip_runtime.h>

// -----------------------------------------------------------------------------
// Global Lookup Tables (Host)
// -----------------------------------------------------------------------------
uint16_t small_left_p3[4][16][16];
uint32_t right_p3[8][32][32];
uint32_t expand_8bit[256];
uint32_t overlap_mask[4];

// -----------------------------------------------------------------------------
// Constant Memory Lookup Tables (Device)
// Constant memory provides high-bandwidth broadcast for read-only tables.
// -----------------------------------------------------------------------------
__constant__ uint16_t c_small_left_p3[4][16][16];
__constant__ uint32_t c_right_p3[8][32][32];
__constant__ uint32_t c_expand_8bit[256];
__constant__ uint32_t c_overlap_mask[4];

extern "C" void omp_set_num_threads(int);
uint32_t* h_valid_p1p2 = NULL;
int* h_valid_count = NULL;

/**
 * @brief Reverses a 7-bit row for symmetry checking.
 * @param row The 7-bit integer representing a row.
 * @return The horizontally reversed 7-bit integer.
 */
__host__ __device__ int rev7(int row) {
    int rev = 0;
    for (int i = 0; i < 7; i++) {
        if ((row >> i) & 1) rev |= (1 << (6 - i));
    }
    return rev;
}

/**
 * @brief Initializes all lookup tables on the host before copying to device.
 *
 * Computes all possible transitions for small windows to quickly determine
 * valid predecessors in the Game of Life rules. Populates small_left_p3,
 * right_p3, and overlap arrays for fast bitwise checking.
 */
void init_tables() {
    memset(small_left_p3, 0, sizeof(small_left_p3));
    memset(right_p3, 0, sizeof(right_p3));
    
    // Initialize overlap masks for merging left and right halves
    for (int i = 0; i < 4; i++) {
        overlap_mask[i] = 0;
        for (int p3_r = 0; p3_r < 32; p3_r++) {
            if ((p3_r & 3) == i) overlap_mask[i] |= (1U << p3_r);
        }
    }
    
    // Populate left-side transition lookup (3-cell neighborhood)
    for (int p1 = 0; p1 < 16; p1++) {
        for (int p2 = 0; p2 < 16; p2++) {
            for (int p3 = 0; p3 < 16; p3++) {
                int T = 0;
                for (int i = 0; i < 2; i++) {
                    int s = ((p1 >> i) & 1) + ((p1 >> (i+1)) & 1) + ((p1 >> (i+2)) & 1) +
                            ((p2 >> i) & 1) +                       ((p2 >> (i+2)) & 1) +
                            ((p3 >> i) & 1) + ((p3 >> (i+1)) & 1) + ((p3 >> (i+2)) & 1);
                    int center = (p2 >> (i+1)) & 1;
                    // Game of life rule: 3 neighbors = birth/survival, 2 neighbors = survival if alive
                    if (s == 3 || (s == 2 && center == 1)) T |= (1 << i);
                }
                small_left_p3[T][p1][p2] |= (1 << p3);
            }
        }
    }
    
    // Populate right-side transition lookup
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
                int T_right = (row_val >> 4) & 7, p1_r = (p1 >> 4) & 31, p2_r = (p2 >> 4) & 31, p3_r = (p3 >> 4) & 31;
                right_p3[T_right][p1_r][p2_r] |= (1U << p3_r);
            }
        }
    }
    
    // Fast bit expansion table for aligning bitmasks
    for (int i = 0; i < 256; i++) {
        uint32_t res = 0;
        for (int bit = 0; bit < 8; bit++) {
            if ((i >> bit) & 1) res |= (0xFU << (bit * 4));
        }
        expand_8bit[i] = res;
    }
    
    // Precompute valid starting pairs (p1, p2) for row transitions
    h_valid_p1p2 = (uint32_t*)malloc(128 * 262144 * sizeof(uint32_t));
    h_valid_count = (int*)malloc(128 * sizeof(int));
    for (int T_row = 0; T_row < 128; T_row++) {
        int count = 0;
        for (int p2 = 0; p2 < 512; p2++) {
            for (int p1 = 0; p1 < 512; p1++) {
                int T_01 = T_row & 3, T_23 = (T_row >> 2) & 3, T_right = (T_row >> 4) & 7;
                int p1_03 = p1 & 15, p1_25 = (p1 >> 2) & 15, p1_r = (p1 >> 4) & 31;
                int p2_03 = p2 & 15, p2_25 = (p2 >> 2) & 15, p2_r = (p2 >> 4) & 31;
                uint64_t mask1 = small_left_p3[T_01][p1_03][p2_03];
                uint64_t mask2 = small_left_p3[T_23][p1_25][p2_25];
                uint64_t expanded1 = mask1 | (mask1 << 16) | (mask1 << 32) | (mask1 << 48);
                uint64_t expanded2 = expand_8bit[mask2 & 255] | ((uint64_t)expand_8bit[mask2 >> 8] << 32);
                uint64_t l_mask = expanded1 & expanded2;
                uint32_t r_mask = right_p3[T_right][p1_r][p2_r];
                bool valid = false;
                while (l_mask) {
                    int p3_l = __builtin_ctzll(l_mask);
                    l_mask &= (l_mask - 1);
                    if (r_mask & overlap_mask[p3_l >> 4]) {
                        valid = true;
                        break;
                    }
                }
                if (valid) {
                    h_valid_p1p2[T_row * 262144 + count] = (p1 << 9) | p2;
                    count++;
                }
            }
        }
        h_valid_count[T_row] = count;
    }
}

/**
 * @brief Checks if a pattern T is canonical (minimal among its symmetries).
 * @param T The 49-bit integer representing the 7x7 grid.
 * @return True if canonical, False if a smaller symmetric variant exists.
 */
__host__ __device__ bool is_canonical(uint64_t T) {
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

// Macros for simulating a stack in the DFS to avoid dynamic memory or recursion
#define GET_P1(sp) ((sp)==0?p1_0:(sp)==1?p1_1:(sp)==2?p1_2:(sp)==3?p1_3:(sp)==4?p1_4:(sp)==5?p1_5:p1_6)
#define GET_P2(sp) ((sp)==0?p2_0:(sp)==1?p2_1:(sp)==2?p2_2:(sp)==3?p2_3:(sp)==4?p2_4:(sp)==5?p2_5:p2_6)
#define GET_L_MASK(sp) ((sp)==0?l_mask_0:(sp)==1?l_mask_1:(sp)==2?l_mask_2:(sp)==3?l_mask_3:(sp)==4?l_mask_4:(sp)==5?l_mask_5:l_mask_6)
#define GET_R_MASK(sp) ((sp)==0?r_mask_0:(sp)==1?r_mask_1:(sp)==2?r_mask_2:(sp)==3?r_mask_3:(sp)==4?r_mask_4:(sp)==5?r_mask_5:r_mask_6)
#define GET_VALID_R(sp) ((sp)==0?valid_r_0:(sp)==1?valid_r_1:(sp)==2?valid_r_2:(sp)==3?valid_r_3:(sp)==4?valid_r_4:(sp)==5?valid_r_5:valid_r_6)
#define GET_P3_L(sp) ((sp)==0?p3_l_0:(sp)==1?p3_l_1:(sp)==2?p3_l_2:(sp)==3?p3_l_3:(sp)==4?p3_l_4:(sp)==5?p3_l_5:p3_l_6)

#define SET_P1(sp, val) do { if ((sp)==0) p1_0=(val); else if ((sp)==1) p1_1=(val); else if ((sp)==2) p1_2=(val); else if ((sp)==3) p1_3=(val); else if ((sp)==4) p1_4=(val); else if ((sp)==5) p1_5=(val); else p1_6=(val); } while(0)
#define SET_P2(sp, val) do { if ((sp)==0) p2_0=(val); else if ((sp)==1) p2_1=(val); else if ((sp)==2) p2_2=(val); else if ((sp)==3) p2_3=(val); else if ((sp)==4) p2_4=(val); else if ((sp)==5) p2_5=(val); else p2_6=(val); } while(0)
#define SET_L_MASK(sp, val) do { if ((sp)==0) l_mask_0=(val); else if ((sp)==1) l_mask_1=(val); else if ((sp)==2) l_mask_2=(val); else if ((sp)==3) l_mask_3=(val); else if ((sp)==4) l_mask_4=(val); else if ((sp)==5) l_mask_5=(val); else l_mask_6=(val); } while(0)
#define SET_R_MASK(sp, val) do { if ((sp)==0) r_mask_0=(val); else if ((sp)==1) r_mask_1=(val); else if ((sp)==2) r_mask_2=(val); else if ((sp)==3) r_mask_3=(val); else if ((sp)==4) r_mask_4=(val); else if ((sp)==5) r_mask_5=(val); else r_mask_6=(val); } while(0)
#define SET_VALID_R(sp, val) do { if ((sp)==0) valid_r_0=(val); else if ((sp)==1) valid_r_1=(val); else if ((sp)==2) valid_r_2=(val); else if ((sp)==3) valid_r_3=(val); else if ((sp)==4) valid_r_4=(val); else if ((sp)==5) valid_r_5=(val); else valid_r_6=(val); } while(0)
#define SET_P3_L(sp, val) do { if ((sp)==0) p3_l_0=(val); else if ((sp)==1) p3_l_1=(val); else if ((sp)==2) p3_l_2=(val); else if ((sp)==3) p3_l_3=(val); else if ((sp)==4) p3_l_4=(val); else if ((sp)==5) p3_l_5=(val); else p3_l_6=(val); } while(0)

/**
 * @brief Depth-First Search to find a predecessor for pattern T.
 *
 * Implements a non-recursive backtracking search to see if a given pattern T
 * has any valid predecessor. Uses a thread-local cache array to prevent redundant
 * sub-tree searches. If a search takes too long, it bails out, marking it a 'hard' state.
 *
 * @param start_p1 Initial row -1
 * @param start_p2 Initial row 0
 * @param T The target 7x7 pattern
 * @param cache Shared memory or local cache to store failed states
 * @param global_nodes Pointer to track nodes explored (for bailout)
 * @param max_nodes Bailout threshold
 * @return 1 if predecessor found, 0 if orphan, 2 if bailout (hard state)
 */
__device__ __host__ int has_predecessor_dfs(int start_p1, int start_p2, uint64_t T, uint32_t* cache, int* global_nodes, int max_nodes) {
    uint64_t l_mask_0 = 0, l_mask_1 = 0, l_mask_2 = 0, l_mask_3 = 0, l_mask_4 = 0, l_mask_5 = 0, l_mask_6 = 0;
    uint32_t r_mask_0 = 0, r_mask_1 = 0, r_mask_2 = 0, r_mask_3 = 0, r_mask_4 = 0, r_mask_5 = 0, r_mask_6 = 0;
    uint32_t valid_r_0 = 0, valid_r_1 = 0, valid_r_2 = 0, valid_r_3 = 0, valid_r_4 = 0, valid_r_5 = 0, valid_r_6 = 0;
    int p1_0 = start_p1, p1_1 = 0, p1_2 = 0, p1_3 = 0, p1_4 = 0, p1_5 = 0, p1_6 = 0;
    int p2_0 = start_p2, p2_1 = 0, p2_2 = 0, p2_3 = 0, p2_4 = 0, p2_5 = 0, p2_6 = 0;
    int p3_l_0 = 0, p3_l_1 = 0, p3_l_2 = 0, p3_l_3 = 0, p3_l_4 = 0, p3_l_5 = 0, p3_l_6 = 0;
    int sp = 0;

    int T_row = T & 127;
            
    int T_01 = T_row & 3, T_23 = (T_row >> 2) & 3, T_right = (T_row >> 4) & 7;
    int p1_03 = start_p1 & 15, p1_25 = (start_p1 >> 2) & 15, p1_r = (start_p1 >> 4) & 31;
    int p2_03 = start_p2 & 15, p2_25 = (start_p2 >> 2) & 15, p2_r = (start_p2 >> 4) & 31;
    
    uint64_t mask1 = c_small_left_p3[T_01][p1_03][p2_03];
    uint64_t mask2 = c_small_left_p3[T_23][p1_25][p2_25];
    uint64_t expanded1 = mask1 | (mask1 << 16) | (mask1 << 32) | (mask1 << 48);
    uint64_t expanded2 = c_expand_8bit[mask2 & 255] | ((uint64_t)c_expand_8bit[mask2 >> 8] << 32);
    l_mask_0 = expanded1 & expanded2;
    r_mask_0 = c_right_p3[T_right][p1_r][p2_r];
    valid_r_0 = 0;

    while (sp >= 0) {
        if (sp > 0) {
            (*global_nodes)++;
            if (*global_nodes > max_nodes) return 2; // Hard state
        }
        
        uint32_t c_valid_r = GET_VALID_R(sp);
        if (c_valid_r) {
            int p3_r = __builtin_ctz(c_valid_r);
            SET_VALID_R(sp, c_valid_r & (c_valid_r - 1));
            int p3 = GET_P3_L(sp) | (p3_r << 4);
            
            if (sp == 6) return 1;
            
            int next_sp = sp + 1;
            int c_p2 = GET_P2(sp);
            SET_P1(next_sp, c_p2);
            SET_P2(next_sp, p3);
            
            uint32_t state = (next_sp << 18) | (c_p2 << 9) | p3;
            uint32_t hash = (state * 2654435761U);
            if (cache[hash % 32] == state) continue;
            
            int T_r = (T >> (next_sp * 7)) & 127;
            int T_01_n = T_r & 3, T_23_n = (T_r >> 2) & 3, T_ri = (T_r >> 4) & 7;
            int np1_03 = c_p2 & 15, np1_25 = (c_p2 >> 2) & 15, np1_r = (c_p2 >> 4) & 31;
            int np2_03 = p3 & 15, np2_25 = (p3 >> 2) & 15, np2_r = (p3 >> 4) & 31;
            
            uint64_t nm1 = c_small_left_p3[T_01_n][np1_03][np2_03];
            uint64_t nm2 = c_small_left_p3[T_23_n][np1_25][np2_25];
            uint64_t nexp1 = nm1 | (nm1 << 16) | (nm1 << 32) | (nm1 << 48);
            uint64_t nexp2 = c_expand_8bit[nm2 & 255] | ((uint64_t)c_expand_8bit[nm2 >> 8] << 32);
            
            SET_L_MASK(next_sp, nexp1 & nexp2);
            SET_R_MASK(next_sp, c_right_p3[T_ri][np1_r][np2_r]);
            SET_VALID_R(next_sp, 0);
            
            sp = next_sp;
            continue;
        }
        
        uint64_t c_l_mask = GET_L_MASK(sp);
        if (c_l_mask) {
            int p3_l = __builtin_ctzll(c_l_mask);
            SET_L_MASK(sp, c_l_mask & (c_l_mask - 1));
            SET_P3_L(sp, p3_l);
            SET_VALID_R(sp, GET_R_MASK(sp) & c_overlap_mask[p3_l >> 4]);
            continue;
        }
        
        // Mark state as failed in the cache
        uint32_t fail_state = (sp << 18) | (GET_P1(sp) << 9) | GET_P2(sp);
        uint32_t fail_hash = (fail_state * 2654435761U);
        cache[fail_hash % 32] = fail_state;
        sp--;
    }
    
    return 0; // Orphan found
}

/**
 * @brief Breadth-First Search kernel for handling "hard" states.
 *
 * Runs a level-synchronous BFS utilizing GPU shared memory as a bit-vector.
 * Resolves states that hit the DFS bailout threshold.
 *
 * @param d_hard_states Array of hard states to process
 * @param h_hard_count Number of hard states
 * @param d_orphans Pointer to accumulate orphan counts
 */


/**
 * @brief Hashes a 64-bit integer.
 * @param x The value to hash.
 * @return Hashed value used for workload distribution.
 */
__host__ __device__ uint64_t mix_hash(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

/**
 * @brief Primary DFS search kernel executed on the GPU.
 *
 * Threads process a chunk of the state space, utilizing shared memory to
 * cache intermediate failed states during the DFS. Found orphans are atomically
 * incremented, and computationally heavy "hard" states are deferred to a BFS pass.
 */
__global__ void search_kernel_dfs(uint64_t T_start, uint64_t T_end, uint64_t my_start_hash, uint64_t my_end_hash, uint64_t HASH_RANGE, unsigned long long* d_orphans, uint64_t* d_hard_states, int* d_hard_count, uint32_t* d_valid_p1p2, int* d_valid_count) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = gridDim.x * blockDim.x;

    // Allocate shared memory cache for fast thread-local backtracking lookups
    __shared__ uint32_t shared_cache[256 * 32];
    uint32_t* cache = &shared_cache[threadIdx.x * 32];
    unsigned long long local_orphans = 0;

    for (uint64_t p = T_start + tid; p < T_end; p += num_threads) {
        // Hash distribution for MPI nodes
                if (!is_canonical(p)) continue; // Skip redundant symmetries

        for (int i = 0; i < 32; i++) cache[i] = 0xFFFFFFFF;

        int status = 0; // 0 = orphan, 1 = pred found, 2 = hard state
        int global_nodes = 0;
        int T_row = p & 127;
        
        int count = d_valid_count[T_row];
        for (int i = 0; i < count && status == 0; i++) {
            uint32_t pair = d_valid_p1p2[T_row * 262144 + i];
            int p1 = pair >> 9;
            int p2 = pair & 511;
            int res = has_predecessor_dfs(p1, p2, p, cache, &global_nodes, 50000);
            if (res == 1) status = 1;
            else if (res == 2) status = 2;
        }
        
        if (status == 0) {
            local_orphans++;
        } else if (status == 2) {
            int idx = atomicAdd(d_hard_count, 1);
            if (idx < 5000000) {
                d_hard_states[idx] = p; // Queue for BFS
            }
        }
    }

    if (local_orphans > 0) { printf("\n[GPU] FOUND %llu ORPHANS IN BLOCK\n", local_orphans);
        atomicAdd(d_orphans, local_orphans);
    }
}

/**
 * @brief Main execution function orchestrating MPI distribution and GPU execution.
 */

#ifndef __HIP_DEVICE_COMPILE__
#ifndef __HIP_DEVICE_COMPILE__
#endif
#endif
__host__ bool has_predecessor_perfect(int row, int p1, int p2, uint64_t T, uint8_t* failed) {
    if (row == 5) return true;
    
    int T_r = (T >> ((4 - row) * 9)) & 511;
    int T_01 = T_r & 3, T_23 = (T_r >> 2) & 3, T_ri = (T_r >> 4) & 7;
    int p1_03 = p1 & 15, p1_25 = (p1 >> 2) & 15, p1_r = (p1 >> 4) & 31;
    int p2_03 = p2 & 15, p2_25 = (p2 >> 2) & 15, p2_r = (p2 >> 4) & 31;
    
    uint64_t mask1 = small_left_p3[T_01][p1_03][p2_03];
    uint64_t mask2 = small_left_p3[T_23][p1_25][p2_25];
    uint64_t expanded1 = mask1 | (mask1 << 16) | (mask1 << 32) | (mask1 << 48);
    uint64_t expanded2 = expand_8bit[mask2 & 255] | ((uint64_t)expand_8bit[mask2 >> 8] << 32);
    uint64_t l_mask = expanded1 & expanded2;
    uint32_t r_mask = right_p3[T_ri][p1_r][p2_r];
    
    while (l_mask) {
        int p3_l = __builtin_ctzll(l_mask);
        l_mask &= (l_mask - 1);
        
        uint32_t valid_r = r_mask & overlap_mask[p3_l >> 4];
        while (valid_r) {
            int p3_r = __builtin_ctz(valid_r);
            valid_r &= (valid_r - 1);
            int p3 = p3_l | (p3_r << 4);
            
            if (failed[(row + 1) * 262144 + p2 * 512 + p3]) continue;
            
            if (has_predecessor_perfect(row + 1, p2, p3, T, failed)) return true;
        }
    }
    
    failed[row * 262144 + p1 * 512 + p2] = 1;
    return false;
}

int main(int argc, char** argv) {
    int provided;
        
    int rank = 0, world_size = 1;
    
    
    bool use_gpu = false;
    int hip_device_count = 0;
    if (hipGetDeviceCount(&hip_device_count) == hipSuccess && hip_device_count > 0) {
        use_gpu = true;
        hipSetDevice(0);
    }

    if (rank == 0 || use_gpu) {
        printf("Rank %d on %s: GPU Count: %d, HIP Error: %d\n", rank, "soldier12", hip_device_count, 0);
    }

    // Pre-compute host tables
    init_tables();
    
    unsigned long long* d_orphans = NULL;
    uint64_t* d_hard_states = NULL;
    int* d_hard_count = NULL;
    uint32_t* d_valid_p1p2 = NULL;
    int* d_valid_count = NULL;

omp_set_num_threads(16);
    if (use_gpu) {
        // Copy static tables to GPU constant memory
        hipMemcpyToSymbol(HIP_SYMBOL(c_small_left_p3), small_left_p3, sizeof(small_left_p3));
        hipMemcpyToSymbol(HIP_SYMBOL(c_right_p3), right_p3, sizeof(right_p3));
        hipMemcpyToSymbol(HIP_SYMBOL(c_expand_8bit), expand_8bit, sizeof(expand_8bit));
        hipMemcpyToSymbol(HIP_SYMBOL(c_overlap_mask), overlap_mask, sizeof(overlap_mask));
        
        hipMalloc(&d_orphans, sizeof(unsigned long long));
        hipMalloc(&d_hard_states, 5000000 * sizeof(uint64_t));
        hipMalloc(&d_hard_count, sizeof(int));
        hipMalloc(&d_valid_p1p2, 128 * 262144 * sizeof(uint32_t));
        hipMalloc(&d_valid_count, 128 * sizeof(int));
        
        hipMemcpy(d_valid_p1p2, h_valid_p1p2, 128 * 262144 * sizeof(uint32_t), hipMemcpyHostToDevice);
        hipMemcpy(d_valid_count, h_valid_count, 128 * sizeof(int), hipMemcpyHostToDevice);
    }
    
    uint64_t T_MAX = 1ULL << 49; // 7x7 state space
    uint64_t step_size = 10000000000ULL; // Process in chunks of 45B states
    uint64_t T_current = 2800000000000ULL;
    step_size = 100000000000ULL;
    unsigned long long total_orphans = 0, total_canonical = 0, total_dfs = 0;
    double my_avg_compute_time = -1.0;
    
    // Checkpointing and Load Balancing
    double node_weights[16];
    for (int i = 0; i < world_size; i++) node_weights[i] = 100.0 / world_size;
    
        
    if (rank == 0) {
        FILE* ckpt = NULL;
        if (ckpt) {
            fscanf(ckpt, "%llu", (unsigned long long*)&T_current);
            fscanf(ckpt, "%llu", &total_orphans);
            fscanf(ckpt, "%llu", &total_canonical);
            fscanf(ckpt, "%llu", &total_dfs);
            fclose(ckpt);
            printf("Resuming from T %llu\n", (unsigned long long)T_current);
        } else {
            printf("Starting fresh from T 0\n");
        }
    }
    
                    double total_weight_init = 0.0;
    for (int i = 0; i < world_size; i++) total_weight_init += node_weights[i];
    for (int i = 0; i < world_size; i++) node_weights[i] = (node_weights[i] / total_weight_init) * 100.0;

    uint64_t HASH_RANGE = 16777216ULL;
    
    // Main search loop
    while (T_current < T_MAX) {
        uint64_t T_end = T_current + step_size;
        if (T_end > T_MAX) T_end = T_MAX;
        
        double total_weight = 0.0;
        for (int i = 0; i < world_size; i++) total_weight += node_weights[i];
        
        double my_start_frac = 0.0;
        for (int i = 0; i < rank; i++) my_start_frac += node_weights[i] / total_weight;
        
        uint64_t my_start_hash = 0; //(uint64_t)(my_start_frac * HASH_RANGE);
        uint64_t my_end_hash = HASH_RANGE; //(uint64_t)((my_start_frac + (node_weights[rank] / total_weight)) * HASH_RANGE);
        if (rank == world_size - 1) my_end_hash = HASH_RANGE; //HASH_RANGE;
        
        if (1) {
            if (rank == 0 || use_gpu) {
                printf("T_current=%llu, rank=%d, weight=%f, my_start_hash=%llu, my_end_hash=%llu\n", 
                       (unsigned long long)T_current, rank, node_weights[rank], 
                       (unsigned long long)my_start_hash, (unsigned long long)my_end_hash);
                fflush(stdout);
            }
        }

        double compute_start = 1.0;

        unsigned long long my_orphans = 0;
        if (1) {
            if (use_gpu) {
                hipMemset(d_orphans, 0, sizeof(unsigned long long));
                hipMemset(d_hard_count, 0, sizeof(int));
                int threads_per_block = 256;
                int blocks = 2048; // 524288 total threads
                
                // Launch Phase 1: Fast DFS
                search_kernel_dfs<<<blocks, threads_per_block>>>(T_current, T_end, my_start_hash, my_end_hash, HASH_RANGE, d_orphans, d_hard_states, d_hard_count, d_valid_p1p2, d_valid_count);
                hipDeviceSynchronize();
                
                int h_hard_count = 0;
                hipMemcpy(&h_hard_count, d_hard_count, sizeof(int), hipMemcpyDeviceToHost);
                if (h_hard_count > 5000000) h_hard_count = 5000000;
                
                // Launch Phase 2: Host Solver
                hipMemcpy(&my_orphans, d_orphans, sizeof(unsigned long long), hipMemcpyDeviceToHost);
                
                if (h_hard_count > 0) {
                    printf("GPU Rank %d: Processing %d Hard States on CPU...\n", rank, h_hard_count);
                    fflush(stdout);
                    uint64_t* h_hard_states = (uint64_t*)malloc(h_hard_count * sizeof(uint64_t));
                    hipMemcpy(h_hard_states, d_hard_states, h_hard_count * sizeof(uint64_t), hipMemcpyDeviceToHost);
                    
                    std::atomic<unsigned long long> hard_orphans(0);
                    std::atomic<int> next_task(0);
                    int num_threads = 16;
                    std::vector<std::thread> threads;
                    
                    for (int t = 0; t < num_threads; t++) {
                        threads.emplace_back([&]() {
                            while (true) {
                                int i = next_task.fetch_add(1);
                                if (i >= h_hard_count) break;
                                uint64_t T_hard = h_hard_states[i];
                                uint32_t cache[32];
                                for (int c = 0; c < 32; c++) cache[c] = 0xFFFFFFFF;
                                
                                bool found = false;
                                for (int v = 0; v < h_valid_count[T_hard & 127]; v++) {
                                    uint32_t pair = h_valid_p1p2[(T_hard & 127) * 262144 + v];
                                    int p1 = pair >> 9;
                                    int p2 = pair & 511;
                                    
                                    int global_nodes = 0;
                                    int res = has_predecessor_dfs(p1, p2, T_hard, cache, &global_nodes, 2000000000);
                                    if (res == 1) {
                                        found = true;
                                        break;
                                    }
                                }
                                
                                if (!found) { hard_orphans.fetch_add(1); printf("\nFOUND HARD ORPHAN: %llu\n", (unsigned long long)T_hard); printf("\nFOUND ORPHAN: %llu\n", (unsigned long long)T_hard); fflush(stdout); }
                            }
                        });
                    }
                    for (auto& t : threads) t.join();
                    free(h_hard_states);
                    my_orphans += hard_orphans;
                }
                
            }
        } else {
            // CPU fallback (DFS only for now)
            unsigned long long my_orphans = 0;
            #pragma omp parallel for reduction(+:my_orphans) schedule(dynamic, 1024)
            for (uint64_t p = T_current; p < T_end; p++) {
                                if (!is_canonical(p)) continue;
                
                uint32_t cache[32];
                for (int i = 0; i < 32; i++) cache[i] = 0xFFFFFFFF;
                
                int status = 0;
                int global_nodes = 0;
                int T_row = p & 127;
                int count = h_valid_count[T_row];
                for (int i = 0; i < count && status == 0; i++) {
                    uint32_t pair = h_valid_p1p2[T_row * 262144 + i];
                    int p1 = pair >> 9;
                    int p2 = pair & 511;
                    int res = has_predecessor_dfs(p1, p2, p, cache, &global_nodes, 2000000000); // CPU uses high bailout
                    if (res == 1) status = 1;
                    else if (res == 2) status = 2; // Hard state ignored on CPU for now
                }
                
                if (status == 0) my_orphans++;
            }
            
        }

        double my_compute_time = 1.0 - compute_start;
        if (my_compute_time < 0.001) my_compute_time = 0.001;
        double my_throughput = node_weights[rank] / my_compute_time;
        if (my_avg_compute_time < 0.0) my_avg_compute_time = my_throughput; 
        else my_avg_compute_time = 0.8 * my_avg_compute_time + 0.2 * my_throughput;

        double my_new_weight = my_avg_compute_time;
        
                
        double total_new_weight = 0;
        for(int i = 0; i < world_size; i++) {
            if (node_weights[i] < 0.01) node_weights[i] = 0.01; // Enforce minimum floor to prevent starvation
            total_new_weight += node_weights[i];
        }
        if(total_new_weight > 0.0 && !isnan(total_new_weight)) { for(int i = 0; i < world_size; i++) node_weights[i] = (node_weights[i] / total_new_weight) * 100.0; } else { for(int i = 0; i < world_size; i++) node_weights[i] = 100.0 / world_size; }

        uint64_t my_canonical = 0, my_dfs = 0;
        uint64_t global_orphans = 0, global_canonical = 0, global_dfs = 0;
        global_orphans = my_orphans;
        global_orphans = my_orphans;
        global_orphans = my_orphans;
        
        if (rank == 0) {
            total_orphans += global_orphans;
            total_canonical += global_canonical;
            total_dfs += global_dfs;
        }
        
        

        if (rank == 0) {
            FILE* f = NULL;
            if (f) {
                fprintf(f, "%llu\n", (unsigned long long)T_end);
                for (int i = 0; i < world_size; i++) {
                    fprintf(f, "%lf\n", node_weights[i]);
                }
                fclose(f);
            }
            printf("goe_search_7x7 %5.2f%% | Orphans: %llu | Canonical: %llu | DFS Nodes: %llu | Time: %.2f s\n", 
                   (double)T_end / T_MAX * 100.0, total_orphans, total_canonical, total_dfs, 1.0 - compute_start);
            fflush(stdout);
        }

        T_current = T_end;
            }

    if (use_gpu && d_orphans) hipFree(d_orphans);
    if (use_gpu && d_hard_states) hipFree(d_hard_states);
    if (use_gpu && d_hard_count) hipFree(d_hard_count);
    if (use_gpu && d_valid_p1p2) hipFree(d_valid_p1p2);
    if (use_gpu && d_valid_count) hipFree(d_valid_count);

        return 0;
}
