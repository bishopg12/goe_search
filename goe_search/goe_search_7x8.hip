/**
 * @file goe_search_7x8.hip
 * @brief Game of Life orphan search implementation for 7x8 bounding boxes.
 *
 * This file implements an optimized search algorithm to find Garden of Eden (orphan) states
 * in Conway's Game of Life within a 7x8 grid. It leverages HIP for execution on AMD GPUs,
 * MPI for distributed computing, and heavily utilizes bitwise logic and shared memory caching
 * to maximize throughput.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
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
 * @param T The 56-bit integer representing the 7x8 grid.
 * @return True if canonical, False if a smaller symmetric variant exists.
 */
__host__ __device__ bool is_canonical(uint64_t T) {
    uint64_t href = 0, vref = 0, rot180 = 0;
    for (int r = 0; r < 8; r++) {
        int row = (T >> (r * 7)) & 127;
        int rev = rev7(row);
        href |= ((uint64_t)rev) << (r * 7);
        vref |= ((uint64_t)row) << ((7 - r) * 7);
        rot180 |= ((uint64_t)rev) << ((7 - r) * 7);
    }
    return !(href < T || vref < T || rot180 < T);
}

// Macros for simulating a stack in the DFS to avoid dynamic memory or recursion (8 depth for 7x8)
#define GET_P1(sp) ((sp)==0?p1_0:(sp)==1?p1_1:(sp)==2?p1_2:(sp)==3?p1_3:(sp)==4?p1_4:(sp)==5?p1_5:(sp)==6?p1_6:p1_7)
#define GET_P2(sp) ((sp)==0?p2_0:(sp)==1?p2_1:(sp)==2?p2_2:(sp)==3?p2_3:(sp)==4?p2_4:(sp)==5?p2_5:(sp)==6?p2_6:p2_7)
#define GET_L_MASK(sp) ((sp)==0?l_mask_0:(sp)==1?l_mask_1:(sp)==2?l_mask_2:(sp)==3?l_mask_3:(sp)==4?l_mask_4:(sp)==5?l_mask_5:(sp)==6?l_mask_6:l_mask_7)
#define GET_R_MASK(sp) ((sp)==0?r_mask_0:(sp)==1?r_mask_1:(sp)==2?r_mask_2:(sp)==3?r_mask_3:(sp)==4?r_mask_4:(sp)==5?r_mask_5:(sp)==6?r_mask_6:r_mask_7)
#define GET_VALID_R(sp) ((sp)==0?valid_r_0:(sp)==1?valid_r_1:(sp)==2?valid_r_2:(sp)==3?valid_r_3:(sp)==4?valid_r_4:(sp)==5?valid_r_5:(sp)==6?valid_r_6:valid_r_7)
#define GET_P3_L(sp) ((sp)==0?p3_l_0:(sp)==1?p3_l_1:(sp)==2?p3_l_2:(sp)==3?p3_l_3:(sp)==4?p3_l_4:(sp)==5?p3_l_5:(sp)==6?p3_l_6:p3_l_7)

#define SET_P1(sp, val) do { if ((sp)==0) p1_0=(val); else if ((sp)==1) p1_1=(val); else if ((sp)==2) p1_2=(val); else if ((sp)==3) p1_3=(val); else if ((sp)==4) p1_4=(val); else if ((sp)==5) p1_5=(val); else if ((sp)==6) p1_6=(val); else p1_7=(val); } while(0)
#define SET_P2(sp, val) do { if ((sp)==0) p2_0=(val); else if ((sp)==1) p2_1=(val); else if ((sp)==2) p2_2=(val); else if ((sp)==3) p2_3=(val); else if ((sp)==4) p2_4=(val); else if ((sp)==5) p2_5=(val); else if ((sp)==6) p2_6=(val); else p2_7=(val); } while(0)
#define SET_L_MASK(sp, val) do { if ((sp)==0) l_mask_0=(val); else if ((sp)==1) l_mask_1=(val); else if ((sp)==2) l_mask_2=(val); else if ((sp)==3) l_mask_3=(val); else if ((sp)==4) l_mask_4=(val); else if ((sp)==5) l_mask_5=(val); else if ((sp)==6) l_mask_6=(val); else l_mask_7=(val); } while(0)
#define SET_R_MASK(sp, val) do { if ((sp)==0) r_mask_0=(val); else if ((sp)==1) r_mask_1=(val); else if ((sp)==2) r_mask_2=(val); else if ((sp)==3) r_mask_3=(val); else if ((sp)==4) r_mask_4=(val); else if ((sp)==5) r_mask_5=(val); else if ((sp)==6) r_mask_6=(val); else r_mask_7=(val); } while(0)
#define SET_VALID_R(sp, val) do { if ((sp)==0) valid_r_0=(val); else if ((sp)==1) valid_r_1=(val); else if ((sp)==2) valid_r_2=(val); else if ((sp)==3) valid_r_3=(val); else if ((sp)==4) valid_r_4=(val); else if ((sp)==5) valid_r_5=(val); else if ((sp)==6) valid_r_6=(val); else valid_r_7=(val); } while(0)
#define SET_P3_L(sp, val) do { if ((sp)==0) p3_l_0=(val); else if ((sp)==1) p3_l_1=(val); else if ((sp)==2) p3_l_2=(val); else if ((sp)==3) p3_l_3=(val); else if ((sp)==4) p3_l_4=(val); else if ((sp)==5) p3_l_5=(val); else if ((sp)==6) p3_l_6=(val); else p3_l_7=(val); } while(0)

/**
 * @brief Depth-First Search to find a predecessor for pattern T.
 *
 * Implements a non-recursive backtracking search to see if a given pattern T
 * has any valid predecessor. Uses a thread-local cache array to prevent redundant
 * sub-tree searches. If a search takes too long, it bails out, marking it a 'hard' state.
 *
 * @param start_p1 Initial row -1
 * @param start_p2 Initial row 0
 * @param T The target 7x8 pattern
 * @param cache Shared memory or local cache to store failed states
 * @param global_nodes Pointer to track nodes explored (for bailout)
 * @return 1 if predecessor found, 0 if orphan, 2 if bailout (hard state)
 */
__device__ int has_predecessor_dfs(int start_p1, int start_p2, uint64_t T, uint32_t* cache, int* global_nodes) {
    uint64_t l_mask_0 = 0, l_mask_1 = 0, l_mask_2 = 0, l_mask_3 = 0, l_mask_4 = 0, l_mask_5 = 0, l_mask_6 = 0, l_mask_7 = 0;
    uint32_t r_mask_0 = 0, r_mask_1 = 0, r_mask_2 = 0, r_mask_3 = 0, r_mask_4 = 0, r_mask_5 = 0, r_mask_6 = 0, r_mask_7 = 0;
    uint32_t valid_r_0 = 0, valid_r_1 = 0, valid_r_2 = 0, valid_r_3 = 0, valid_r_4 = 0, valid_r_5 = 0, valid_r_6 = 0, valid_r_7 = 0;
    int p1_0 = start_p1, p1_1 = 0, p1_2 = 0, p1_3 = 0, p1_4 = 0, p1_5 = 0, p1_6 = 0, p1_7 = 0;
    int p2_0 = start_p2, p2_1 = 0, p2_2 = 0, p2_3 = 0, p2_4 = 0, p2_5 = 0, p2_6 = 0, p2_7 = 0;
    int p3_l_0 = 0, p3_l_1 = 0, p3_l_2 = 0, p3_l_3 = 0, p3_l_4 = 0, p3_l_5 = 0, p3_l_6 = 0, p3_l_7 = 0;
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
            if (*global_nodes > 5000) return 2; // Hard state bailout threshold for 7x8
        }
        
        uint32_t c_valid_r = GET_VALID_R(sp);
        if (c_valid_r) {
            int p3_r = __builtin_ctz(c_valid_r);
            SET_VALID_R(sp, c_valid_r & (c_valid_r - 1));
            int p3 = GET_P3_L(sp) | (p3_r << 4);
            
            if (sp == 7) return 1; // Success! Valid predecessor found up to the 8th row
            
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
 * Resolves states that hit the DFS bailout threshold (8 levels deep for 7x8).
 *
 * @param d_hard_states Array of hard states to process
 * @param h_hard_count Number of hard states
 * @param d_orphans Pointer to accumulate orphan counts
 */
__global__ void search_kernel_bfs(uint64_t* d_hard_states, int h_hard_count, unsigned long long* d_orphans) {
    __shared__ uint32_t s_current[8192]; // Bit vector for current frontier
    __shared__ uint32_t s_next[8192];    // Bit vector for next frontier
    
    int block_id = blockIdx.x;
    if (block_id >= h_hard_count) return;
    
    uint64_t T = d_hard_states[block_id];
    int tid = threadIdx.x;
    
    // Initialize full current frontier
    for (int i = tid; i < 8192; i += blockDim.x) {
        s_current[i] = 0xFFFFFFFF;
    }
    __syncthreads();
    
    bool empty = false;
    // Iterate through all 8 rows
    for (int r = 0; r < 8 && !empty; r++) {
        for (int i = tid; i < 8192; i += blockDim.x) {
            s_next[i] = 0;
        }
        __syncthreads();
        
        int T_r = (T >> (r * 7)) & 127;
        int T_01 = T_r & 3, T_23 = (T_r >> 2) & 3, T_ri = (T_r >> 4) & 7;
        
        for (int i = tid; i < 8192; i += blockDim.x) {
            uint32_t word = s_current[i];
            while (word) {
                int bit = __builtin_ctz(word);
                word &= (word - 1);
                
                int state = (i * 32) + bit;
                int p1 = state >> 9;
                int p2 = state & 511;
                
                int p1_03 = p1 & 15, p1_25 = (p1 >> 2) & 15, p1_r = (p1 >> 4) & 31;
                int p2_03 = p2 & 15, p2_25 = (p2 >> 2) & 15, p2_r = (p2 >> 4) & 31;
                
                uint64_t mask1 = c_small_left_p3[T_01][p1_03][p2_03];
                uint64_t mask2 = c_small_left_p3[T_23][p1_25][p2_25];
                uint64_t expanded1 = mask1 | (mask1 << 16) | (mask1 << 32) | (mask1 << 48);
                uint64_t expanded2 = c_expand_8bit[mask2 & 255] | ((uint64_t)c_expand_8bit[mask2 >> 8] << 32);
                uint64_t l_mask = expanded1 & expanded2;
                uint32_t r_mask = c_right_p3[T_ri][p1_r][p2_r];
                
                while (l_mask) {
                    int p3_l = __builtin_ctzll(l_mask);
                    l_mask &= (l_mask - 1);
                    
                    uint32_t valid_r = r_mask & c_overlap_mask[p3_l >> 4];
                    while (valid_r) {
                        int p3_r = __builtin_ctz(valid_r);
                        valid_r &= (valid_r - 1);
                        int p3 = p3_l | (p3_r << 4);
                        
                        int next_state = (p2 << 9) | p3;
                        // Thread-safe OR operation into next frontier bit vector
                        atomicOr(&s_next[next_state / 32], 1U << (next_state % 32));
                    }
                }
            }
        }
        __syncthreads();
        
        if (tid == 0) empty = true;
        __syncthreads();
        
        // Copy next frontier to current and check if search space is exhausted
        for (int i = tid; i < 8192; i += blockDim.x) {
            if (s_next[i]) empty = false;
            s_current[i] = s_next[i];
        }
        __syncthreads();
    }
    
    if (tid == 0 && empty) {
        atomicAdd(d_orphans, 1);
    }
}

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
        if (mix_hash(p) % HASH_RANGE < my_start_hash || mix_hash(p) % HASH_RANGE >= my_end_hash) continue;
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
            int res = has_predecessor_dfs(p1, p2, p, cache, &global_nodes);
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

    if (local_orphans > 0) {
        atomicAdd(d_orphans, local_orphans);
    }
}

/**
 * @brief Main execution function orchestrating MPI distribution and GPU execution.
 */
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    bool use_gpu = false;
    int hip_device_count = 0;
    if (hipGetDeviceCount(&hip_device_count) == hipSuccess && hip_device_count > 0) {
        use_gpu = true;
        hipSetDevice(0);
    }

    if (rank == 0 || use_gpu) {
        printf("Rank %d on %s: GPU Count: %d, HIP Error: %d\n", rank, getenv("SLURMD_NODENAME"), hip_device_count, 0);
    }

    // Pre-compute host tables
    init_tables();
    
    unsigned long long* d_orphans = NULL;
    uint64_t* d_hard_states = NULL;
    int* d_hard_count = NULL;
    uint32_t* d_valid_p1p2 = NULL;
    int* d_valid_count = NULL;

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
    
    uint64_t T_MAX = 1ULL << 56; // 7x8 state space
    uint64_t step_size = 45000000000ULL; // Process in chunks of 45B states
    uint64_t T_current = 0;
    
    // Checkpointing and Load Balancing
    FILE* ckpt = fopen("checkpoint_7x8.dat", "r");
    int node_weights[16];
    for (int i = 0; i < world_size; i++) node_weights[i] = 100 / world_size;
    node_weights[0] += 100 % world_size;
    if (ckpt) {
        fscanf(ckpt, "%llu", (unsigned long long*)&T_current);
        for (int i = 0; i < world_size; i++) {
            fscanf(ckpt, "%d", &node_weights[i]);
        }
        fclose(ckpt);
    }

    // Determine MPI node hash bounds
    uint64_t HASH_RANGE = 10000;
    uint64_t my_start_hash = 0, my_end_hash = 0;
    uint64_t current_hash = 0;
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            my_start_hash = current_hash;
            my_end_hash = current_hash + (node_weights[i] * HASH_RANGE / 100);
        }
        current_hash += (node_weights[i] * HASH_RANGE / 100);
    }

    unsigned long long total_orphans = 0;

    printf("Rank %d: Initialization finished! Beginning search...\n", rank);
    fflush(stdout);
    
    double start_time = MPI_Wtime(), last_checkpoint_time = MPI_Wtime();

    // Main search loop
    while (T_current < T_MAX) {
        uint64_t T_end = T_current + step_size;
        if (T_end > T_MAX) T_end = T_MAX;

        double compute_start = MPI_Wtime();

        if (node_weights[rank] > 0.0) {
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
                
                // Launch Phase 2: BFS for hard states
                if (h_hard_count > 0) {
                    printf("GPU Rank %d: Processing %d Hard States with BFS...\n", rank, h_hard_count);
                    fflush(stdout);
                    search_kernel_bfs<<<h_hard_count, 256>>>(d_hard_states, h_hard_count, d_orphans);
                    hipDeviceSynchronize();
                }
                
                unsigned long long my_orphans = 0;
                hipMemcpy(&my_orphans, d_orphans, sizeof(unsigned long long), hipMemcpyDeviceToHost);
                
                // Reduce orphan counts globally
                unsigned long long global_step_orphans = 0;
                MPI_Allreduce(&my_orphans, &global_step_orphans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
                total_orphans += global_step_orphans;
            }
        } else {
            // Wait for CPU fallback or sync
            unsigned long long my_orphans = 0;
            unsigned long long global_step_orphans = 0;
            MPI_Allreduce(&my_orphans, &global_step_orphans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            total_orphans += global_step_orphans;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            FILE* f = fopen("checkpoint_7x8.dat", "w");
            if (f) {
                fprintf(f, "%llu\n", (unsigned long long)T_end);
                for (int i = 0; i < world_size; i++) {
                    fprintf(f, "%d\n", node_weights[i]);
                }
                fclose(f);
            }
            printf("Progress: %llu / %llu (%.2f%%) - Orphans found: %llu, Time: %.2f s\n", 
                   (unsigned long long)T_end, (unsigned long long)T_MAX, 
                   (double)T_end / T_MAX * 100.0, total_orphans, MPI_Wtime() - compute_start);
            fflush(stdout);
        }

        T_current = T_end;
    }

    if (use_gpu && d_orphans) hipFree(d_orphans);
    if (use_gpu && d_hard_states) hipFree(d_hard_states);
    if (use_gpu && d_hard_count) hipFree(d_hard_count);
    if (use_gpu && d_valid_p1p2) hipFree(d_valid_p1p2);
    if (use_gpu && d_valid_count) hipFree(d_valid_count);

    MPI_Finalize();
    return 0;
}
