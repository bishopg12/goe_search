/**
 * @file count_unique.c
 * @brief Computes a unique state count for a compressed Game of Life search representation.
 *
 * This program populates a 3-dimensional array tracking valid state transitions
 * for 3 consecutive rows, using bitwise operations to compute the Game of Life
 * rules. It then counts the number of unique transition masks across a section
 * of the array.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Array to store the allowed values for the 3rd row given the target and first two rows */
uint64_t left_p3[16][64][64];

/**
 * @brief Main function of the program.
 *
 * Iterates through all possible bit combinations for 3 neighboring rows,
 * applies Game of Life logic, populates the state transition table,
 * and finally counts and prints the number of unique values generated.
 *
 * @return 0 on successful execution.
 */
int main() {
    // Iterate over all 9-bit patterns for 3 consecutive rows
    for (int p1 = 0; p1 < 512; p1++) {
        for (int p2 = 0; p2 < 512; p2++) {
            for (int p3 = 0; p3 < 512; p3++) {
                int row_val = 0;
                
                // Process 7 columns
                for (int c = 1; c <= 7; c++) {
                    // Extract a 3-bit window for each row, shifted by c-1
                    int c1 = (p1 >> (c - 1)) & 7, c2 = (p2 >> (c - 1)) & 7, c3 = (p3 >> (c - 1)) & 7;
                    
                    // Sum up all alive cells in the 3x3 neighborhood
                    int live = (c1 & 1) + ((c1 >> 1) & 1) + ((c1 >> 2) & 1) +
                               (c2 & 1) + ((c2 >> 2) & 1) +
                               (c3 & 1) + ((c3 >> 1) & 1) + ((c3 >> 2) & 1);
                               
                    // Check if the center cell is alive
                    int center = (c2 >> 1) & 1;
                    
                    // Apply Game of Life rules: survive on 2/3, born on 3
                    if (live == 3 || (center && live == 2)) row_val |= (1 << (c - 1));
                }
                
                // Extract relevant subsets of the bit patterns
                int T_left = row_val & 15, p1_l = p1 & 63, p2_l = p2 & 63, p3_l = p3 & 63;
                
                // Record that this combination of p3 is valid for T_left, p1_l, and p2_l
                left_p3[T_left][p1_l][p2_l] |= (1ULL << p3_l);
            }
        }
    }

    // Allocate an array to store unique elements to prevent stack overflow
    uint64_t* unique = malloc(65536 * sizeof(uint64_t));
    int count = 0;
    
    // Count the unique bitmasks in the generated table for specific index ranges
    for (int t = 0; t < 16; t++) {
        for (int p1 = 0; p1 < 64; p1++) {
            for (int p2 = 0; p2 < 64; p2++) {
                uint64_t val = left_p3[t][p1][p2];
                int found = 0;
                
                // Check if the value is already in the unique array
                for (int i = 0; i < count; i++) {
                    if (unique[i] == val) { found = 1; break; }
                }
                
                // Add the value if it hasn't been encountered yet
                if (!found) { unique[count++] = val; }
            }
        }
    }
    
    // Output the final count
    printf("Unique values: %d\n", count);
    return 0;
}
