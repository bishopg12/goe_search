"""
Module: verify_compression.py

This module verifies a bitwise compression scheme used for tracking
Game of Life states across multiple cell generations. It compares a full
state transition table against a compressed representation to ensure no
errors are introduced during the compression process.
"""

def evolve(p0, p1, p2):
    """
    Evolves three rows of states to compute the resulting target states.
    
    Args:
        p0 (int): Integer representing the bitmask of row 0.
        p1 (int): Integer representing the bitmask of row 1 (center row).
        p2 (int): Integer representing the bitmask of row 2.
        
    Returns:
        int: Bitmask representing the new states of the target row.
    """
    T = 0
    for i in range(7):
        sum = 0
        # Calculate number of alive neighbors for cell i
        sum += ((p0 >> i) & 1) + ((p0 >> (i+1)) & 1) + ((p0 >> (i+2)) & 1)
        sum += ((p1 >> i) & 1) +                       ((p1 >> (i+2)) & 1)
        sum += ((p2 >> i) & 1) + ((p2 >> (i+1)) & 1) + ((p2 >> (i+2)) & 1)
        center = (p1 >> (i+1)) & 1
        
        # Apply Game of Life rules
        if sum == 3 or (sum == 2 and center == 1):
            T |= (1 << i)
    return T

# Build the small table
# small_left_p3 stores the allowed bits of p3 for given T, p1, and p2 combinations
small_left_p3 = [[[0 for _ in range(16)] for _ in range(16)] for _ in range(4)]
for p1 in range(16):
    for p2 in range(16):
        for p3 in range(16):
            # p1, p2, p3 are 4 bits. We only care about T[0..1]
            # wait, evolve needs bits up to i+2. For i=1, it needs bit 3.
            # So 4 bits is exactly enough for T[0..1]!
            T = 0
            for i in range(2):
                s = 0
                s += ((p1 >> i) & 1) + ((p1 >> (i+1)) & 1) + ((p1 >> (i+2)) & 1)
                s += ((p2 >> i) & 1) +                       ((p2 >> (i+2)) & 1)
                s += ((p3 >> i) & 1) + ((p3 >> (i+1)) & 1) + ((p3 >> (i+2)) & 1)
                center = (p2 >> (i+1)) & 1
                
                # Check survival or birth
                if s == 3 or (s == 2 and center == 1):
                    T |= (1 << i)
            # Record that this p3 can lead to state T given p1 and p2
            small_left_p3[T][p1][p2] |= (1 << p3)

def expand_mask2(mask):
    """
    Expands a 16-bit mask to a 64-bit mask by duplicating each bit 4 times.
    
    Args:
        mask (int): A 16-bit mask to be expanded.
        
    Returns:
        int: The 64-bit expanded mask.
    """
    res = 0
    for i in range(16):
        if (mask >> i) & 1:
            res |= (0xF << (i * 4))
    return res

# Verify against the big table
errors = 0
for p1 in range(64):
    for p2 in range(64):
        for p3 in range(64):
            # Compute expected target mask
            T = evolve(p1, p2, p3) & 15
            
            # Big table logic:
            # left_p3[T][p1][p2] has bit p3 set.
            
            # Our compressed logic:
            # We split the problem into two 2-bit chunks
            T_01 = T & 3
            T_23 = (T >> 2) & 3
            
            # Extract relevant 4-bit windows from p1 and p2
            p1_03 = p1 & 15
            p2_03 = p2 & 15
            p1_25 = (p1 >> 2) & 15
            p2_25 = (p2 >> 2) & 15
            
            # Retrieve masks from the precomputed small table
            mask1 = small_left_p3[T_01][p1_03][p2_03]
            mask2 = small_left_p3[T_23][p1_25][p2_25]
            
            # Expand masks to match full 64-bit layout
            expanded1 = mask1 | (mask1 << 16) | (mask1 << 32) | (mask1 << 48)
            expanded2 = expand_mask2(mask2)
            
            # Combine masks
            final_mask = expanded1 & expanded2
            
            # If the specific p3 bit is not set in the final_mask, we have an error
            if not ((final_mask >> p3) & 1):
                errors += 1

print(f"Errors: {errors}")
