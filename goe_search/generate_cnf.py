"""
Module: generate_cnf.py

This module generates a Boolean Satisfiability (SAT) problem in DIMACS CNF format 
for the reverse Game of Life problem. It creates a flat and fast CNF by explicitly
defining the truth table for each cell's neighborhood.
"""

import sys

def get_gol_next_state(p_vars):
    """
    Computes the next state of a Game of Life cell given its 3x3 neighborhood.
    
    Args:
        p_vars (list[bool]): A list of 9 boolean values representing the 3x3 
                             neighborhood, where p_vars[4] is the center cell.
                             
    Returns:
        bool: True if the center cell will be alive in the next generation, 
              False otherwise.
    """
    # p_vars is a list of 9 boolean values representing the 3x3 neighborhood
    # p_vars[4] is the center cell
    alive_neighbors = sum(p_vars) - p_vars[4]
    if p_vars[4]:
        # Cell survives if it has 2 or 3 alive neighbors
        return alive_neighbors == 2 or alive_neighbors == 3
    else:
        # Cell is born if it has exactly 3 alive neighbors
        return alive_neighbors == 3

def generate_cnf(width, height):
    """
    Generates the CNF clauses for a target Game of Life grid.
    
    Args:
        width (int): The width of the target grid.
        height (int): The height of the target grid.
        
    Returns:
        tuple: A tuple containing the total number of variables and a list of clauses.
    """
    # T is the Target grid (W x H)
    # P is the Predecessor grid ((W+2) x (H+2))
    
    p_width = width + 2
    p_height = height + 2
    
    def P_var(x, y):
        """Returns the variable index for a cell in the Predecessor grid."""
        return y * p_width + x + 1
        
    def T_var(x, y):
        """Returns the variable index for a cell in the Target grid."""
        return p_width * p_height + y * width + x + 1

    clauses = []
    
    for ty in range(height):
        for tx in range(width):
            # The 9 Predecessor variables that determine Target(tx, ty)
            p_coords = [
                (tx, ty),   (tx+1, ty),   (tx+2, ty),
                (tx, ty+1), (tx+1, ty+1), (tx+2, ty+1),
                (tx, ty+2), (tx+1, ty+2), (tx+2, ty+2)
            ]
            
            p_indices = [P_var(x, y) for x, y in p_coords]
            t_index = T_var(tx, ty)
            
            # Generate 512 clauses for this cell using a truth table approach
            # This completely avoids auxiliary variables and keeps the CNF flat and fast
            for i in range(512):
                p_vals = [(i >> j) & 1 for j in range(9)]
                t_expected = get_gol_next_state(p_vals)
                
                # If a combination of P vars produces T_expected, then the opposite of T_expected is INVALID.
                # We add a clause to legally forbid this invalid combination.
                clause = []
                for j in range(9):
                    clause.append(-p_indices[j] if p_vals[j] else p_indices[j])
                
                # Forbid the opposite of the expected T
                clause.append(-t_index if not t_expected else t_index)
                clauses.append(clause)
                
    return (p_width * p_height + width * height), clauses

if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) != 3:
        print("Usage: python3 generate_cnf.py <width> <height>")
        print("Example: python3 generate_cnf.py 8 8 > gol_8x8.cnf")
        sys.exit(1)
        
    w, h = int(sys.argv[1]), int(sys.argv[2])
    num_vars, clauses = generate_cnf(w, h)
    
    # Print in standard DIMACS CNF format
    print(f"p cnf {num_vars} {len(clauses)}")
    for c in clauses:
        print(" ".join(map(str, c)) + " 0")
