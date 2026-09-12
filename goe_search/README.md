
## Dynamic Grid Solver Generation
The repository now includes `gen_solver.py`, a dynamic C-code generator that can instantiate highly optimized, OpenMP-accelerated Garden of Eden solvers for any arbitrary `W x H` bounding box.

To generate, compile, and run a solver for a 5x6 grid:
```bash
python3 gen_solver.py 5 6
gcc -O3 -fopenmp -march=native -ffast-math goe_search_5x6.c -o goe_5x6
./goe_5x6
```
