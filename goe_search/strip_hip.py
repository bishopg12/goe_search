"""
Module: strip_hip.py

This script processes a HIP C++ source file and strips out all GPU-specific 
execution blocks, macros, and initialization sequences. The output is a standard 
CPU-only C++ source file.
"""

import re

# Read the original HIP source file
with open('goe_search_7x7.hip', 'r') as f:
    code = f.read()

# Remove __global__ functions
# This regex matches the __global__ void declaration and its associated code block
code = re.sub(r'__global__\s+void\s+\w+\s*\(.*?\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', code, flags=re.DOTALL)

# One more pass for nested braces
# Some __global__ functions might have deeply nested blocks that require an extra pass
code = re.sub(r'__global__\s+void\s+\w+\s*\(.*?\)\s*\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}', '', code, flags=re.DOTALL)

# Remove hip includes
# Standard HIP headers are not needed for a CPU-only build
code = code.replace('#include <hip/hip_runtime.h>', '')

# Replace hip macros
# We define empty or dummy macros for HIP-specific annotations and calls
macros = """
#define __device__ 
#define __host__ 
#define __constant__ 
#define __shared__
#ifdef __HIPCC__
void search_kernel_dfs() {}
void search_kernel_bfs() {}
#endif
"""
code = macros + code

# Strip GPU execution block
# This replaces `if (use_gpu)` blocks with `if (0)` so they are optimized out
code = re.sub(r'if\s*\(use_gpu\)\s*\{.*?\}\s*else\s*\{', 'if (0) {} else {', code, flags=re.DOTALL)

# Strip GPU initialization
# Finds the block initializing GPU devices and removes it entirely
gpu_init = re.search(r'if\s*\(use_gpu\)\s*\{(.*?)\}\s*MPI_Barrier', code, re.DOTALL)
if gpu_init:
    code = code.replace(gpu_init.group(1), '\n')
    
# Remove hipFree
# Strips calls to free GPU memory as no memory will be allocated
code = re.sub(r'if\s*\(use_gpu.*hipFree.*?;', '', code)

# Write out the modified, CPU-only code
with open('goe_search_7x7_v7_cpu_final.cpp', 'w') as f:
    f.write(code)
