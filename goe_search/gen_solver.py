import sys

W = int(sys.argv[1])
H = int(sys.argv[2])

P_WIDTH = W + 2
P_STATES = 1 << P_WIDTH
T_STATES = 1 << W
T_BITS = W * H

code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>
#include <string.h>

uint64_t valid_p3[{T_STATES}][{P_STATES}][{P_STATES}][{(P_STATES + 63) // 64}] __attribute__((aligned(32)));

int cell_next(int p0, int p1, int p2, int c) {{
    int sum = 0;
    for (int i = c; i <= c + 2; i++) {{
        sum += (p0 >> i) & 1;
        if (i != c + 1) sum += (p1 >> i) & 1;
        sum += (p2 >> i) & 1;
    }}
    int alive = (p1 >> (c + 1)) & 1;
    if (alive && (sum == 2 || sum == 3)) return 1;
    if (!alive && sum == 3) return 1;
    return 0;
}}

void init_tables() {{
    memset(valid_p3, 0, sizeof(valid_p3));
    for (int p0 = 0; p0 < {P_STATES}; p0++) {{
        for (int p1 = 0; p1 < {P_STATES}; p1++) {{
            for (int p2 = 0; p2 < {P_STATES}; p2++) {{
                int T_k = 0;
                for (int c = 0; c < {W}; c++) {{
                    T_k |= cell_next(p0, p1, p2, c) << c;
                }}
                valid_p3[T_k][p1][p2][p0 / 64] |= (1ULL << (p0 % 64));
            }}
        }}
    }}
}}

int revW(int row) {{
    int rev = 0;
    for (int i = 0; i < {W}; i++) {{
        if ((row >> i) & 1) rev |= (1 << ({W} - 1 - i));
    }}
    return rev;
}}

bool is_canonical(uint64_t T) {{
    uint64_t href = 0, vref = 0, rot180 = 0;
    for (int r = 0; r < {H}; r++) {{
        int row = (T >> (r * {W})) & {T_STATES - 1};
        int rev = revW(row);
        href |= ((uint64_t)rev) << (r * {W});
        vref |= ((uint64_t)row) << (({H} - 1 - r) * {W});
        rot180 |= ((uint64_t)rev) << (({H} - 1 - r) * {W});
    }}
    if (href < T || vref < T || rot180 < T) return false;
    return true;
}}

bool has_predecessor(int row, int p1, int p2, uint64_t T, int current_id, int failed[{H}][{P_STATES}][{P_STATES}]) {{
    if (row == {H}) return true;
    if (failed[row][p1][p2] == current_id) return false;
    int T_row = (T >> (row * {W})) & {T_STATES - 1};
    
    for (int chunk = 0; chunk < {(P_STATES + 63) // 64}; chunk++) {{
        uint64_t m = valid_p3[T_row][p1][p2][chunk];
        while (m) {{
            int p3 = chunk * 64 + __builtin_ctzll(m);
            if (has_predecessor(row + 1, p2, p3, T, current_id, failed)) return true;
            m &= m - 1;
        }}
    }}
    failed[row][p1][p2] = current_id;
    return false;
}}

int main() {{
    init_tables();
    uint64_t total_T = 1ULL << {T_BITS};
    uint64_t global_orphans = 0;
    double start_time = omp_get_wtime();
    
    #pragma omp parallel
    {{
        int failed[{H}][{P_STATES}][{P_STATES}] = {{{0}}};
        int current_id = 1;
        uint64_t local_orphans = 0;
        
        #pragma omp for schedule(dynamic, 10000)
        for (uint64_t T = 0; T < total_T; T++) {{
            if (!is_canonical(T)) continue;
            current_id++;
            if (current_id > 1000000000) {{ memset(failed, 0, sizeof(failed)); current_id = 1; }}
            
            bool is_orphan = true;
            for (int p1 = 0; p1 < {P_STATES} && is_orphan; p1++) {{
                for (int p2 = 0; p2 < {P_STATES} && is_orphan; p2++) {{
                    if (has_predecessor(0, p1, p2, T, current_id, failed)) {{
                        is_orphan = false;
                    }}
                }}
            }}
            if (is_orphan) local_orphans++;
        }}
        #pragma omp atomic
        global_orphans += local_orphans;
    }}
    
    double end_time = omp_get_wtime();
    printf("==================================================\\n");
    printf("Search {W}x{H} Completed in %.2f seconds.\\n", end_time - start_time);
    if (global_orphans == 0) {{
        printf("PROOF SUCCESSFUL: There are NO orphan states in a {W}x{H} grid.\\n");
    }} else {{
        printf("FOUND %llu orphan states (modulo symmetry) in a {W}x{H} grid.\\n", (unsigned long long)global_orphans);
    }}
    printf("==================================================\\n");
    return 0;
}}
"""
with open(f"goe_search_{W}x{H}.c", "w") as f:
    f.write(code)
