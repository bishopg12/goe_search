import sys
import subprocess
from generate_cnf import generate_cnf

if len(sys.argv) < 2:
    print("Usage: python3 verify_T_kissat.py <T_value>")
    sys.exit(1)

T = int(sys.argv[1])
w, h = 7, 7
kissat_path = "/home/bishopg/antigravity/kissat/build/kissat"
num_vars, clauses = generate_cnf(w, h)

p_width, p_height = w + 2, h + 2
target_clauses = []
for y in range(h):
    for x in range(w):
        t_var = p_width * p_height + y * w + x + 1
        bit_idx = y * w + x
        alive = (T >> bit_idx) & 1
        target_clauses.append(f"{t_var} 0" if alive else f"{-t_var} 0")

target_str = "\n".join(target_clauses) + "\n"
cnf_body_lines = [" ".join(map(str, c)) + " 0" for c in clauses]
cnf_body = "\n".join(cnf_body_lines) + "\n"
full_cnf = f"p cnf {num_vars} {len(clauses) + len(target_clauses)}\n{cnf_body}{target_str}"

process = subprocess.Popen([kissat_path, "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate(input=full_cnf.encode())

if process.returncode == 10:
    print(f"[+] State {T} is SATISFIABLE (has a predecessor, NOT an orphan).")
elif process.returncode == 20:
    print(f"[!] State {T} is UNSATISFIABLE! THIS IS A TRUE GARDEN OF EDEN!")
    print(f"Target State (49-bit int): {T}")
    for y in range(h):
        row_str = ""
        for x in range(w):
            alive = (T >> (y * w + x)) & 1
            row_str += "O" if alive else "."
        print(row_str)
else:
    print(f"Kissat error: code {process.returncode}")
