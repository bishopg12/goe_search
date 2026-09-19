import sys
import os
import random
import subprocess
import multiprocessing
import time
from generate_cnf import generate_cnf

def worker(worker_id, w, h, kissat_path, cnf_header, cnf_body):
    p_width, p_height = w + 2, h + 2
    attempts = 0
    
    while True:
        target = 0
        for i in range(49):
            if random.random() < 0.95:
                target |= (1 << i)
        
        attempts += 1
        
        target_clauses = []
        for y in range(h):
            for x in range(w):
                t_var = p_width * p_height + y * w + x + 1
                bit_idx = y * w + x
                alive = (target >> bit_idx) & 1
                target_clauses.append(f"{t_var} 0" if alive else f"{-t_var} 0")
                
        target_str = "\n".join(target_clauses) + "\n"
        full_cnf = f"p cnf {cnf_header[0]} {cnf_header[1] + len(target_clauses)}\n{cnf_body}{target_str}"
        
        process = subprocess.Popen([kissat_path, "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate(input=full_cnf.encode())
        
        if process.returncode == 20:
            print(f"\n[!] SUCCESS! FOUND GARDEN OF EDEN! (Worker {worker_id}, Attempt {attempts})", flush=True)
            print(f"Target State (49-bit int): {target}", flush=True)
            print("Grid:", flush=True)
            for y in range(h):
                row_str = ""
                for x in range(w):
                    bit_idx = y * w + x
                    alive = (target >> bit_idx) & 1
                    row_str += "O" if alive else "."
                print(row_str, flush=True)
            sys.exit(0)
            
        if attempts % 100 == 0:
            print(f"Worker {worker_id} evaluated {attempts} states...", flush=True)

if __name__ == "__main__":
    w, h = 7, 7
    kissat_path = "/home/bishopg/antigravity/kissat/build/kissat"
    num_vars, clauses = generate_cnf(w, h)
    cnf_body_lines = [" ".join(map(str, c)) + " 0" for c in clauses]
    cnf_body = "\n".join(cnf_body_lines) + "\n"
    cnf_header = (num_vars, len(clauses))
    num_cores = multiprocessing.cpu_count()
    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=worker, args=(i, w, h, kissat_path, cnf_header, cnf_body))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
