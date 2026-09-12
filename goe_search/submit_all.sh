#!/bin/bash
export OMP_NUM_THREADS=16

cd /home/bishopg/antigravity/goe_search

SIZES="6 7; 5 5; 5 6; 4 4; 4 5; 3 3; 3 4; 2 2; 2 3; 1 1; 1 2"
IFS=';' read -r -a size_array <<< "$SIZES"

for size_pair in "${size_array[@]}"; do
    W=$(echo $size_pair | awk '{print $1}')
    H=$(echo $size_pair | awk '{print $2}')
    
    # Python is already run
    # gcc -O3 -fopenmp -march=native -ffast-math goe_search_${W}x${H}.c -o goe_${W}x${H}
    
    # Copy to soldier10 and soldier11
    scp -o StrictHostKeyChecking=no goe_${W}x${H} bishopg@soldier10:/home/bishopg/
    scp -o StrictHostKeyChecking=no goe_${W}x${H} bishopg@soldier11:/home/bishopg/
    
    cat << SBATCH_EOF > run_${W}x${H}.sbatch
#!/bin/bash
#SBATCH --job-name=goe_${W}x${H}
#SBATCH --output=/home/bishopg/goe_${W}x${H}.out
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=soldier10,soldier11

export OMP_NUM_THREADS=16
cd /home/bishopg
./goe_${W}x${H}
SBATCH_EOF

    sbatch run_${W}x${H}.sbatch
done
