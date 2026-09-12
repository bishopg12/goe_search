#!/bin/bash
#SBATCH --job-name=goe_7x7_p0
#SBATCH --output=/home/bishopg/antigravity/goe_search/hybrid_run.log
#SBATCH --partition=debug
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=soldier[0-9]
#SBATCH --exclude=soldier10,soldier11,soldier12

#SBATCH hetjob
#SBATCH --job-name=goe_7x7_p1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=soldier12
#SBATCH --exclude=soldier[0-11]

export OMPI_MCA_btl=tcp,self
export OMP_NUM_THREADS=16

srun --het-group=0,1 --mpi=pmix --cpu-bind=none /home/bishopg/antigravity/goe_search/wrapper.sh
