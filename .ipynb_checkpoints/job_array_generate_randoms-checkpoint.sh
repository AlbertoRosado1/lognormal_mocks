#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J randoms
#SBATCH -t 00:15:00
#SBATCH -o slurm/rands_%A_%a.out
#SBATCH -L SCRATCH
#SBATCH --array=1-20

echo $SLURM_ARRAY_TASK_ID
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

srun -n 64 python generate_randoms_mpi.py --nbar 2500 --zmin 2.75 --zmax 3.25 --boxsize 8000 --version test1 --id $SLURM_ARRAY_TASK_ID

