#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J lognormal_mocks
#SBATCH -t 00:10:00
#SBATCH -L SCRATCH
#SBATCH -o slurm/%A_%a.out
#SBATCH --array=1-100

echo $SLURM_ARRAY_TASK_ID
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

srun -n 64 python generate_mocks_mpi.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --version test1 --id $SLURM_ARRAY_TASK_ID

