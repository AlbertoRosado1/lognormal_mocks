#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J cutsky_randoms
#SBATCH -t 00:15:00
#SBATCH -o slurm/cutsky_rands_%A_%a.out
#SBATCH -L SCRATCH
#SBATCH --array=1-20

echo $SLURM_ARRAY_TASK_ID
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

python generate_cutsky_randoms.py --nbar 2500 --zmin 2.75 --zmax 3.25 --boxsize 8000 --ramin 175 --ramax 290 --decmin -10 --decmax 70 --version test1 --cutsky_version test1.2 --id $SLURM_ARRAY_TASK_ID
