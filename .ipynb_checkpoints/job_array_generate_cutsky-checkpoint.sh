#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J cutsky
#SBATCH -t 00:15:00
#SBATCH -L SCRATCH
#SBATCH -o slurm/cutsky_%A_%a.out
#SBATCH --array=1-100

echo $SLURM_ARRAY_TASK_ID
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

python generate_cutsky_mock.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --ramin 175 --ramax 290 --decmin -10 --decmax 70 --version test1 --cutsky_version test1.2 --id $SLURM_ARRAY_TASK_ID
