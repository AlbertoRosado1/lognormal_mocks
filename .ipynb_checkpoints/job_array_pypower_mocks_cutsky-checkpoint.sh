#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J pk_cutsky
#SBATCH -t 00:15:00
#SBATCH -L SCRATCH
#SBATCH -o slurm/pk_cutsky_%A_%a.out
#SBATCH --array=1-100

echo $SLURM_ARRAY_TASK_ID
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

srun -n 128 python pypower_cutsky_mocks.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1.2 --nmesh 1024 --nrandoms 20 --case contiguous --id $SLURM_ARRAY_TASK_ID --use_single_mock

srun -n 128 python pypower_cutsky_mocks.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1.2 --nmesh 1024 --nrandoms 20 --case checkerboard --id $SLURM_ARRAY_TASK_ID --use_single_mock
