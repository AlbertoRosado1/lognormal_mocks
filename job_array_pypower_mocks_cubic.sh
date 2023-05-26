#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J pk_cubic
#SBATCH -t 00:15:00
#SBATCH -L SCRATCH
#SBATCH -o slurm/pk_cubic_%A_%a.out
#SBATCH --array=1-2

echo $SLURM_ARRAY_TASK_ID
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

srun -n 128 python pypower_cubic_mocks.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1 --nmesh 1024 --nrandoms 20 --id $SLURM_ARRAY_TASK_ID --use_single_mock

#srun -n 128 python pypower_cubic_mocks.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1 --nmesh 1024 --nrandoms 20 --id 1 --use_single_mock
