#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J logmocks
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

#srun -n 2 python generate_mocks.py --bias 3.0 --nbar 1250 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --id 1
srun python generate_mocks.py --bias 3.0 --nbar 1250 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --id 1


# srun -n 64 python generate_mocks.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --version test1 --id 1
