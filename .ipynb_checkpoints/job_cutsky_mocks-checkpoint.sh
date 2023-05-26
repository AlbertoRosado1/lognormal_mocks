#!/bin/bash
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J cutsky
#SBATCH -t 01:00:00
#SBATCH -L SCRATCH

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

srun -N 1 python cutsky_mock.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --ramin 175 --ramax 290 --decmin -10 --decmax 30 --id 1 &

srun -N 1 python cutsky_mock.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --ramin 175 --ramax 290 --decmin 50 --decmax 70 --id 2 &
wait