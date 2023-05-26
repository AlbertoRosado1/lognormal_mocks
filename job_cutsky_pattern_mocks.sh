#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J cutsky_pattern
#SBATCH -t 02:00:00
#SBATCH -L SCRATCH

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

srun -N 1 python cutsky_pattern_mock.py --bias 3.0 --nbar 1250 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --ramin 175 --ramax 290 --decmin -10 --decmax 70 --id 1 --version test1