#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

for seed_id in {1..10}
do
    echo "working on generating mock $seed_id"
    srun -n 64 python generate_mocks_mpi.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --version test1 --id $seed_id
done