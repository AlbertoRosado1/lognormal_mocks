#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

 srun -n 128 python pypower_cubic_mocks.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1 --nmesh 1024 --nrandoms 20 --phstart 1 --phend 10