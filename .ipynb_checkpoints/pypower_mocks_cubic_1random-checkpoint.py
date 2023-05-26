import numpy as np
import os
import argparse
from pypower import CatalogFFTPower, setup_logging
from mpi4py import MPI
import fitsio
# python pypower_mocks_cubic.py --nmesh 512 --boxsize 8000 --randoms_fn '/pscratch/sd/a/arosado/lognormal_mocks/randoms_3.0_1250.0_1024_8000.0_3.0_1_cubic.fits' --data_fn '/pscratch/sd/a/arosado/lognormal_mocks/data_3.0_1250.0_1024_8000.0_3.0_1_cubic.fits' --out_dir '/pscratch/sd/a/arosado/lognormal_mocks/' --file_id 'cubic'

# python pypower_mocks_cubic.py --nmesh 512 --boxsize 8000 --randoms_fn '/pscratch/sd/a/arosado/lognormal_mocks/randoms_3.0_2500.0_1024_8000.0_3.00_1_full.fits' --data_fn '/pscratch/sd/a/arosado/lognormal_mocks/data_3.0_2500.0_1024_8000.0_3.00_1_full.fits' --out_dir '/pscratch/sd/a/arosado/lognormal_mocks/' --file_id 'cubic_test'
parser = argparse.ArgumentParser()
parser.add_argument('--nmesh',type=int,required=True,help='nmesh?')
parser.add_argument('--boxsize',type=int,required=True,help='boxsize?')
parser.add_argument('--randoms_fn',type=str,required=True, help='randoms filename?')
parser.add_argument('--data_fn',type=str,required=True, help='data filename?')
parser.add_argument('--out_dir',type=str,required=True, help='output directory for result?')
parser.add_argument('--file_id',type=str,required=True, help='how result fn ends power_{nmesh}_{file_id}.npy?')

opt = parser.parse_args()

# arguments
nmesh = opt.nmesh
boxsize = opt.boxsize
rands_fn = opt.randoms_fn
data_fn = opt.data_fn
out_dir = opt.out_dir
file_id = opt.file_id

use_mpi=True

if use_mpi:
    comm=MPI.COMM_WORLD
else:
    comm=MPI.COMM_SELF

# Read input catalogs, scattering on all MPI ranks
def read(fn,columns=('Position'),ext=1,mpicomm=comm):
    gsize=fitsio.FITS(fn)[ext].get_nrows()
    start,stop=mpicomm.rank*gsize // mpicomm.size,(mpicomm.rank+1)*gsize // mpicomm.size
    tmp=fitsio.read(fn,ext=ext,columns=columns,rows=range(start,stop))
    return tmp #[tmp[col] for col in columns]

# To activate logging
setup_logging()

edges={'min':0,'step':0.001}
ells=(0,2,4)

#randoms= fitsio.read(rands_fn)
#data = fitsio.read(data_fn) 

randoms_position = read(rands_fn)
data_position = read(data_fn) 

result = CatalogFFTPower(data_positions1=data_position, randoms_positions1=randoms_position,
                         edges=edges, ells=ells, interlacing=2, boxsize=boxsize, nmesh=nmesh, resampler='tsc',
                         los=None, position_type='pos',mpicomm=comm,wrap=True)

#result = CatalogFFTPower(data_positions1=data['Position'], randoms_positions1=randoms['Position'],
#                         edges=edges, ells=ells, interlacing=2, boxsize=boxsize, nmesh=nmesh, resampler='tsc',
#                         los=None, position_type='pos',mpicomm=comm,wrap=True)

result.save(os.path.join(out_dir, f"power_{nmesh}_{file_id}.npy"))