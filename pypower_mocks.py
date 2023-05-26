import numpy as np
import os
import argparse
from pypower import CatalogFFTPower, setup_logging
from mpi4py import MPI
import fitsio
# python pypower_mocks.py --nmesh 512 --boxsize 8000 --randoms_fn '/pscratch/sd/a/arosado/lognormal_mocks/cutsky/test1/randoms_cutsky_3.0_1250.0_1024_8000.0_3.0_concat.fits' --data_fn '/pscratch/sd/a/arosado/lognormal_mocks/cutsky/test1/data_cutsky_3.0_1250.0_1024_8000.0_3.0_concat.fits' --out_dir '/pscratch/sd/a/arosado/lognormal_mocks/cutsky/test1/' --file_id 'concat'

# python pypower_mocks.py --nmesh 512 --boxsize 8000 --randoms_fn '/pscratch/sd/a/arosado/lognormal_mocks/cutsky/test1/randoms_cutsky_3.0_1250.0_1024_8000.0_3.0.fits' --data_fn '/pscratch/sd/a/arosado/lognormal_mocks/cutsky/test1/data_cutsky_3.0_1250.0_1024_8000.0_3.0.fits' --out_dir '/pscratch/sd/a/arosado/lognormal_mocks/cutsky/test1/' --file_id 'contiguous'
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
def read(fn,columns=('RA','DEC','Distance'),ext=1,mpicomm=comm):
    gsize=fitsio.FITS(fn)[ext].get_nrows()
    start,stop=mpicomm.rank*gsize // mpicomm.size,(mpicomm.rank+1)*gsize // mpicomm.size
    tmp=fitsio.read(fn,ext=ext,columns=columns,rows=range(start,stop))
    return [tmp[col] for col in columns]

# To activate logging
setup_logging()

edges={'min':0,'step':0.001}
ells=(0,2,4)

ra_rand,dec_rand,dist_rand=read(rands_fn)
random_positions = [ra_rand,dec_rand,dist_rand]
#randoms_weights = np.ones(ra_rand.size)

ra,dec,dist=read(data_fn)
data_positions = [ra,dec,dist]
#data_weights = np.ones(ra.size)


result = CatalogFFTPower(data_positions1=data_positions, randoms_positions1=random_positions,
                         edges=edges, ells=ells, boxsize=boxsize, nmesh=nmesh, resampler='tsc', 
                         interlacing=2, los=None, position_type='rdd',mpicomm=comm)

result.save(os.path.join(out_dir, f"power_{nmesh}_{file_id}.npy"))