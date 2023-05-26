import numpy as np
import os
import argparse
from pypower import CatalogFFTPower, setup_logging
from mpi4py import MPI
import fitsio
# srun -n 128 python pypower_cubic_mocks.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1 --nmesh 1024 --nrandoms 20 --case contiguous --id $SLURM_ARRAY_TASK_ID --use_single_mock
# srun -n 128 python pypower_cubic_mocks.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1 --nmesh 1024 --nrandoms 20 --case checkerboard --id $SLURM_ARRAY_TASK_ID --use_single_mock

parser = argparse.ArgumentParser()
parser.add_argument('-b','--bias',type=float,required=True,help='bias used for mocks?')
parser.add_argument('-nb','--nbar',type=float,required=True,help='nbar [deg^-2] used for mocks')
parser.add_argument('-ze','--zeff',type=float,required=True,help='effective redshift used for mocks')
parser.add_argument('--mock_nmesh',type=int,required=True,help='nmesh used in mocks?')
parser.add_argument('-nm','--nmesh',type=int,required=True,help='nmesh for Pk calculation?')
parser.add_argument('--dk',type=float,default=0.005,help='kbin width?')
parser.add_argument('-bs','--boxsize',type=float,required=True,help='boxsize?')
parser.add_argument('--nrandoms',type=int,default=20,help='numbers of randoms to use?')
parser.add_argument('-v','--version',type=str,required=True, help='version?')
parser.add_argument("--phstart", type=int,help="mock realization to start from.",default=1)
parser.add_argument("--phend", type=int,help="mock realization to finish on.",default=10)

# options useful for running on a specific mock
parser.add_argument('--use_single_mock',action='store_true',help='use this for calculating pk on a singe mock?')
parser.add_argument('--id',type=int,help='id given to seed when generating mock?',default=1)

# options useful for running pk on contiguous or checkerboard case
parser.add_argument('--case',type=str,choices=['contiguous','checkerboard'],required=True, help='use for calculating pk for contiguous or checkerboard case?')

opt = parser.parse_args()

# arguments
use_single_mock = opt.use_single_mock
mock_id = opt.id
version = opt.version
nmesh = opt.nmesh
nrandoms = opt.nrandoms

# mock specific arguments
bias = opt.bias
nbar_deg = opt.nbar
zeff = opt.zeff
boxsize = opt.boxsize
mock_nmesh = opt.mock_nmesh
phstart = opt.phstart
phend = opt.phend

mask_column = 'mask'+'_'+opt.case
print(f"using mask={mask_column}")

use_mpi=True

if use_mpi:
    comm=MPI.COMM_WORLD
else:
    comm=MPI.COMM_SELF
size = comm.Get_size()
rank = comm.Get_rank()

if os.getenv('PSCRATCH')==None:
    base_dir = os.path.join(os.getenv('CSCRATCH'), 'lognormal_mocks')
else:
    base_dir = os.path.join(os.getenv('PSCRATCH'), 'lognormal_mocks')
cutsky_dir = os.path.join(base_dir, 'cutsky', version)
pypower_outdir = os.path.join(base_dir, 'results_pypower', version)
if rank == 0:
    if not os.path.isdir(pypower_outdir):
        os.makedirs(pypower_outdir)


# Read input catalogs, scattering on all MPI ranks
def read(fn,columns=('RA','DEC','Distance', mask_column),ext=1,mpicomm=comm):
    gsize=fitsio.FITS(fn)[ext].get_nrows()
    start,stop=mpicomm.rank*gsize // mpicomm.size,(mpicomm.rank+1)*gsize // mpicomm.size
    tmp=fitsio.read(fn,ext=ext,columns=columns,rows=range(start,stop))
    return [tmp[col] for col in columns]

# To activate logging
setup_logging()

edges={'min':0,'step':opt.dk}
ells=(0,2,4)

# read and concatenate randoms
for N in range(1,nrandoms+1):
    file_id = f"{nbar_deg}_{boxsize}_{zeff}_ph{str(N).zfill(2)}"
    randoms_fn = os.path.join(cutsky_dir, 'randoms_cutsky_'+file_id+'.fits')
    if rank==0: print(f"loading randoms {N}/{nrandoms} from {randoms_fn}")
    if N==1:
        ra_rand, dec_rand, dist_rand, mask_rand = read(randoms_fn)
    else:
        RA_RAND, DEC_RAND, DIST_RAND, MASK_RAND = read(randoms_fn)
        ra_rand   = np.concatenate([ra_rand, RA_RAND])
        dec_rand  = np.concatenate([dec_rand, DEC_RAND])
        dist_rand = np.concatenate([dist_rand, DIST_RAND])
        mask_rand = np.concatenate([mask_rand, MASK_RAND])
    if rank==0: print(f"total randoms loaded = {len(ra_rand)}")
    
randoms_position = [ra_rand[mask_rand],dec_rand[mask_rand], dist_rand[mask_rand]]

# read data
if not use_single_mock:
    for mock_id in np.arange(phstart,phend+1):
        if rank==0: print(f"calculating Pk for {mock_id}/{phend} mocks")
        ph = str(mock_id).zfill(4)
        result_fn = os.path.join(pypower_outdir, f"power_cutsky_{nrandoms}_{nmesh}_dk{opt.dk}_{ph}.npy")
        if not os.path.exists(result_fn):
            file_id = f"{bias}_{nbar_deg}_{mock_nmesh}_{boxsize}_{zeff}_ph{ph}"
            data_fn = os.path.join(cutsky_dir, 'data_cutsky_'+file_id+'.fits')
            if rank==0: print(f"using mocks from {data_fn}")
            
            ra, dec, dist, mask = read(data_fn) 
            data_position = [ra[mask], dec[mask], dist[mask]]

            result = CatalogFFTPower(data_positions1=data_position, randoms_positions1=randoms_position,
                                     edges=edges, ells=ells, interlacing=2, boxsize=boxsize, nmesh=nmesh, 
                                     resampler='tsc', los=None, position_type='rdd',mpicomm=comm)#,wrap=True)

            result.save(result_fn)

            comm.Barrier() # not sure if this is neccessary, this waits for current task to finish before moving on? 
else: # run for a single mock with ph = str(mock_id).zfill(4)
    ph = str(mock_id).zfill(4)
    file_id = f"{bias}_{nbar_deg}_{mock_nmesh}_{boxsize}_{zeff}_ph{ph}"
    data_fn = os.path.join(cutsky_dir, 'data_cutsky_'+file_id+'.fits')
    result_fn = os.path.join(pypower_outdir, f"power_cutsky_{opt.case}_{nrandoms}_{nmesh}_dk{opt.dk}_{ph}.npy")
    if not os.path.exists(result_fn):
        if rank==0: print(f"calculating Pk for {data_fn}")

        ra, dec, dist, mask = read(data_fn) 
        data_position = [ra[mask], dec[mask], dist[mask]]

        result = CatalogFFTPower(data_positions1=data_position, randoms_positions1=randoms_position,
                                 edges=edges, ells=ells, interlacing=2, boxsize=boxsize, nmesh=nmesh, 
                                 resampler='tsc', los=None, position_type='rdd',mpicomm=comm)#,wrap=True)

        result.save(result_fn)
        
        comm.Barrier()
    else:
        if rank==0: print(f"already calculated Pk for {data_fn}, look at {result_fn}")
    