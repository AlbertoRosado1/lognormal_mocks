import numpy as np
import os
import argparse
from pypower import CatalogFFTPower, setup_logging
from mpi4py import MPI
import fitsio
# python pypower_mocks_cubic.py --bias 3.0 --nbar 2500 --zeff 3.0 --mock_nmesh 1024 --boxsize 8000 --version test1 --nmesh 1024 --nrandoms 5 --phstart 1 --phend 10

parser = argparse.ArgumentParser()
parser.add_argument('-b','--bias',type=float,required=True,help='bias used for mocks?')
parser.add_argument('-nb','--nbar',type=float,required=True,help='nbar [deg^-2] used for mocks')
parser.add_argument('-ze','--zeff',type=float,required=True,help='effective redshift used for mocks')
parser.add_argument('--mock_nmesh',type=int,required=True,help='nmesh used in mocks?')
parser.add_argument('-nm','--nmesh',type=int,required=True,help='nmesh for Pk calculation?')
parser.add_argument('-bs','--boxsize',type=float,required=True,help='boxsize?')
parser.add_argument('--nrandoms',type=int,default=20,help='numbers of randoms to use?')
parser.add_argument('-v','--version',type=str,required=True, help='version?')
parser.add_argument("--phstart", type=int,help="mock realization to start from.",default=1)
parser.add_argument("--phend", type=int,help="mock realization to finish on.",default=10)

# options useful for running on a specific mock
parser.add_argument('--use_single_mock',action='store_true',help='use this for calculating pk on a singe mock?')
parser.add_argument('--id',type=int,help='id given to seed when generating mock?',default=1)

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
cubic_dir = os.path.join(base_dir, 'cubic', version)
pypower_outdir = os.path.join(base_dir, 'results_pypower', version)
if rank == 0:
    if not os.path.isdir(pypower_outdir):
        os.makedirs(pypower_outdir)


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

# read and concatenate randoms
for N in range(1,nrandoms+1):
    file_id = f"{nbar_deg}_{boxsize}_{zeff}_ph{str(N).zfill(2)}"
    randoms_fn = os.path.join(cubic_dir, 'randoms_cubic_'+file_id+'.fits')
    if rank==0: print(f"loading randoms {N}/{nrandoms} from {randoms_fn}")
    if N==1:
        randoms_position = read(randoms_fn)
    else:
        RANDOMS_POSITION = read(randoms_fn)
        randoms_position = np.concatenate([randoms_position,RANDOMS_POSITION])
    if rank==0: print(f"total randoms loaded = {len(randoms_position)}")
    

# read data
if not use_single_mock:
    for mock_id in np.arange(phstart,phend+1):
        if rank==0: print(f"calculating Pk for {mock_id}/{phend} mocks")
        ph = str(mock_id).zfill(4)
        result_fn = os.path.join(pypower_outdir, f"power_cubic_{nrandoms}_{nmesh}_{ph}.npy")
        if not os.path.exists(result_fn):
            file_id = f"{bias}_{nbar_deg}_{mock_nmesh}_{boxsize}_{zeff}_ph{ph}"
            data_fn = os.path.join(cubic_dir, 'data_cubic_'+file_id+'.fits')
            if rank==0: print(f"using mocks from {data_fn}")

            data_position = read(data_fn) 

            result = CatalogFFTPower(data_positions1=data_position, randoms_positions1=randoms_position,
                                     edges=edges, ells=ells, interlacing=2, boxsize=boxsize, nmesh=nmesh, 
                                     resampler='tsc', los=None, position_type='pos',mpicomm=comm,wrap=True)

            result.save(result_fn)

            comm.Barrier() # not sure if this is neccessary, this waits for current task to finish before moving on? 
else: # run for a single mock with ph = str(mock_id).zfill(4)
    ph = str(mock_id).zfill(4)
    file_id = f"{bias}_{nbar_deg}_{mock_nmesh}_{boxsize}_{zeff}_ph{ph}"
    data_fn = os.path.join(cubic_dir, 'data_cubic_'+file_id+'.fits')
    result_fn = os.path.join(pypower_outdir, f"power_cubic_{nrandoms}_{nmesh}_{ph}.npy")
    if not os.path.exists(result_fn):
        if rank==0: print(f"calculating Pk for {data_fn}")

        data_position = read(data_fn) 

        result = CatalogFFTPower(data_positions1=data_position, randoms_positions1=randoms_position,
                                 edges=edges, ells=ells, interlacing=2, boxsize=boxsize, nmesh=nmesh, 
                                 resampler='tsc', los=None, position_type='pos',mpicomm=comm,wrap=True)

        result.save(result_fn)
    else:
        if rank==0: print(f"already calculated Pk for {data_fn}, look at {result_fn}")