# generetate lognormal mocks  NO CUTSKY yet
# to run 
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# srun -n 64 python generate_mocks_mpi.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --version test1 --id $seed_id
import os
import numpy as np
from mockfactory import (EulerianLinearMock, LagrangianLinearMock,
                         Catalog, BoxCatalog, RandomBoxCatalog, box_to_cutsky,
                         DistanceToRedshift, TabulatedRadialMask, HealpixAngularMask,
                         utils, setup_logging)
from cosmoprimo.fiducial import DESI

# Set up logging
setup_logging()

def nbar_angular2comoving(nbar,zmin,zmax):
    # change angular density to comoving density
    cosmo = DESI()
    V23 = 4*np.pi/3 * (cosmo.comoving_radial_distance(zmax)**3 - cosmo.comoving_radial_distance(zmin)**3)
    print(f"comoving volume between {zmin} and {zmax}: {V23:.2e} Mpc3 / h3")

    # create a conversion factor so we can change nbar from deg^-2 to h^3/Mpc^3  
    Asky = 4*np.pi*(180/np.pi)**2 #41252.96124941928#41253# square deg
    conv = V23 /Asky # Mpc^3 per square deg # Mpc^3/deg^2
    print(f"conversion factor {conv:.2e}")
    
    nbar_deg = nbar # deg^-2
    nb = nbar_deg / conv
    print(f"converted nbar={nbar_deg} 1 / deg2 to nbar={nb:.2e} h3 / Mpc3")
    return nb

#-----------------------------------------------------
from mpi4py import MPI

use_mpi=True

if use_mpi:
    comm = MPI.COMM_WORLD
else:
    comm=MPI.COMM_SELF
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    from time import time
    import argparse
    parser = argparse.ArgumentParser(description='generate lognormal mocks')
    parser.add_argument('-b','--bias',type=float,required=True,help='bias?')
    parser.add_argument('-nb','--nbar',type=float,required=True,help='nbar [deg^-2]')
    parser.add_argument('--zmin',type=float,required=True,help='zmin?')
    parser.add_argument('--zmax',type=float,required=True,help='zmax?')
    parser.add_argument('-ze','--zeff',type=float,required=False,default=None,help='effective redshift')
    parser.add_argument('-nm','--nmesh',type=int,required=True,help='nmesh?')
    parser.add_argument('-bs','--boxsize',type=float,required=True,help='boxsize?')
    parser.add_argument('--los',type=str,required=False,default=None, help='line of sight for RSD?')
    parser.add_argument('-v','--version',type=str,required=True, help='version?')

    parser.add_argument('--id',type=int,required=True,help='id given to seed?')

    opt = parser.parse_args()
    
    #-----------------------------------------------------

    version = opt.version
    seed_id = opt.id
    if os.getenv('PSCRATCH')==None:
        base_dir = os.path.join(os.getenv('CSCRATCH'), 'lognormal_mocks')
    else:
        base_dir = os.path.join(os.getenv('PSCRATCH'), 'lognormal_mocks')
    cubic_dir = os.path.join(base_dir, 'cubic', version)
    if rank == 0:
        if not os.path.isdir(cubic_dir):
            os.makedirs(cubic_dir)

    # redshift parameters
    zmin = opt.zmin # 2.75
    zmax = opt.zmax # 3.25
    if opt.zeff is None:
        zeff = (zmax + zmin)/2.
    print(f"zmin={zmin}, zmax={zmax}, zeff={zeff}")

    # change angular density to comoving density
    nb_deg = opt.nbar #2500 # nbar in deg^-2 
    nb = nbar_angular2comoving(nb_deg,zmin,zmax) # give function nbar in deg^-2 

    # Set other parameters
    bias, nbar, nmesh, boxsize = opt.bias, nb, opt.nmesh, opt.boxsize # 3.0, nb, 256, 8000.
    los = opt.los #'x' # line of sight
    
    file_id = f"{bias}_{nb_deg}_{nmesh}_{boxsize}_{zeff}_ph{str(seed_id).zfill(4)}"
    data_fn = os.path.join(cubic_dir, 'data_cubic_'+file_id+'.fits')
    
    t_i   = time()
else:
    bias = None
    nbar = None
    nmesh = None
    boxsize = None
    los = None # line of sight
    data_fn = None
    zeff = None
    seed_id = None
    
#-----------------------------------------------------
# bcast
bias = comm.bcast(bias, root=0)            # bias
nbar = comm.bcast(nbar, root=0)            # nbar
nmesh = comm.bcast(nmesh, root=0)          # number of mesh
boxsize = comm.bcast(boxsize, root=0)      # boxsize in Mpc/h
los = comm.bcast(los, root=0)              # line of sight
data_fn = comm.bcast(data_fn, root=0)      # path for writing the output
zeff = comm.bcast(zeff, root=0)            # effective redshift
seed_id = comm.bcast(seed_id, root=0)      # seed for random sampling

#-----------------------------------------------------
# Loading DESI fiducial cosmology
cosmo = DESI()
power = cosmo.get_fourier().pk_interpolator().to_1d(z=zeff)
dist = cosmo.comoving_radial_distance(zeff)
#dmin = cosmo.comoving_radial_distance(zmin)
#dmax = cosmo.comoving_radial_distance(zmax)
f = cosmo.sigma8_z(z=zeff,of='theta_cb')/cosmo.sigma8_z(z=zeff,of='delta_cb') # growth rate
boxcenter = [dist, 0, 0]
mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=1000+seed_id, unitary_amplitude=False)
# this is Lagrangian bias, Eulerian bias - 1
mock.set_real_delta_field(bias=bias-1)
mock.set_analytic_selection_function(nbar=nbar)
mock.poisson_sample(seed=3000+seed_id)
mock.set_rsd(f=f, los=los)
data = mock.to_catalog()
# We've got data, now turn to randoms
#from mockfactory.make_survey import RandomBoxCatalog
#randoms = RandomBoxCatalog(nbar=4.*nbar, boxsize=boxsize, boxcenter=boxcenter, seed=5000+seed_id) # i put 4 as an arbitrary factor for the example

#-----------------------------------------------------
# save catalogs to .fits

data.write(data_fn)

#randoms_fn = os.path.join(cubic_dir, 'randoms_'+file_id+'_cubic.fits')
#randoms.write(randoms_fn)

#comm.Barrier()

if rank==0:
    with open(os.path.join(cubic_dir,'details.txt'), 'a') as f:
        f.write(f"{data_fn}\n")
        f.write(f"#objects = {len(data)}\n")
        f.write(f"zmin={zmin}, zmax={zmax}, zeff={zeff}\n")
        f.write(f"bias={bias}\n nbar={nb_deg}deg^-2 \n nmesh={nmesh}\n boxsize={boxsize}\n seed_id={seed_id}\n")
        f.write(f"los={los}\n")
        f.write(f'generated in {time()-t_i} s\n')
        #f.write(f"\n")