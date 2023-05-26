# generetate lognormal mocks  NO CUTSKY yet
# to run 
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# salloc -N 1 -n 2 -C cpu -q regular -t 00:15:00 python cutsky_mock.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --ramin 0 --ramax 120 --decmin 10 --decmax 30
# salloc -N 1 -n 2 -C cpu -q debug -t 00:10:00 python cutsky_mock.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 128 --boxsize 1000 --ramin 0 --ramax 120 --decmin 10 --decmax 30

# python cutsky_mock.py --bias 3.0 --nbar 1250 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --ramin 175 --ramax 290 --decmin -10 --decmax 70 --id 1 --version 'test3'

import os
import numpy as np
from matplotlib import pyplot as plt
from mockfactory import (EulerianLinearMock, LagrangianLinearMock,
                         Catalog, BoxCatalog, RandomBoxCatalog, box_to_cutsky,
                         DistanceToRedshift, TabulatedRadialMask, HealpixAngularMask,
                         utils, setup_logging)
from cosmoprimo.fiducial import DESI
import argparse

parser = argparse.ArgumentParser(description='apply cutsky to lognormal mocks')
parser.add_argument('-b','--bias',type=float,required=True,help='bias?')
parser.add_argument('-nb','--nbar',type=float,required=True,help='nbar [deg^-2]')
parser.add_argument('--zmin',type=float,required=True,help='zmin?')
parser.add_argument('--zmax',type=float,required=True,help='zmax?')
parser.add_argument('-ze','--zeff',type=float,required=False,default=None,help='effective redshift')
parser.add_argument('-nm','--nmesh',type=int,required=True,help='nmesh?')
parser.add_argument('-bs','--boxsize',type=float,required=True,help='boxsize?')
parser.add_argument('--los',type=str,required=False,default='x', help='line of sight for RSD?')

# CutSky
parser.add_argument('--dmin',type=int,required=False,default=None,help='min distance?')
parser.add_argument('--dmax',type=int,required=False,default=None,help='mix distance?')

parser.add_argument('--ramin',type=int,required=True,help='RA min?')
parser.add_argument('--ramax',type=int,required=True,help='RA max?')

parser.add_argument('--decmin',type=int,required=True,help='DEC min?')
parser.add_argument('--decmax',type=int,required=True,help='DEC max?')

parser.add_argument('--id',type=int,required=True,help='just give it id to seperate from other cuts?')

parser.add_argument('-v','--version',type=str,required=True, help='version?')

opt = parser.parse_args()

#-----------------------------------------------------
# Set up logging
setup_logging()
version = opt.version
base_dir = os.path.join(os.getenv('PSCRATCH'), 'lognormal_mocks')
cutsky_dir = os.path.join(base_dir, 'cutsky', version)

# redshift parameters
zmin = opt.zmin # 2.75
zmax = opt.zmax # 3.25
if opt.zeff is None:
    zeff = (zmax + zmin)/2.
print(f"zmin={zmin}, zmax={zmax}, zeff={zeff}")

# change angular density to comoving density
nb_deg = opt.nbar #2500 # nbar in deg^-2 
nb = nb_deg

# Set other parameters
bias, nbar, nmesh, boxsize = opt.bias, nb, opt.nmesh, opt.boxsize # 3.0, nb, 256, 8000.
los = opt.los #'x' # line of sight

# Loading DESI fiducial cosmology
cosmo = DESI()
power = cosmo.get_fourier().pk_interpolator().to_1d(z=zeff)
dist = cosmo.comoving_radial_distance(zeff)
dmin = cosmo.comoving_radial_distance(zmin)
dmax = cosmo.comoving_radial_distance(zmax)
f = cosmo.sigma8_z(z=zeff,of='theta_cb')/cosmo.sigma8_z(z=zeff,of='delta_cb') # growth rate
boxcenter = [dist, 0, 0]

#-----------------------------------------------------
# We need to provide back everything BoxCatalog needs
file_id = f"{bias}_{nb_deg}_{nmesh}_{boxsize}_{zeff:.2f}_{opt.id}"

data_fn = os.path.join(base_dir, 'data_'+file_id+'_full.fits')
data = BoxCatalog.read(data_fn, boxsize=boxsize, boxcenter=boxcenter, position='Position', velocity='Displacement')

randoms_fn = os.path.join(base_dir, 'randoms_'+file_id+'_full.fits')
randoms = RandomBoxCatalog.read(randoms_fn, boxsize=boxsize, boxcenter=boxcenter, position='Position')

#-----------------------------------------------------
# Let us cut the above box to some geometry
#drange = [dist - size/3., dist + size/3.]
if (opt.dmin is None) and (opt.dmax is None):
    drange = [dmin,dmax] #[dist - (dist-dmin), dist + (dmax-dist)]
else:
    drange = np.array([opt.dmin,opt.dmax])
rarange = np.array([opt.ramin, opt.ramax])
decrange = np.array([opt.decmin, opt.decmax])
print('Choosing distance, RA, Dec ranges [{:.2f} - {:.2f}], [{:.2f} - {:.2f}], [{:.2f} - {:.2f}].'.format(*drange, *rarange, *decrange))
# noutput = None will cut as many catalogs as possible
data = data.cutsky(drange=drange, rarange=rarange, decrange=decrange)
randoms = randoms.cutsky(drange=drange, rarange=rarange, decrange=decrange)
print(randoms.size)

#-----------------------------------------------------
distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)
for catalog in [data, randoms]:
    catalog['Distance'], catalog['RA'], catalog['DEC'] = utils.cartesian_to_sky(catalog.position)
    catalog['Z'] = distance_to_redshift(catalog['Distance'])
    
#-----------------------------------------------------
# save catalogs to .fits
file_id = f"{bias}_{nb_deg}_{nmesh}_{boxsize}_{zeff}_{opt.id}"
fn = os.path.join(cutsky_dir, 'data_cutsky_'+file_id+'.fits')
data.write(fn)

fn = os.path.join(cutsky_dir, 'randoms_cutsky_'+file_id+'.fits')
randoms.write(fn)