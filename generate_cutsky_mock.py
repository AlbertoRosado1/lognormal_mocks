# python generate_cutsky_mock.py --bias 3.0 --nbar 2500 --zmin 2.75 --zmax 3.25 --nmesh 1024 --boxsize 8000 --ramin 175 --ramax 290 --decmin -10 --decmax 70 --version test1 --id 1
import os
import fitsio
from astropy.table import Table
from time import time
import healpy as hp
import numpy as np
from mockfactory import (BoxCatalog, RandomBoxCatalog,
                         DistanceToRedshift, utils, setup_logging)
from cosmoprimo.fiducial import DESI
import useful_functions as ut
import argparse

parser = argparse.ArgumentParser(description='apply cutsky to lognormal mocks')
parser.add_argument('-b','--bias',type=float,required=True,help='bias?')
parser.add_argument('-nb','--nbar',type=float,required=True,help='nbar [deg^-2]')
parser.add_argument('--zmin',type=float,required=True,help='zmin?')
parser.add_argument('--zmax',type=float,required=True,help='zmax?')
parser.add_argument('-ze','--zeff',type=float,required=False,default=None,help='effective redshift')
parser.add_argument('-nm','--nmesh',type=int,required=True,help='nmesh?')
parser.add_argument('-bs','--boxsize',type=float,required=True,help='boxsize?')
parser.add_argument('--los',type=str,required=False,default=None, help='line of sight for RSD?')

# CutSky
parser.add_argument('--dmin',type=int,required=False,default=None,help='min distance?')
parser.add_argument('--dmax',type=int,required=False,default=None,help='mix distance?')

parser.add_argument('--ramin',type=int,required=True,help='RA min?')
parser.add_argument('--ramax',type=int,required=True,help='RA max?')

parser.add_argument('--decmin',type=int,required=True,help='DEC min?')
parser.add_argument('--decmax',type=int,required=True,help='DEC max?')

parser.add_argument('--id',type=int,required=True,help='seed id used for cubic mock?')

parser.add_argument('-v','--version',type=str,required=True, help='cubic mocks version?')
parser.add_argument('-cv','--cutsky_version',type=str,required=True, help='cutsky mock version?')

opt = parser.parse_args()

# Set up logging
setup_logging()
version = opt.version
cutsky_version = opt.cutsky_version
seed_id = opt.id
if os.getenv('PSCRATCH')==None:
    base_dir = os.path.join(os.getenv('CSCRATCH'), 'lognormal_mocks')
else:
    base_dir = os.path.join(os.getenv('PSCRATCH'), 'lognormal_mocks')
cubic_dir = os.path.join(base_dir, 'cubic', version)
cutsky_dir = os.path.join(base_dir, 'cutsky', cutsky_version)

if not os.path.isdir(cutsky_dir):
    os.makedirs(cutsky_dir)
    print(f"created {cutsky_dir}")
    
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

# start time 
t_i   = time()
#-----------------------------------------------------
# We need to provide back everything BoxCatalog needs
ph = str(seed_id).zfill(4)
file_id = f"{bias}_{nb_deg}_{nmesh}_{boxsize}_{zeff}_ph{ph}"

data_fn = os.path.join(cubic_dir, 'data_cubic_'+file_id+'.fits')
print(f"using {data_fn}")
data = BoxCatalog.read(data_fn, boxsize=boxsize, boxcenter=boxcenter, position='Position', velocity='Displacement')

#-----------------------------------------------------
# Let us cut the above box to some geometry
if (opt.dmin is None) and (opt.dmax is None):
    drange = [dmin,dmax]
else:
    drange = np.array([opt.dmin,opt.dmax])
rarange = [opt.ramin,opt.ramax]
decrange = [opt.decmin, opt.decmax]
print(f"generating parent cutsky with RA={rarange}, and DEC={decrange}")
data_cutsky = data.cutsky(drange=drange, rarange=rarange, decrange=decrange)

distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)
for catalog in [data_cutsky]:
    catalog['Distance'], catalog['RA'], catalog['DEC'] = utils.cartesian_to_sky(catalog.position)
    catalog['Z'] = distance_to_redshift(catalog['Distance'])
    
#-----------------------------------------------------
# Let us cut further (checkerboard case)
rarange_list = ut.split_range(opt.ramin, opt.ramax, nchunks=4,sep=10)
decrange_list = ut.split_range(opt.decmin, opt.decmax, nchunks=3,sep=10)
print(f"Choosing distance, RA, Dec ranges:")
print(f"RA ranges: {rarange_list}")
print(f"DEC ranges: {decrange_list}")

count = np.zeros(data_cutsky['RA'].size)
for i, rarange in enumerate(rarange_list):
    count += (rarange[0] <= data_cutsky['RA']) & (data_cutsky['RA'] <= rarange[1]) * 1 # multply by 1 gives integer array
for j, decrange in enumerate(decrange_list):
    count += (decrange[0] <= data_cutsky['DEC']) & (data_cutsky['DEC'] <= decrange[1]) * 1    
print(np.unique(count))
mask_checkerboard = count == count.max()
area_checkerboard = mask_checkerboard.sum()/nb_deg
print(mask_checkerboard.sum(),data_cutsky['RA'].size, mask_checkerboard.sum()/data_cutsky['RA'].size)

#-----------------------------------------------------
# Let us cut further (contiguous case)
t_ii = time()
mask_contiguous = ut.mask_to_match_skyarea(data_cutsky, nb_deg, area_checkerboard, field='DEC', step=0.01)
area_contiguous = mask_contiguous.sum()/nb_deg
print(area_checkerboard, area_contiguous)
print(f"determined optimal cut for contiguous case in {time()-t_ii:.2f} s")

#-----------------------------------------------------
# save cutsky options as masks that can be easily applied
data_cutsky['mask_contiguous'] = mask_contiguous
data_cutsky['mask_checkerboard'] = mask_checkerboard

file_id = f"{bias}_{nb_deg}_{nmesh}_{boxsize}_{zeff}_ph{ph}"
data_fn = os.path.join(cutsky_dir, 'data_cutsky_'+file_id+'.fits')
data_cutsky.write(data_fn)

print(f"finished in {time()-t_i:.2f} s")