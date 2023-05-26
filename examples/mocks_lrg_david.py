# Here is the code to generate lognormal mocks. Two things: 1) i added a piece of code if you want to get the mean density from the data nz file instead of just giving a number. If you want to use mpi make sure to load the file on mpiroot otherwise you'll get weird results (i can add it). 2) i was using a tile mask for desi but i don't think desimodel is included in Arnaud's cosmodesi environment (i only tried main) so you may want to use your own environment for this

import os
import numpy as np
from matplotlib import pyplot as plt
from mockfactory import (EulerianLinearMock, LagrangianLinearMock,
                         Catalog, BoxCatalog, RandomBoxCatalog, box_to_cutsky,
                         DistanceToRedshift, TabulatedRadialMask, HealpixAngularMask,
                         utils, setup_logging)
from cosmoprimo.fiducial import DESI
import fitsio
import desimodel.footprint
from astropy.table import Table, vstack

#-----------------------------------------------------
# Set up logging
setup_logging()
zmin = 0.8
zmax = 1.1
zeff = (zmax + zmin)/2.

# to get tracer mean density from data
ttype1 = 'LRG'
filename = '/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/'+ttype1+'_main_N_nz.txt' # For LRG only
# filename = '/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/'+ttype1+'_N_nz.txt'
ff = np.loadtxt(filename)
zfile = ff[:,0]
nz = ff[:,3]
zcut = np.where((zfile>=zmin)&(zfile<=zmax))[0]
zslice = zfile[zcut]
nbarz = nz[zcut]
nb = np.mean(nbarz)
base_dir = "/pscratch/sd/a/arosado/mocks_tests/" #'../'
# base_dir = '/folder/of/your/choice/'  


#define some parameters
bias, nbar, nmesh, boxsize,los = 2.3, nb, 1024, 4500,'x' #adapt for your case

#-----------------------------------------------------
#-----------------------------------------------------
for i in range(1,2):
    print(i)
    # Loading DESI fiducial cosmology
    cosmo = DESI()
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=zeff)
    dist = cosmo.comoving_radial_distance(zeff)
    dmin = cosmo.comoving_radial_distance(zmin)
    dmax = cosmo.comoving_radial_distance(zmax)
    f = cosmo.sigma8_z(z=zeff,of='theta_cb')/cosmo.sigma8_z(z=zeff,of='delta_cb') # growth rate
    boxcenter = [dist, 0, 0]
    mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=None, unitary_amplitude=False)
    # this is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias-1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=None)
    mock.set_rsd(f=f, los=los)
    data = mock.to_catalog()
    # We've got data, now turn to randoms
    from mockfactory.make_survey import RandomBoxCatalog
    randoms = RandomBoxCatalog(nbar=4.*nbar, boxsize=boxsize, seed=None) # i put 4 as an arbitrary factor for the example

    #-----------------------------------------------------
    # Let us cut the above box to some geometry
    #drange = [dist - size/3., dist + size/3.]
    drange = [dist - (dist-dmin), dist + (dmax-dist)]
    rarange = [175, 290]
    decrange = [-10, 70]
    # noutput = None will cut as many catalogs as possible
    data = data.cutsky(drange=drange, rarange=rarange, decrange=decrange)
    randoms = randoms.cutsky(drange=drange, rarange=rarange, decrange=decrange)

    #-----------------------------------------------------
    # from mockfactory.make_survey import DistanceToRedshift
    # distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)
    # For data, we want to apply RSD *before* selection function
    # isometry, mask_radial, mask_angular = data.isometry_for_cutsky(drange=drange, rarange=rarange, decrange=decrange)
    # First move data to its final position
    # data_cutsky = data.cutsky_from_isometry(isometry, dradec=None)
    # Apply RSD
    # data_cutsky['RSDPosition'] = data_cutsky.rsd_position(f=f)
    # data_cutsky['Distance'], data_cutsky['RA'], data_cutsky['DEC'] = utils.cartesian_to_sky(data_cutsky['RSDPosition'])
    # data_cutsky['Z'] = distance_to_redshift(data_cutsky['Distance'])
    # Apply selection function
    #mask = mask_radial(data['Distance']) & mask_angular(data['RA'], data['DEC'])
    #data = data[mask]
    from mockfactory.make_survey import DistanceToRedshift
    distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)
    for catalog in [data, randoms]:
        catalog['Distance'], catalog['RA'], catalog['DEC'] = utils.cartesian_to_sky(catalog.position)
        catalog['Z'] = distance_to_redshift(catalog['Distance'])                               

    #------------------------------------------------------
    # Let us apply some redshift cuts
    from mockfactory.make_survey import TabulatedRadialMask
    mask_radial = TabulatedRadialMask(z=zslice, nbar=nbarz)
    # mask_radial = TabulatedRadialMask(filename=filename,zrange=(zmin,zmax))


    ### FOR TILE MASK
    #fulldat = fitsio.read('./DESI_data/QSOtargetsDR9v1.1.1.fits') #testing on quasar targets
    tls = fitsio.read('/global/cfs/cdirs/desi/survey/catalogs/DA02/LSS/tiles-DARK.fits') # change according to tracer choice
    # tls = fitsio.read('/global/cfs/cdirs/desi/survey/catalogs/DA02/LSS/tiles-BRIGHT.fits')
    ind_dat = desimodel.footprint.is_point_in_desi(tls,data['RA'], data['DEC'])
    data = data[ind_dat]
    ind_rdm = desimodel.footprint.is_point_in_desi(tls,randoms['RA'], randoms['DEC'])
    randoms = randoms[ind_rdm]


    #------------------------------------------------------                               
    #get data to subsample to match unweighted da0.2
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # if rank == 0:
    # else:    
    #     data_cat_z = None
    # data_cat_z = comm.bcast(data_cat_z, root=0)
    #------------------------------------------------------                               
    #save in fits format
    # if rank == 0:
    # fits_name_dat = './DESI_data/E_da0.2/LRGzdone_clustering.dat.fits'
    # data_cat = Table.read(fits_name_dat, format='fits')
    # valid_d = (data_cat['Z'] > zmin)&(data_cat['Z'] < zmax)
    # data_cat_z = data_cat[valid_d]


    fn = os.path.join(base_dir, 'data_'+str(bias)+'_'+str(nmesh)+'_'+str(i)+'.fits')
    mask = mask_radial(data['Z'], seed=None)
    data[mask].write(fn)
    # print(len(data['Z'][mask]))
    # print(len(data_cat_z['Z']))
    # pratio = len(data[mask]['Position'][:,0])/len(data_cat_z['Z'])
    # subsamp_dat = np.random.choice(len(data[mask]['Position'][:,0]), int(round(len(data[mask]['Position'][:,0])/pratio)), replace=False)
    # data[mask][subsamp_dat].save_fits(fn)

    fn = os.path.join(base_dir, 'randoms_'+str(bias)+'_'+str(nmesh)+'_'+str(i)+'.fits')
    mask = mask_radial(randoms['Z'], seed=None)
    randoms[mask].write(fn)
    # subsamp_rdm = np.random.choice(len(randoms[mask]['Position'][:,0]), int(round(len(randoms[mask]['Position'][:,0])/pratio)), replace=False)
    # randoms[mask][subsamp_rdm].save_fits(fn)