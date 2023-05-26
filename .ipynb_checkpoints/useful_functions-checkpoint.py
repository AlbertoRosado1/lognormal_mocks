import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from pypower import CatalogFFTPower

# define some useful functions
def plot_in_box(data, elev=None, azim=None, roll=None, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*data['Position'][::100].T, marker='.', alpha=0.2, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev, azim, roll)
    
def plot_mollview(data,nside, **kwargs):
    data_hpmap = hpixsum(nside,data['RA'],data['DEC'])
    hp.mollview(data_hpmap, **kwargs)
    
def hpixsum(nside, ra, dec, weights=None):
    hpix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    npix = hp.nside2npix(nside)
    weight_hp = np.bincount(hpix, weights=weights, minlength=npix)
    return weight_hp

def split_range(start,end,nchunks=1,sep=0):
    # split list [start,end] into nchunks lists
    chunk = ((abs(end - start) - (nchunks-1)*sep)/ (nchunks))
    ranges = []
    for i in range(0,nchunks):
        ind1 = start + chunk*i + sep*i
        ind2 = start + chunk*(i+1) + sep*i
        l = [ind1, ind2]
        ranges.append(l)
    return ranges

def objects_to_remove(N, nbar, skyarea):
    A = skyarea # desired sky area
    N = int(N) # number of objects
    nbar = nbar
    c =  N - nbar* A  # number of objects that have to be removed for catalog to have desired nbar within certain sky area
    print(f"number of objects to be removed {c} ({c/N*100:.2f}%)")
    return c

def mask_to_match_skyarea(data, nbar, skyarea, field='RA', step=0.1):
    l = []
    c = objects_to_remove(data[field].size, nbar, skyarea)
    field_range = np.arange(data[field].min(),data[field].max()+1,step)
    for i, r in enumerate(field_range):
        #print(f"looping {i}/{field_range.size}", end="\r")
        mask = data[field] > r
        l.append(mask.sum() / c)
    ones = np.ones_like(field_range)
    idx = np.argwhere(np.diff(np.sign(l - ones))).flatten()
    print(f"optimal {field} to perform cut: {field_range[idx][0]:.2f}")
    return data[field] < field_range[idx][0]

def get_mean_pk(pk_fns, ell=0, remove_shotnoise=False, rebin=None, return_shotnoise=False):
    pks = []
    snoise = []
    for fn in pk_fns:
        result = CatalogFFTPower.load(fn)
        poles= result.poles
        if rebin is not None: poles = poles[::rebin]
        k,pk = poles(ell=ell, return_k=True, complex=False, remove_shotnoise = remove_shotnoise)
        pks.append(pk)
        snoise.append(poles.shotnoise)
    print('Shot noise is {:.4f}.'.format(poles.shotnoise))
    pk_mean = np.mean(pks, axis=0)
    pk_sigma = np.std(pks, axis=0)
    snoise_mean = np.mean(snoise, axis=0)
    
    if return_shotnoise:
        return k, pk_mean, pk_sigma, snoise_mean
    else:
        return k, pk_mean, pk_sigma