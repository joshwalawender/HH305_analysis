#!/usr/env/python

## Import General Tools
import sys
import os
import logging

from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy import units as u

import numpy as np
from matplotlib import pyplot as plt

from astropy import visualization as v

h_beta_std = 4861.3615 * u.Angstrom
H_beta_std_velocity = u.doppler_optical(h_beta_std)

##-------------------------------------------------------------------------
## plot_image_and_lines
##-------------------------------------------------------------------------
def plot_image_and_lines(cube, wavs, xrange, yrange, Hbeta_ref=None,
                         title='', filename=None, include_OIII=False):

    zpix = np.arange(0,cube.shape[0])
    lambda_delta = 5
    hbeta_z = np.where((np.array(wavs) > h_beta_std.value-lambda_delta)\
                       & (np.array(wavs) < h_beta_std.value+lambda_delta))[0]
    image = np.mean(cube[min(hbeta_z):max(hbeta_z)+1, :, :], axis=0)

    spect = [np.mean(cube[z, yrange[0]:yrange[1]+1, xrange[0]:xrange[1]+1]) for z in zpix]
    i_peak = spect.index(max(spect))

    background_0 = models.Polynomial1D(degree=2)
    H_beta_0 = models.Gaussian1D(amplitude=500, mean=4861, stddev=1.,
                      bounds={'mean': (4855,4865), 'stddev': (0.1,5)})
    OIII4959_0 = models.Gaussian1D(amplitude=100, mean=4959, stddev=1.,
                        bounds={'mean': (4955,4965), 'stddev': (0.1,5)})
    OIII5007_0 = models.Gaussian1D(amplitude=200, mean=5007, stddev=1.,
                        bounds={'mean': (5002,5012), 'stddev': (0.1,5)})
    fitter = fitting.LevMarLSQFitter()
    if include_OIII is True:
        model0 = background_0 + H_beta_0 + OIII4959_0 + OIII5007_0
    else:
        model0 = background_0 + H_beta_0

    model0.mean_1 = wavs[i_peak]
    model = fitter(model0, wavs, spect)
    residuals = np.array(spect-model(wavs))


    plt.figure(figsize=(20,8))
    
    plt.subplot(1,4,1)
    plt.title(title)
    norm = v.ImageNormalize(image, interval=v.MinMaxInterval(), stretch=v.LogStretch(1))
    plt.imshow(image, origin='lower', norm=norm)
    region_x = [xrange[0]-0.5, xrange[1]+0.5, xrange[1]+0.5, xrange[0]-0.5, xrange[0]-0.5]
    region_y = [yrange[0]-0.5, yrange[0]-0.5, yrange[1]+0.5, yrange[1]+0.5, yrange[0]-0.5]
    plt.plot(region_x, region_y, 'r-', alpha=0.5, lw=2)

    plt.subplot(1,4,2)
    if Hbeta_ref is not None:
        Hbeta_velocity = (model.mean_1.value*u.Angstrom).to(u.km/u.s,
                           equivalencies=u.doppler_optical(Hbeta_ref*u.angstrom))
        title = f'H-beta ({model.mean_1.value:.1f} A, v={Hbeta_velocity.value:.1f} km/s)'
    else:
        title = f'H-beta ({model.mean_1.value:.1f} A, sigma={model.stddev_1.value:.3f} A)'
    plt.title(title)
    w = [l for l in np.arange(4856,4866,0.05)]
    if Hbeta_ref is not None:
        vs = [(l*u.Angstrom).to(u.km/u.s,
              equivalencies=u.doppler_optical(Hbeta_ref*u.angstrom)).value
              for l in wavs]
        plt.plot(vs, spect, drawstyle='steps-mid', label='data')
        vs = [(l*u.Angstrom).to(u.km/u.s,
              equivalencies=u.doppler_optical(Hbeta_ref*u.angstrom)).value
              for l in w]
        plt.plot(vs, model(w), 'r-', alpha=0.7, label='Fit')
        plt.xlabel('Velocity (km/s)')
        plt.xlim(-200,200)
    else:
        plt.plot(wavs, spect, drawstyle='steps-mid', label='data')
        plt.plot(w, model(w), 'r-', alpha=0.7, label='Fit')
        plt.xlabel('Wavelength (angstroms)')
        plt.xlim(4856,4866)
    plt.grid()
    plt.ylabel('Flux')
    plt.legend(loc='best')

    plt.subplot(1,4,3)
    if include_OIII is True:
        title = f'OIII 4959 ({model.mean_2.value:.1f} A, sigma={model.stddev_2.value:.3f} A)'
    else:
        title = f'OIII 4959'
    plt.title(title)
    plt.plot(wavs, spect, drawstyle='steps-mid', label='data')
    w = [l for l in np.arange(4954,4964,0.05)]
    plt.plot(w, model(w), 'r-', alpha=0.7, label='Fit')
    plt.xlabel('Wavelength (angstroms)')
    plt.ylabel('Flux')
    plt.legend(loc='best')
    plt.xlim(4954,4964)

    plt.subplot(1,4,4)
    if include_OIII is True:
        title = f'OIII 5007 ({model.mean_3.value:.1f} A, sigma={model.stddev_3.value:.3f} A)'
    else:
        title = f'OIII 5007'
    plt.title(title)
    plt.plot(wavs, spect, drawstyle='steps-mid', label='data')
    w = [l for l in np.arange(5002,5012,0.05)]
    plt.plot(w, model(w), 'r-', alpha=0.7, label='Fit')
    plt.xlabel('Wavelength (angstroms)')
    plt.ylabel('Flux')
    plt.legend(loc='best')
    plt.xlim(5002,5012)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.10)
    else:
        plt.show()

    return spect, model


##-------------------------------------------------------------------------
## HH305E
##-------------------------------------------------------------------------
data_path = os.path.expanduser('~/Sync/KeckData/KCWI/HH305')
hdul = fits.open(os.path.join(data_path, 'HH305E.fits'))
wcs = WCS(hdul[0].header)
zpix = np.arange(0,hdul[0].data.shape[0])
wavs = [wcs.all_pix2world(np.array([[0,0,z]]), 1)[0][2]*1e10 for z in zpix]
cube = hdul[0].data[:, 15:80, 5:29]

##-------------------------------------------------------------------------
# Select Nebular Background
xpix = 19
ypix = 55
xdelta = 4
ydelta = 8
yrange = [ypix-ydelta, ypix+ydelta]
xrange = [xpix-xdelta, xpix+xdelta]

spect, model = plot_image_and_lines(cube, wavs, xrange, yrange,
                             include_OIII=True,
                             title='Nebular Background',
                             filename='HH305E Nebular Background.png')
Hbeta_neb = model.mean_1.value

##-------------------------------------------------------------------------
## Subtract Nebular Background
neb_subtracted = cube.copy()
for z in zpix:
#     neb_subtracted[z,:,:] = neb_subtracted[z,:,:] - model(wavs[z])
    neb_subtracted[z,:,:] = neb_subtracted[z,:,:] - spect[z]

##-------------------------------------------------------------------------
## Shocks in HH305E
xpix = 4
ypix = 28
xdelta = 2
ydelta = 2
yrange = [ypix-ydelta, ypix+ydelta]
xrange = [xpix-xdelta, xpix+xdelta]

model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
                             include_OIII=False,
                             Hbeta_ref=Hbeta_neb,
                             title='Nebula Subtracted',
                             filename='HH305E Shock Region 1.png')

xpix = 8
ypix = 48
xdelta = 2
ydelta = 2
yrange = [ypix-ydelta, ypix+ydelta]
xrange = [xpix-xdelta, xpix+xdelta]

model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
                             include_OIII=False,
                             Hbeta_ref=Hbeta_neb,
                             title='Nebula Subtracted',
                             filename='HH305E Shock Region 2.png')

xpix = 7
ypix = 10
xdelta = 2
ydelta = 2
yrange = [ypix-ydelta, ypix+ydelta]
xrange = [xpix-xdelta, xpix+xdelta]

model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
                             include_OIII=False,
                             Hbeta_ref=Hbeta_neb,
                             title='Nebula Subtracted',
                             filename='HH305E Shock Region 3.png')

xpix = 16
ypix = 15
xdelta = 2
ydelta = 4
yrange = [ypix-ydelta, ypix+ydelta]
xrange = [xpix-xdelta, xpix+xdelta]

model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
                             include_OIII=False,
                             Hbeta_ref=Hbeta_neb,
                             title='Nebula Subtracted',
                             filename='HH305E Shock Region 4.png')

