#!/usr/env/python

## Import General Tools
import sys
import os
import logging
from datetime import datetime as dt

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
## Create logger object
##-------------------------------------------------------------------------
log = logging.getLogger('MyLogger')
log.setLevel(logging.DEBUG)
## Set up console output
LogConsoleHandler = logging.StreamHandler()
LogConsoleHandler.setLevel(logging.DEBUG)
LogFormat = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
LogConsoleHandler.setFormatter(LogFormat)
log.addHandler(LogConsoleHandler)


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

    spect = [np.mean(cube[z, yrange[0]:yrange[1]+1, xrange[0]:xrange[1]+1])
             for z in zpix]
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
    norm = v.ImageNormalize(image, interval=v.MinMaxInterval(),
                            stretch=v.LogStretch(1))
    plt.imshow(image, origin='lower', norm=norm)
    region_x = [xrange[0]-0.5, xrange[1]+0.5, xrange[1]+0.5, xrange[0]-0.5, xrange[0]-0.5]
    region_y = [yrange[0]-0.5, yrange[0]-0.5, yrange[1]+0.5, yrange[1]+0.5, yrange[0]-0.5]
    plt.plot(region_x, region_y, 'r-', alpha=0.5, lw=2)

    plt.subplot(1,4,2)
    if Hbeta_ref is not None:
        Hbeta_velocity = (model.mean_1.value*u.Angstrom).to(u.km/u.s,
                           equivalencies=u.doppler_optical(Hbeta_ref))
        title = f'H-beta ({model.mean_1.value:.1f} A, v={Hbeta_velocity.value:.1f} km/s)'
    else:
        title = f'H-beta ({model.mean_1.value:.1f} A, sigma={model.stddev_1.value:.3f} A)'
    plt.title(title)
    w = [l for l in np.arange(4856,4866,0.05)]
    if Hbeta_ref is not None:
        vs = [(l*u.Angstrom).to(u.km/u.s,
              equivalencies=u.doppler_optical(Hbeta_ref)).value
              for l in wavs]
        plt.plot(vs, spect, drawstyle='steps-mid', label='data')
        vs = [(l*u.Angstrom).to(u.km/u.s,
              equivalencies=u.doppler_optical(Hbeta_ref)).value
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
fits_file = os.path.join(data_path, 'HH305E.fits')
log.info(f'Loading {fits_file}')
hdul = fits.open(fits_file)
wcs = WCS(hdul[0].header)
zpix = np.arange(0,hdul[0].data.shape[0])
wavs = [wcs.all_pix2world(np.array([[0,0,z]]), 1)[0][2]*1e10 for z in zpix]
xtrim = [5, 29]
ytrim = [15, 80]
cube = hdul[0].data[:, ytrim[0]:ytrim[1], xtrim[0]:xtrim[1]]
log.info(f"Trimmed cube has dimensions: {cube.shape}")



##-------------------------------------------------------------------------
## Generate mask of low H-beta emission
##-------------------------------------------------------------------------
zpix = np.arange(0,cube.shape[0])
lambda_delta = 5
hbeta_z = np.where((np.array(wavs) > h_beta_std.value-lambda_delta)\
                   & (np.array(wavs) < h_beta_std.value+lambda_delta))[0]
image = np.mean(cube[min(hbeta_z):max(hbeta_z)+1, :, :], axis=0)
mask_pcnt = 10
nmask = (image < np.percentile(image, mask_pcnt))


log.info('Generating image of H-beta emission and mask for nebular emission')
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.title('Sum of H-beta Bins')
norm = v.ImageNormalize(image,
                        interval=v.ManualInterval(vmin=image.min()-5, vmax=image.max()+10),
                        stretch=v.LogStretch(10))
im = plt.imshow(image, origin='lower', norm=norm, cmap='Greys')
plt.colorbar(im)

plt.subplot(1,2,2)
plt.title('Region for sampling nebular emission')
mimage = np.ma.MaskedArray(image)
mimage.mask = ~nmask
mimagef = np.ma.filled(mimage, fill_value=0)
norm = v.ImageNormalize(mimagef,
                        interval=v.ManualInterval(vmin=image.min()-5, vmax=np.percentile(image, mask_pcnt)+5),
                        stretch=v.LinearStretch())
im = plt.imshow(mimagef, origin='lower', norm=norm, cmap='Greys')
plt.colorbar(im)

plt.savefig('H-beta Image.png', bbox_inches='tight', pad_inches=0.10)

##-------------------------------------------------------------------------
## Subtract Nebular Background
log.info('Subtracting nebular background from cube')
neb_subtracted = cube.copy()

neb_spect = [0]*cube.shape[0]
for z in zpix:
    neb_spect[z] = np.mean(cube[z][nmask])
    neb_subtracted[z,:,:] = neb_subtracted[z,:,:] - neb_spect[z]

if not os.path.exists(os.path.join(data_path, 'HH305E_nebsub.fits')):
    hdr = hdul[0].header
    xbkg = [xtrim[0]+xrange[0], xtrim[0]+xrange[1]]
    ybkg = [ytrim[0]+yrange[0], ytrim[0]+yrange[1]]
    now = dt.utcnow().strftime('%Y/%m/%d %H:%M:%S UT')
    hdr.set('HISTORY', f'Background subtracted x:{xbkg} y:{ybkg} [{now}]')
    hdu = fits.PrimaryHDU(data=neb_subtracted, header=hdr)
    hdu.writeto(os.path.join(data_path, 'HH305E_nebsub.fits'))


##-------------------------------------------------------------------------
## Model and Plot Nebular Background

log.info('Model and plot nebular background')
background_0 = models.Polynomial1D(degree=2)
H_beta_0 = models.Gaussian1D(amplitude=500, mean=4861, stddev=1.,
                  bounds={'mean': (4855,4865), 'stddev': (0.1,5)})
OIII4959_0 = models.Gaussian1D(amplitude=100, mean=4959, stddev=1.,
                    bounds={'mean': (4955,4965), 'stddev': (0.1,5)})
OIII5007_0 = models.Gaussian1D(amplitude=200, mean=5007, stddev=1.,
                    bounds={'mean': (5002,5012), 'stddev': (0.1,5)})
model0 = background_0 + H_beta_0 + OIII4959_0 + OIII5007_0
fitter = fitting.LevMarLSQFitter()

i_peak = neb_spect.index(max(neb_spect))
model0.mean_1 = wavs[i_peak]
model = fitter(model0, wavs, neb_spect)
residuals = np.array(neb_spect-model(wavs))

plt.figure(figsize=(20,8))

plt.subplot(1,3,1)
title = f'H-beta ({model.mean_1.value:.1f} A, sigma={model.stddev_1.value:.3f} A)'
plt.title(title)
w = [l for l in np.arange(4856,4866,0.05)]
plt.plot(wavs, neb_spect, drawstyle='steps-mid', label='data')
plt.plot(w, model(w), 'r-', alpha=0.7, label='Fit')
plt.xlabel('Wavelength (angstroms)')
plt.xlim(4856,4866)
plt.grid()
plt.ylabel('Flux')
plt.legend(loc='best')

plt.subplot(1,3,2)
title = f'OIII 4959 ({model.mean_2.value:.1f} A, sigma={model.stddev_2.value:.3f} A)'
plt.title(title)
plt.plot(wavs, neb_spect, drawstyle='steps-mid', label='data')
w = [l for l in np.arange(4954,4964,0.05)]
plt.plot(w, model(w), 'r-', alpha=0.7, label='Fit')
plt.xlabel('Wavelength (angstroms)')
plt.ylabel('Flux')
plt.legend(loc='best')
plt.xlim(4954,4964)

plt.subplot(1,3,3)
title = f'OIII 5007 ({model.mean_3.value:.1f} A, sigma={model.stddev_3.value:.3f} A)'
plt.title(title)
plt.plot(wavs, neb_spect, drawstyle='steps-mid', label='data')
w = [l for l in np.arange(5002,5012,0.05)]
plt.plot(w, model(w), 'r-', alpha=0.7, label='Fit')
plt.xlabel('Wavelength (angstroms)')
plt.ylabel('Flux')
plt.legend(loc='best')
plt.xlim(5002,5012)

plt.savefig('Nebular Background.png', bbox_inches='tight', pad_inches=0.10)

Hbeta_neb = model.mean_1.value * u.Angstrom




# xpix = 19
# ypix = 55
# xdelta = 4
# ydelta = 8
# yrange = [ypix-ydelta, ypix+ydelta]
# xrange = [xpix-xdelta, xpix+xdelta]
# 
# log.info('Modeling and plotting nebular background')
# spect, model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
#                              include_OIII=False,
#                              title='Nebular Background',
#                              filename='HH305E Nebular Background.png')


##-------------------------------------------------------------------------
## Shocks in HH305E
# xpix = 4
# ypix = 28
# xdelta = 2
# ydelta = 2
# yrange = [ypix-ydelta, ypix+ydelta]
# xrange = [xpix-xdelta, xpix+xdelta]
# 
# log.info(f'Modeling and plotting region around {xpix}, {ypix}')
# model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
#                              include_OIII=False,
#                              Hbeta_ref=Hbeta_neb,
#                              title='Nebula Subtracted',
#                              filename='HH305E Shock Region 1.png')
# 
# xpix = 8
# ypix = 48
# xdelta = 2
# ydelta = 2
# yrange = [ypix-ydelta, ypix+ydelta]
# xrange = [xpix-xdelta, xpix+xdelta]
# 
# log.info(f'Modeling and plotting region around {xpix}, {ypix}')
# model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
#                              include_OIII=False,
#                              Hbeta_ref=Hbeta_neb,
#                              title='Nebula Subtracted',
#                              filename='HH305E Shock Region 2.png')
# 
# xpix = 7
# ypix = 10
# xdelta = 2
# ydelta = 2
# yrange = [ypix-ydelta, ypix+ydelta]
# xrange = [xpix-xdelta, xpix+xdelta]
# 
# log.info(f'Modeling and plotting region around {xpix}, {ypix}')
# model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
#                              include_OIII=False,
#                              Hbeta_ref=Hbeta_neb,
#                              title='Nebula Subtracted',
#                              filename='HH305E Shock Region 3.png')
# 
# xpix = 16
# ypix = 15
# xdelta = 2
# ydelta = 4
# yrange = [ypix-ydelta, ypix+ydelta]
# xrange = [xpix-xdelta, xpix+xdelta]
# 
# log.info(f'Modeling and plotting region around {xpix}, {ypix}')
# model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
#                              include_OIII=False,
#                              Hbeta_ref=Hbeta_neb,
#                              title='Nebula Subtracted',
#                              filename='HH305E Shock Region 4.png')


##-------------------------------------------------------------------------
## Build Flux Map
##-------------------------------------------------------------------------
log.info(f'Building maps of H-beta flux and valocity')
zpix = np.arange(0,cube.shape[0])
lambda_delta = 5
hbeta_z = np.where((np.array(wavs) > h_beta_std.value-lambda_delta)\
                   & (np.array(wavs) < h_beta_std.value+lambda_delta))[0]
Hbeta_image = np.mean(cube[min(hbeta_z):max(hbeta_z)+1, :, :], axis=0)


map_shape = (int(np.floor(Hbeta_image.shape[0]/2)),
             int(np.floor(Hbeta_image.shape[1]/2)))
Hbeta_flux_map = np.ma.zeros(map_shape)
velocity_map = np.ma.zeros(map_shape)
turbulence_map = np.ma.zeros(map_shape)
residuals_map = np.zeros(map_shape)

Hbeta_flux_map2 = np.ma.zeros(map_shape)
velocity_map2 = np.ma.zeros(map_shape)
turbulence_map2 = np.ma.zeros(map_shape)


plt.figure(figsize=(40,40))

pixel_count = 0
log.info('  Iterating through all pixels in image')
for ypix in range(map_shape[0]-1,-1,-1):
    log.info(f'  Row {map_shape[0]-ypix} of {map_shape[0]}')
    for xpix in range(map_shape[1]):
        pixel_count += 1

        spect = neb_subtracted[:, 2*ypix:2*ypix+2, 2*xpix:2*xpix+2]
        spect = np.mean(np.mean(spect, axis=1), axis=1)
        i_peak = np.where(spect == spect.max())[0]
        model0.mean_1 = wavs[i_peak[0]]

        model = fitter(model0, wavs, spect)
        H_beta_flux = (2*np.pi)**0.5 * model.amplitude_1.value * model.stddev_1.value
        H_beta_velocity = (model.mean_1.value * u.angstrom).to(u.km/u.s,
                           equivalencies=u.doppler_optical(Hbeta_neb))
        H_beta_sigma = ((model.mean_1.value + model.stddev_1.value) * u.angstrom).to(u.km/u.s,
                         equivalencies=u.doppler_optical(model.mean_1.value * u.angstrom))
        residuals = np.array(spect-model(wavs))

        Hbeta_flux_map[ypix, xpix] = H_beta_flux
        velocity_map[ypix, xpix] = H_beta_velocity.value
        turbulence_map[ypix, xpix] = H_beta_sigma.value
        residuals_map[ypix, xpix] = np.sum(np.abs(residuals))

        wavs = np.array(wavs)
        whbeta = np.where((wavs >= 4856) & (wavs <= 4866))[0]
        dlambda = np.mean(np.gradient(wavs[whbeta]))
        H_beta_flux2 = np.sum(spect[whbeta])*dlambda
        H_beta_wav2 = np.sum(wavs[whbeta]*spect[whbeta])/H_beta_flux2*dlambda
        H_beta_velocity2 = (H_beta_wav2 * u.angstrom).to(u.km/u.s,
                           equivalencies=u.doppler_optical(Hbeta_neb))
        delta_wavs = wavs[whbeta] - H_beta_wav2
        H_beta_sigma_wav2 = np.std(delta_wavs*spect[whbeta]/H_beta_flux2)
        H_beta_sigma2 = ((H_beta_wav2 + H_beta_sigma_wav2) * u.angstrom).to(u.km/u.s,
                         equivalencies=u.doppler_optical(H_beta_wav2 * u.angstrom))
        Hbeta_flux_map2[ypix, xpix] = H_beta_flux2
        velocity_map2[ypix, xpix] = H_beta_velocity2.value
        turbulence_map2[ypix, xpix] = H_beta_sigma2.value

#         print(min(spect[whbeta]), np.mean(spect[whbeta]))
#         print(H_beta_wav2, model.mean_1.value)
#         print(H_beta_sigma_wav2, H_beta_sigma2.value)
#         print(model.stddev_1.value, H_beta_sigma.value)
#         print()

        plt.subplot(map_shape[0], map_shape[1], pixel_count)
        w = [l for l in np.arange(4856,4866,0.05)]
        vs = [(l*u.Angstrom).to(u.km/u.s,
              equivalencies=u.doppler_optical(Hbeta_neb)).value
              for l in wavs]
        plt.plot(vs, spect, drawstyle='steps-mid', label='data')
        vs = [(l*u.Angstrom).to(u.km/u.s,
              equivalencies=u.doppler_optical(Hbeta_neb)).value
              for l in w]
        plt.plot(vs, model(w), 'r-', alpha=0.4, label='Fit')
        plt.plot([H_beta_velocity2.value]*2, [-100, 250], 'g-', alpha=0.5)
        plt.text(0, -50, f"{xpix}, {ypix} ({H_beta_flux:.0f})")
        plt.yticks([0,100,200])
#         plt.ylim(-100,300)
        plt.ylim(-10,30)
        plt.xticks([-100,0,100])
        plt.xlim(-175,175)
        if ypix < map_shape[0]-1:
            plt.gca().set_xticklabels([])
        if xpix > 0:
            plt.gca().set_yticklabels([])
        plt.gca().tick_params(direction='in')
        plt.grid()


plt.savefig('Spectrum Map.png', bbox_inches='tight', pad_inches=0.10)

log.info('  Generating maps as png files')

plt.figure(figsize=(14,8))

plt.subplot(1,4,1)
plt.title('H-beta Flux (Log Stretch)')
norm = v.ImageNormalize(Hbeta_flux_map,
                        interval=v.ManualInterval(vmin=0, vmax=400),
                        stretch=v.LinearStretch())
#                         stretch=v.LogStretch(10))
im = plt.imshow(np.ma.filled(Hbeta_flux_map, fill_value=-10),
                origin='lower', norm=norm, cmap='Greys')
plt.colorbar(im)

plt.subplot(1,4,2)
plt.title('Residuals')
norm = v.ImageNormalize(residuals_map,
                        interval=v.ZScaleInterval(),
                        stretch=v.LinearStretch())
im = plt.imshow(np.ma.filled(residuals_map, fill_value=0),
                origin='lower', norm=norm, cmap='Greys')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.colorbar(im)

plt.subplot(1,4,3)
plt.title('Velocity')
norm = v.ImageNormalize(velocity_map,
                        interval=v.ManualInterval(vmin=-60, vmax=60),
                        stretch=v.LinearStretch())
im = plt.imshow(np.ma.filled(velocity_map, fill_value=0),
                origin='lower', norm=norm, cmap='coolwarm')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.colorbar(im)

plt.subplot(1,4,4)
plt.title('Line Width')
norm = v.ImageNormalize(turbulence_map,
                        interval=v.ZScaleInterval(),
                        stretch=v.LinearStretch())
im = plt.imshow(np.ma.filled(turbulence_map, fill_value=0),
                origin='lower', norm=norm, cmap='Greys')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.colorbar(im)

plt.savefig('Velocity Map.png', bbox_inches='tight', pad_inches=0.10)





plt.figure(figsize=(14,8))

plt.subplot(1,4,1)
plt.title('H-beta Flux (Log Stretch)')
norm = v.ImageNormalize(Hbeta_flux_map2,
                        interval=v.ManualInterval(vmin=0, vmax=500),
                        stretch=v.LinearStretch())
#                         stretch=v.LogStretch(10))
im = plt.imshow(np.ma.filled(Hbeta_flux_map2, fill_value=-10),
                origin='lower', norm=norm, cmap='Greys')
plt.colorbar(im)

# plt.subplot(1,4,2)
# plt.title('Residuals')
# norm = v.ImageNormalize(residuals_map,
#                         interval=v.ZScaleInterval(),
#                         stretch=v.LinearStretch())
# im = plt.imshow(np.ma.filled(residuals_map, fill_value=0),
#                 origin='lower', norm=norm, cmap='Greys')
# plt.gca().set_xticklabels([])
# plt.gca().set_yticklabels([])
# plt.colorbar(im)

plt.subplot(1,4,3)
plt.title('Velocity')
norm = v.ImageNormalize(velocity_map2,
                        interval=v.ManualInterval(vmin=-60, vmax=60),
                        stretch=v.LinearStretch())
im = plt.imshow(np.ma.filled(velocity_map2, fill_value=0),
                origin='lower', norm=norm, cmap='coolwarm')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.colorbar(im)

plt.subplot(1,4,4)
plt.title('Line Width')
norm = v.ImageNormalize(turbulence_map2,
                        interval=v.ZScaleInterval(),
                        stretch=v.LinearStretch())
im = plt.imshow(np.ma.filled(turbulence_map2, fill_value=0),
                origin='lower', norm=norm, cmap='Greys')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.colorbar(im)

plt.savefig('Velocity Map2.png', bbox_inches='tight', pad_inches=0.10)
