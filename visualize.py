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
## Load HH305E Data
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
## Subtract Nebular Background from Cube
##-------------------------------------------------------------------------
log.info('Generating image of H-beta emission and mask for nebular emission')
zpix = np.arange(0,cube.shape[0])
lambda_delta = 5
hbeta_z = np.where((np.array(wavs) > h_beta_std.value-lambda_delta)\
                   & (np.array(wavs) < h_beta_std.value+lambda_delta))[0]
image = np.mean(cube[min(hbeta_z):max(hbeta_z)+1, :, :], axis=0)
mask_pcnt = 10
nmask = (image < np.percentile(image, mask_pcnt))

log.info('Subtracting nebular background from cube')
neb_subtracted = cube.copy()

neb_spect = [0]*cube.shape[0]
for z in zpix:
    neb_spect[z] = np.mean(cube[z][nmask])
    neb_subtracted[z,:,:] = neb_subtracted[z,:,:] - neb_spect[z]

if not os.path.exists(os.path.join(data_path, 'HH305E_nebsub.fits')):
    hdr = hdul[0].header
    now = dt.utcnow().strftime('%Y/%m/%d %H:%M:%S UT')
    hdr.set('HISTORY', f'Background subtracted {now}')
    hdu = fits.PrimaryHDU(data=neb_subtracted, header=hdr)
    hdu.writeto(os.path.join(data_path, 'HH305E_nebsub.fits'))

##-------------------------------------------------------------------------
## Plot mask of low H-beta emission
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.title('Sum of H-beta Bins')
norm = v.ImageNormalize(image,
                        interval=v.ManualInterval(vmin=image.min()-5, vmax=image.max()+10),
                        stretch=v.LogStretch(10))
im = plt.imshow(image, origin='lower', norm=norm, cmap='Greys')
plt.colorbar(im)

plt.subplot(1,2,2)
plt.title('Nebular Emission Mask')
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
nebmodel = fitter(model0, wavs, neb_spect)
residuals = np.array(neb_spect-nebmodel(wavs))

plt.figure(figsize=(20,8))

plt.subplot(1,3,1)
title = f'H-beta ({nebmodel.mean_1.value:.1f} A, sigma={nebmodel.stddev_1.value:.3f} A)'
plt.title(title)
w = [l for l in np.arange(4856,4866,0.05)]
plt.plot(wavs, neb_spect, drawstyle='steps-mid', label='data')
plt.plot(w, nebmodel(w), 'r-', alpha=0.7, label='Fit')
plt.xlabel('Wavelength (angstroms)')
plt.xlim(4856,4866)
plt.grid()
plt.ylabel('Flux')
plt.legend(loc='best')

plt.subplot(1,3,2)
title = f'OIII 4959 ({nebmodel.mean_2.value:.1f} A, sigma={nebmodel.stddev_2.value:.3f} A)'
plt.title(title)
plt.plot(wavs, neb_spect, drawstyle='steps-mid', label='data')
wOIII4959 = [l for l in np.arange(4954,4964,0.05)]
plt.plot(wOIII4959, nebmodel(wOIII4959), 'r-', alpha=0.7, label='Fit')
plt.xlabel('Wavelength (angstroms)')
plt.ylabel('Flux')
plt.legend(loc='best')
plt.xlim(4954,4964)

plt.subplot(1,3,3)
title = f'OIII 5007 ({nebmodel.mean_3.value:.1f} A, sigma={nebmodel.stddev_3.value:.3f} A)'
plt.title(title)
plt.plot(wavs, neb_spect, drawstyle='steps-mid', label='data')
wOIII5007 = [l for l in np.arange(5002,5012,0.05)]
plt.plot(wOIII5007, nebmodel(wOIII5007), 'r-', alpha=0.7, label='Fit')
plt.xlabel('Wavelength (angstroms)')
plt.ylabel('Flux')
plt.legend(loc='best')
plt.xlim(5002,5012)

plt.savefig('Nebular Background.png', bbox_inches='tight', pad_inches=0.10)

Hbeta_neb = nebmodel.mean_1.value * u.Angstrom
OIII4959_neb = nebmodel.mean_2.value * u.Angstrom
OIII5007_neb = nebmodel.mean_3.value * u.Angstrom

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


fig_Hbeta = plt.figure(figsize=(40,40))
fig_OIII = plt.figure(figsize=(40,40))

pixel_count = 0
log.info('  Iterating through all pixels in image')
vs_for_spect = [(l*u.Angstrom).to(u.km/u.s,
                equivalencies=u.doppler_optical(Hbeta_neb)).value
                for l in wavs]
vs_for_model = [(l*u.Angstrom).to(u.km/u.s,
                 equivalencies=u.doppler_optical(Hbeta_neb)).value
                 for l in w]
vs_for_OIII4959 = [(l*u.Angstrom).to(u.km/u.s,
                equivalencies=u.doppler_optical(OIII4959_neb)).value
                for l in wavs]
vs_for_OIII5007 = [(l*u.Angstrom).to(u.km/u.s,
                equivalencies=u.doppler_optical(OIII5007_neb)).value
                for l in wavs]

wavs = np.array(wavs)
whbeta = np.where((wavs >= 4856) & (wavs <= 4866))[0]
wOIII4959 = np.where((wavs >= 4954) & (wavs <= 4964))[0]
wOIII5007 = np.where((wavs >= 5002) & (wavs <= 5012))[0]
dlambda = np.mean(np.gradient(wavs[whbeta]))

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

        ax = fig_Hbeta.add_subplot(map_shape[0]+2, map_shape[1], pixel_count)
        ax.plot(vs_for_spect, spect, 'k-', drawstyle='steps-mid', label='data')
#         spanmin = H_beta_velocity2.value-H_beta_sigma2.value
#         spanmax = H_beta_velocity2.value+H_beta_sigma2.value
#         ax.plot([H_beta_velocity2.value]*2, [-100, 300], 'g-', alpha=0.75)
#         ax.axvspan(spanmin, spanmax, ymin=-100, ymax=300, fc='g', alpha=0.1)

        ax.plot(vs_for_model, model(w), 'r-', alpha=0.5, label='Fit')
        ax.plot([H_beta_velocity.value]*2, [-100, 300], 'r-', alpha=0.25)

        ax.yaxis.set_ticks([0,200])
        ax.set_ylim(-100,300)
        ax.text(-150, 50, f"F={H_beta_flux:.1f}")
        ax.xaxis.set_ticks([-100,0,100])
        ax.set_xlim(-175,175)
        ax.xaxis.set_tick_params(direction='in')
        ax.grid()

        if xpix > 0:
            ax.set_yticklabels([])
        else:
            if ypix == int(map_shape[0]/2):
                ax.set_ylabel('Signal (e-)')

        if ypix > 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Velocity (km/s)')

        # OIII Plots
        axo = fig_OIII.add_subplot(map_shape[0]+2, map_shape[1], pixel_count)
        axo.plot(vs_for_OIII4959, spect, 'b-', drawstyle='steps-mid',
                 label='OIII 4959')
        axo.plot(vs_for_OIII5007, spect, 'r-', drawstyle='steps-mid',
                 label='OIII 5007')

        axo.yaxis.set_ticks([-20,0,20])
        axo.set_ylim(-25,25)
        axo.xaxis.set_ticks([-100,0,100])
        axo.set_xlim(-175,175)
        axo.xaxis.set_tick_params(direction='in')
        axo.grid()

        if xpix > 0:
            axo.set_yticklabels([])
        else:
            if ypix == int(map_shape[0]/2):
                axo.set_ylabel('Signal (e-)')

        if ypix > 0:
            axo.set_xticklabels([])
        else:
            axo.set_xlabel('Velocity (km/s)')

ax = fig_Hbeta.add_subplot(map_shape[0]+2, map_shape[1], pixel_count+map_shape[1]+1)
ax.set_title('Nebular Background')
ax.plot(vs_for_spect, neb_spect, 'k-', drawstyle='steps-mid', label='data')
ax.plot(vs_for_model, model(w), 'r-', alpha=0.5, label='Fit')
ax.plot([0]*2, [-100, 300], 'r-', alpha=0.25)
ax.yaxis.set_ticks([0,200])
ax.set_ylim(-100,300)
ax.xaxis.set_ticks([-100,0,100])
ax.set_xlim(-175,175)
ax.xaxis.set_tick_params(direction='in')
ax.set_xlabel('Velocity (km/s)')
ax.grid()

ax = fig_Hbeta.add_subplot(map_shape[0]+2, map_shape[1], pixel_count+map_shape[1]+2)
ax.set_title('Nebular Background')
ax.plot(vs_for_spect, neb_spect, 'k-', drawstyle='steps-mid', label='data')
ax.yaxis.set_ticks([0,200,400])
ax.set_ylim(-100,600)
ax.xaxis.set_ticks([-100,0,100])
ax.set_xlim(-175,175)
ax.xaxis.set_tick_params(direction='in')
ax.set_xlabel('Velocity (km/s)')
ax.grid()

axo = fig_OIII.add_subplot(map_shape[0]+2, map_shape[1], pixel_count+map_shape[1]+1)
axo.set_title('Nebular Background')
axo.plot(vs_for_OIII4959, neb_spect, 'b-', drawstyle='steps-mid', label='data')
axo.plot(vs_for_OIII5007, neb_spect, 'r-', drawstyle='steps-mid', label='data')
axo.yaxis.set_ticks([-20,0,20])
axo.set_ylim(-25,25)
axo.xaxis.set_ticks([-100,0,100])
axo.set_xlim(-175,175)
axo.xaxis.set_tick_params(direction='in')
axo.grid()
axo.set_xlabel('Velocity (km/s)')

axo = fig_OIII.add_subplot(map_shape[0]+2, map_shape[1], pixel_count+map_shape[1]+2)
axo.set_title('Nebular Background')
axo.plot(vs_for_OIII4959, neb_spect, 'b-', drawstyle='steps-mid', label='data')
axo.plot(vs_for_OIII5007, neb_spect, 'r-', drawstyle='steps-mid', label='data')
axo.yaxis.set_ticks([0,100])
axo.set_ylim(-10,200)
axo.xaxis.set_ticks([-100,0,100])
axo.set_xlim(-175,175)
axo.xaxis.set_tick_params(direction='in')
axo.grid()
axo.set_xlabel('Velocity (km/s)')

log.info('  Saving spectrum maps')
fig_Hbeta.savefig('Spectrum Map.png', bbox_inches='tight', pad_inches=0.10)
fig_OIII.savefig('Spectrum Map OIII.png', bbox_inches='tight', pad_inches=0.10)

##-------------------------------------------------------------------------
## Generate Map1
##-------------------------------------------------------------------------
log.info('  Generating property maps')

plt.figure(figsize=(16,8))

plt.subplot(1,4,1)
plt.title('H-beta Flux')
norm = v.ImageNormalize(Hbeta_flux_map,
                        interval=v.ManualInterval(vmin=0, vmax=500),
                        stretch=v.LogStretch(10))
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


##-------------------------------------------------------------------------
## Generate Map2
##-------------------------------------------------------------------------
plt.figure(figsize=(16,8))

plt.subplot(1,4,1)
plt.title('H-beta Flux')
norm = v.ImageNormalize(Hbeta_flux_map2,
                        interval=v.ManualInterval(vmin=0, vmax=500),
#                         stretch=v.LinearStretch())
                        stretch=v.LogStretch(10))
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
xpix = 4
ypix = 28
xdelta = 2
ydelta = 2
yrange = [ypix-ydelta, ypix+ydelta]
xrange = [xpix-xdelta, xpix+xdelta]

log.info(f'Modeling and plotting region around {xpix}, {ypix}')
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

log.info(f'Modeling and plotting region around {xpix}, {ypix}')
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

log.info(f'Modeling and plotting region around {xpix}, {ypix}')
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

log.info(f'Modeling and plotting region around {xpix}, {ypix}')
model = plot_image_and_lines(neb_subtracted, wavs, xrange, yrange,
                             include_OIII=False,
                             Hbeta_ref=Hbeta_neb,
                             title='Nebula Subtracted',
                             filename='HH305E Shock Region 4.png')

