# -*- coding: utf-8 -*-

import logging
import os

import h5py
import numpy as np
import scipy.misc as misc

import src.data.tools as tools
import src.config.filepaths as fp

import matplotlib.pyplot as plt


def read_h5(f):
    return h5py.File(f,  "r")


def create_resampler(viirs_data):
    lats = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Latitude'][:]
    lons = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Longitude'][:]
    return tools.utm_resampler(lats, lons, 750)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def rgb_viirs(viirs_data, resampler, tcc=True):

    if tcc:
        r = viirs_data['All_Data']['VIIRS-M5-SDR_All']['Radiance'][:]
        g = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
        b = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]
    else:
        r = viirs_data['All_Data']['VIIRS-M15-SDR_All']['Radiance'][:]
        g = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
        b = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]

    mask = g<0
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)

    resampled_r = resampler.resample_image(r, masked_lats, masked_lons, fill_value=0)
    resampled_g = resampler.resample_image(g, masked_lats, masked_lons, fill_value=0)
    resampled_b = resampler.resample_image(b, masked_lats, masked_lons, fill_value=0)

    r = image_histogram_equalization(resampled_r)
    g = image_histogram_equalization(resampled_g)
    b = image_histogram_equalization(resampled_b)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    rgb = np.dstack((r, g, b))
    return rgb


def main():

    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr):

        if os.path.isfile(os.path.join(
                fp.path_to_viirs_sdr_reprojected_fcc, viirs_sdr_fname.replace('h5', 'png'))):
            print(viirs_sdr_fname, 'already resampled')
            continue

        logger.info("Processing viirs file: " + viirs_sdr_fname)

        if 'DS' in viirs_sdr_fname:
            continue

        try:
            viirs_sdr = read_h5(os.path.join(fp.path_to_viirs_sdr, viirs_sdr_fname))
            utm_resampler = create_resampler(viirs_sdr)

            tcc = rgb_viirs(viirs_sdr, utm_resampler)
            misc.imsave(os.path.join(fp.path_to_viirs_sdr_reprojected_tcc, viirs_sdr_fname.replace('h5', 'png')), tcc)

            fcc = rgb_viirs(viirs_sdr, utm_resampler, tcc=False)
            misc.imsave(os.path.join(fp.path_to_viirs_sdr_reprojected_fcc, viirs_sdr_fname.replace('h5', 'png')), fcc)
        except Exception as e:
            print('Failed with exception:', e)
            continue

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()