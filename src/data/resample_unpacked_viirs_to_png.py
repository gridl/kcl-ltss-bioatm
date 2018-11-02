# -*- coding: utf-8 -*-

import logging
import os
import re
import glob
from datetime import datetime

import h5py
import numpy as np
import scipy.misc as misc
import pandas as pd

import src.config.filepaths as fp
from sklearn.cluster import DBSCAN


def subset_fires_to_image(lat, lon, fire_df, date_to_find):

    fire_subset = fire_df[fire_df.date_time == date_to_find]
    fire_subset = fire_subset[((fire_df.latitude > np.min(lat)) &
                               (fire_df.latitude < np.max(lat)) &
                               (fire_df.longitude > np.min(lon)) &
                               (fire_df.longitude < np.max(lon)))]
    return fire_subset


def grid_indexes(lat):

    rows = np.arange(lat.shape[0])
    cols = np.arange(lat.shape[1])
    cols, rows = np.meshgrid(cols, rows)
    return rows, cols


def mean_fire_position(fire_subset_df):
    coords = fire_subset_df[['latitude', 'longitude']].values
    db = DBSCAN(eps=10 / 6371., min_samples=1, algorithm='ball_tree', metric='haversine').fit(
        np.radians(coords))
    fire_subset_df['cluster_id'] = db.labels_
    return fire_subset_df.groupby('cluster_id').agg({'latitude': np.mean, 'longitude': np.mean})



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def locate_fire_in_image(fire_coords, lats, lons, rows, cols):
    import time

    fire_rows = []
    fire_cols = []

    for fire_lat, fire_lon in zip(fire_coords.latitude.values, fire_coords.longitude.values):

        try:
            mask = (lats > fire_lat - 0.05) & (lats < fire_lat + 0.05) & \
                   (lons > fire_lon - 0.05) & (lons < fire_lon + 0.05)
            sub_lats = lats[mask]
            sub_lons = lons[mask]
            sub_rows = rows[mask]
            sub_cols = cols[mask]


            # find exact loc using haversine distance
            sub_index = np.argmin(haversine(fire_lon, fire_lat, sub_lons, sub_lats))

            # append row and col for  exact location
            fire_rows.append(sub_rows[sub_index])
            fire_cols.append(sub_cols[sub_index])

        except:
            continue

    return fire_rows, fire_cols


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def rgb_viirs(viirs_data, tcc=True):

    if tcc:
        r = viirs_data['All_Data']['VIIRS-M5']['Radiance'][:]
        g = viirs_data['All_Data']['VIIRS-M4']['Radiance'][:]
        b = viirs_data['All_Data']['VIIRS-M1']['Radiance'][:]
    else:
        r = viirs_data['All_Data']['VIIRS-M15-SDR_All']['Radiance'][:]
        g = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
        b = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]

    r = image_histogram_equalization(r)
    g = image_histogram_equalization(g)
    b = image_histogram_equalization(b)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    rgb = np.dstack((r, g, b))
    return rgb


def main():

    # load in fires
    path = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_id_test'
    viirs_fire_csv = 'fire_archive_V1_24485.csv'
    viirs_fire_df = pd.read_csv(os.path.join(path, 'VIIRS', viirs_fire_csv))
    viirs_fire_df['date_time'] = pd.to_datetime(viirs_fire_df['acq_date'])

    # set up datetime list for processing
    file_times = []
    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr_reprojected_h5):
        try:
            file_times.append(re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group())
        except:
            continue
    file_times = set(file_times)

    for file_time in file_times:

        viirs_dt = datetime.strptime(file_time, 'd%Y%m%d_t%H%M%S')
        date_to_find = pd.Timestamp(viirs_dt.year, viirs_dt.month, viirs_dt.day)

        sdr_file_path = glob.glob(fp.path_to_viirs_sdr_reprojected_h5 + '/*' + file_time + '*')[0]
        viirs_sdr = h5py.File(sdr_file_path, 'r')

        # read out lat and lon and subset fires to scene
        lat = viirs_sdr['Latitude'][:]
        lon = viirs_sdr['Longitude'][:]
        fire_subset_df = subset_fires_to_image(lat, lon, viirs_fire_df, date_to_find)

        mean_fire_geo_locs = mean_fire_position(fire_subset_df)

        # find fire coordinate in the image
        image_rows, image_cols = grid_indexes(lat)
        fire_rows, fire_cols = locate_fire_in_image(mean_fire_geo_locs,
                                                     lat, lon, image_rows, image_cols)

        # tcc = rgb_viirs(viirs_sdr)
        # misc.imsave(os.path.join(fp.path_to_viirs_sdr_reprojected_tcc, viirs_sdr_fname.replace('h5', 'png')), tcc)
        #
        # fcc = rgb_viirs(viirs_sdr, tcc=False)
        # misc.imsave(os.path.join(fp.path_to_viirs_sdr_reprojected_fcc, viirs_sdr_fname.replace('h5', 'png')), fcc)

        im = viirs_sdr['VIIRS-M1'][:]
        im = np.round((im * (255 / np.max(im))) * 1).astype('uint8')

        out_im = np.dstack((im, im, im))
        out_im[fire_rows, fire_cols, 0] = 255
        out_im[fire_rows, fire_cols, 1] = 0
        out_im[fire_rows, fire_cols, 2] = 0

        misc.imsave(os.path.join(fp.path_to_viirs_sdr_reprojected_blue,
                                 'viirs_blue_' + file_time + '.png'), out_im)

if __name__ == "__main__":
    main()