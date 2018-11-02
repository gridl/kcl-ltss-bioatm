import os
import re
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy.misc import imsave

import src.features.plume_identifier as pi
import src.data.tools as tools
import src.config.filepaths as fp


def main():

    PIXEL_SIZE = 750  # size of resampled pixels in m for VIIRS data
    FILL_VALUE = np.nan  # resampling fill value

    viirs_fire_df = pd.read_csv(fp.path_to_fire_df)
    viirs_fire_df['date_time'] = pd.to_datetime(viirs_fire_df['acq_date'])

    def resample(img, lat, lon, null_value=0):
        resampler = tools.utm_resampler(lat, lon, PIXEL_SIZE)

        lonlats = resampler.area_def.get_lonlats()
        lat_grid = lonlats[1]
        lon_grid = lonlats[0]

        mask = img < null_value
        masked_lats = np.ma.masked_array(resampler.lats, mask)
        masked_lons = np.ma.masked_array(resampler.lons, mask)
        img = resampler.resample_image(img, masked_lats, masked_lons, fill_value=FILL_VALUE)
        return img, lat_grid, lon_grid

    # define paths to data
    aod_files = os.listdir(fp.path_to_viirs_aod)
    geo_files = os.listdir(fp.path_to_viirs_geo)
    for aod_fname, geo_fname in zip(aod_files, geo_files):
        aod_h5 = h5py.File(os.path.join(fp.path_to_viirs_aod, aod_fname), "r")
        geo_h5 = h5py.File(os.path.join(fp.path_to_viirs_geo, geo_fname), "r")

        aod = aod_h5['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
        lat = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Latitude'][:]
        lon = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Longitude'][:]

        dt_str = re.search("[d][0-9]{8}[_][t][0-9]{6}", aod_fname).group()
        dt = datetime.strptime(dt_str, 'd%Y%m%d_t%H%M%S')
        date_to_find = pd.Timestamp(dt.year, dt.month, dt.day)

        aod_r, lat_r, lon_r = resample(aod, lat, lon)
        _, plume_mask = pi.identify(aod_r, lat_r, lon_r, date_to_find, viirs_fire_df)

        # save the plume mask
        imsave(os.path.join(fp.path_to_viirs_masks, dt_str+'-mask.png'), plume_mask.astype(int))

if __name__ == "__main__":
    main()