import logging
import os

import h5py
import pandas as pd
import numpy as np

import src.data.tools as tools


def extract_fires_for_roi(df, ts, extent):
    time_subset = df[df.date_time == ts]
    time_space_subset = time_subset[((df.latitude > extent['min_lat']) &
                                    (df.latitude < extent['max_lat']) &
                                    (df.longitude > extent['min_lon']) &
                                    (df.longitude < extent['max_lon']))]
    return time_space_subset


def read_aod_mask(arr, bit_pos, bit_len, value):
    '''Generates mask with given bit information.
    Parameters
        bit_pos		-	Position of the specific QA bits in the value string.
        bit_len		-	Length of the specific QA bits.
        value  		-	A value indicating the desired condition.
    '''
    bitlen = int('1' * bit_len, 2)

    if type(value) == str:
        value = int(value, 2)

    pos_value = bitlen << bit_pos
    con_value = value << bit_pos
    mask = (arr & pos_value) == con_value
    return mask


def get_image_coords(fire_lats, fire_lons, resampled_lats, resampled_lons):
    inverse_lats = resampled_lats * -1  # invert lats for correct indexing

    y_size, x_size = resampled_lats.shape

    min_lat = np.min(inverse_lats[inverse_lats > -1000])
    range_lat = np.max(inverse_lats) - min_lat

    min_lon = np.min(resampled_lons)
    range_lon = np.max(resampled_lons[resampled_lons < 1000]) - min_lon

    # get approximate fire location, remembering to invert the lat
    y = ((fire_lats * -1) - min_lat) / range_lat * y_size
    x = (fire_lons - min_lon) / range_lon * x_size

    return y.astype(int), x.astype(int)



def identify(aod, r):

    extent = {'min_lat': lats_r.min(),
              'max_lat': lats_r.max(),
              'min_lon': lons_r.min(),
              'max_lon': lons_r.max()}
    date_to_find = pd.Timestamp(2015, 7, 6)
    image_fires_df = extract_fires_for_roi(fire_df, date_to_find, extent)


def main():

    # data setup for testing
    path = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/reprojected_viirs/aod_classifier_test'
    aod_fname = 'IVAOT_npp_d20150706_t0603097_e0604339_b19108_c20150706075934735034_noaa_ops.h5'
    geo_fname = 'GMTCO_npp_d20150706_t0603097_e0604339_b19108_c20171126104946623808_noaa_ops.h5'
    aod_h5 = h5py.File(os.path.join(path, aod_fname), "r")
    geo_h5 = h5py.File(os.path.join(path, geo_fname), "r")

    fire_csv = 'fire_archive_V1_26373.csv'
    fire_df = pd.read_csv(os.path.join(path, fire_csv))
    fire_df['date_time'] = pd.to_datetime(fire_df['acq_date'])

    aod = aod_h5['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    aod_qual = aod_h5['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['QF1'][:]
    lat = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Latitude'][:]
    lon = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Longitude'][:]

    flag = np.zeros(aod_qual.shape)
    for k, v in zip(['00', '01', '10', '11'], [0, 1, 2, 3]):
        mask = read_aod_mask(aod_qual, 0, 2, k)
        flag[mask] = v

    # resample data to UTM
    utm_resampler = tools.utm_resampler(lat, lon, 750)
    fv = -999.0

    mask = aod < 0
    masked_lats = np.ma.masked_array(utm_resampler.lats, mask)
    masked_lons = np.ma.masked_array(utm_resampler.lons, mask)

    lats_r = utm_resampler.resample_image(utm_resampler.lats, masked_lats, masked_lons, fill_value=fv)
    lons_r = utm_resampler.resample_image(utm_resampler.lons, masked_lats, masked_lons, fill_value=fv)
    aod_r = utm_resampler.resample_image(aod, masked_lats, masked_lons, fill_value=fv)
    flag_r = utm_resampler.resample_image(flag, masked_lats, masked_lons, fill_value=fv)

    # subset to test roi
    ymin = 425
    ymax = 850
    xmin = 1800
    xmax = 2300

    aod_r = aod_r[425:850, 1800:2300]
    flag_r = flag_r[425:850, 1800:2300]
    lats_r = lats_r[425:850, 1800:2300]
    lons_r = lons_r[425:850, 1800:2300]




if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()