import logging
import os

import h5py
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

import src.data.tools as tools

# Constants
PIXEL_SIZE = 750  # size of resampled pixels in m
FILL_VALUE = np.nan  # resampling fill value
MIN_FRP = 10  # Only fires greatert han 10 MW are considered in clustering
CLUSTER_DIST = 10  # fires less than this distance apart (in KM) are clustered
P_ID_WIN_SIZE = 10  # plume identification window size in pix (half window e.g. for 21 use 10)
AOD_RATIO_LIMIT = 3  # if ratio is greater than this then assume a plume (also
DISTANCE_MATRIX = construct_dist_matrix()  # used to determine the distance of a fire from a plume in pixels


def construct_dist_matrix():
    x = np.arange(-P_ID_WIN_SIZE, P_ID_WIN_SIZE+1)
    y = np.arange(-P_ID_WIN_SIZE, P_ID_WIN_SIZE+1)
    dx, dy = np.meshgrid(x, y)
    return np.sqrt(dx**2 + dy**2)


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


def resample(aod, flag, lat, lon):
    resampler = tools.utm_resampler(lat, lon, PIXEL_SIZE)

    lonlats = resampler.area_def.get_lonlats()
    lat_grid = lonlats[1]
    lon_grid = lonlats[0]

    mask = aod < 0
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    aod = resampler.resample_image(aod, masked_lats, masked_lons, fill_value=FV)
    flag = resampler.resample_image(flag, masked_lats, masked_lons, fill_value=FV)
    return aod, flag, lat_grid, lon_grid


def subset_fires_to_image(lat, lon, fire_df, date_to_find):

    fire_subset = fire_df[fire_df.date_time == date_to_find]
    fire_subset = fire_subset[((fire_df.latitude > np.min(lat)) &
                               (fire_df.latitude < np.max(lat)) &
                               (fire_df.longitude > np.min(lon)) &
                               (fire_df.longitude < np.max(lon)))]
    fire_subset = fire_subset.loc[fire_subset.frp > MIN_FRP]
    return fire_subset


def mean_fire_position(fire_subset_df):
    coords = fire_subset_df[['latitude', 'longitude']].values
    db = DBSCAN(eps=CLUSTER_DIST / 6371., min_samples=1, algorithm='ball_tree', metric='haversine').fit(
        np.radians(coords))
    fire_subset_df['cluster_id'] = db.labels_
    return fire_subset_df.groupby('cluster_id').agg({'latitude': np.mean, 'longitude': np.mean})


def grid_indexes(lat):
    mask = lat != FILL_VALUE
    rows = np.arange(lat.shape[0])
    cols = np.arange(lat.shape[1])
    cols, rows = np.meshgrid(cols, rows)
    return rows[mask], cols[mask]


def build_balltree(lat, lon):
    mask = lat != FILL_VALUE
    array_lat_lon = np.dstack([np.deg2rad(lat[mask]),
                               np.deg2rad(lon[mask])])[0]
    return BallTree(array_lat_lon, metric='haversine')


def locate_fires_in_image(tree, fire_pos, rows, cols):
    point_locations = np.dstack((np.deg2rad(fire_pos.latitude.values),
                                 np.deg2rad(fire_pos.longitude.values))).squeeze()
    distance, index = tree.query(point_locations, k=1)
    return rows[index], cols[index], np.rad2deg(distance)


def locate_fires_near_plumes(aod, fire_rows, fire_cols):

    r_near_plume = []
    c_near_plume = []
    max_mean_aod_near_fire = []

    for r, c in zip(fire_rows, fire_cols):

        r = r[0]
        c = c[0]

        # get bb and aod
        min_r = r - P_ID_WIN_SIZE if r - P_ID_WIN_SIZE > 0 else 0
        max_r = r + P_ID_WIN_SIZE + 1 if r + P_ID_WIN_SIZE + 1 < aod.shape[0] else aod.shape[0]
        min_c = c - P_ID_WIN_SIZE if c - P_ID_WIN_SIZE > 0 else 0
        max_c = c + P_ID_WIN_SIZE + 1 if c + P_ID_WIN_SIZE + 1 < aod.shape[1] else aod.shape[1]

        aod_for_window = aod[min_r:max_r, min_c:max_c]

        # skip windows on edge of image
        if aod_for_window.size != (P_ID_WIN_SIZE * 2 + 1) ** 2:
            continue

        # find means of all 9 background windows
        sub_window_means = []
        step_size = int((P_ID_WIN_SIZE * 2 + 1) / 3)
        for i in [0, step_size, step_size * 2]:
            for j in [0, step_size, step_size * 2]:
                sub_window_means.append(np.mean(aod_for_window[i:i + step_size,
                                                               j:j + step_size]))

        # The ratio allows us to eliminate fires under smoke clouds, or without clear backgrounds, as
        # the smoke signal needs to be at least a factor of three higher than background.  If all the
        # background is not clear then it will not be located.
        min_mean = np.min(sub_window_means)
        max_mean = np.max(sub_window_means)

        if max_mean / min_mean > AOD_RATIO_LIMIT:
            r_near_plume.append(r)
            c_near_plume.append(c)
            max_mean_aod_near_fire.append(max_mean)

        return r_near_plume, c_near_plume, max_mean_aod_near_fire


def extract_label(labelled_image, r, c):
    labelled_subset = labelled_image[r - P_ID_WIN_SIZE:r + P_ID_WIN_SIZE + 1,
                      c - P_ID_WIN_SIZE:c + P_ID_WIN_SIZE + 1]
    label_mask = labelled_subset != 0
    if label_mask.any():
        labelled_subset = labelled_subset[label_mask]
        distances = DISTANCE_MATRIX[label_mask]
        return labelled_subset[np.argmin(distances)]
    else:
        return None


def match_fires_to_plumes(aod, fire_rows_plume, fire_cols_plume, max_mean_aods):

    # stores the final set of suitable labels as a mask
    label_store = np.zeros(aod.shape)
    singleton_fire_rows = []
    singlteon_fire_cols = []

    # iterate over all the fires
    for r, c, mma in zip(fire_rows_plume, fire_cols_plume, max_mean_aods):

        # consturct mask for plume
        ratio = mma / aod  # ratio is max mean aod in window near fire over image aod
        mask = ratio <= AOD_RATIO_LIMIT  # if ratio is smaller than limit then must be above local background
        mask = binary_erosion(mask)  # get rid of single mask points
        mask = binary_dilation(mask)  # bring back to full size

        # label the mask
        labelled_image = label(mask)

        # extract label for current fire
        label_for_fire = extract_label(labelled_image, r, c)
        if label_for_fire is None:
            continue  # no label within window then continue

        # extract labelled subsets around the other fires
        for i, (l, s) in enumerate(zip(fire_rows_plume, fire_cols_plume)):

            # dont compare to self
            if (l == r) & (s == c):
                continue

            # check labels for all other fires
            labelled_subset_for_another_fire = extract_label(labelled_image, l, s)
            if label_for_fire is None:
                continue  # no label within window then continue
            elif label_for_fire == labelled_subset_for_another_fire:
                break  # not a singleton fire so do not include

        # if we make it here then no other label matches that of the current fire
        # so we can store it in the label store (which is a binary mask)
        label_store[labelled_image == label_for_fire] = 1
        singleton_fire_rows.append(r)
        singlteon_fire_cols.append(c)

    return label_store, singleton_fire_rows, singlteon_fire_cols





def identify(aod, flag, lat, lon, date_to_find, fire_df):

    # first resample data to to remove VIIRS scanning effects
    aod, flag, lat, lon = resample(aod, flag, lat, lon)

    # subset fires to only those in the image and with certain FRP
    fire_subset_df = subset_fires_to_image(lat, lon, fire_df, date_to_find)

    # get mean fire cluster geographic locations
    mean_fire_geo_locs = mean_fire_position(fire_subset_df)

    # build sensor grid indexes
    image_rows, image_cols = grid_indexes(lat)

    # build a balltree for the sensor grid
    tree = build_balltree(lat, lon)

    # locate fires in sensor coordinates
    fire_rows, fire_cols, _ = locate_fires_in_image(tree, mean_fire_geo_locs, image_rows, image_cols)

    # determine those fires that are near to plumes
    fire_rows_plume, fire_cols_plume, max_mean_aods = locate_fires_near_plumes(aod, fire_rows, fire_cols)

    # find plumes with singleton fires (i.e. plumes that are not attached to another fire
    # that is burning more than 10km away)
    matched_plumes, matched_fire_rows, matched_fire_cols = match_fires_to_plumes(aod, fire_rows_plume,
                                                                                 fire_cols_plume, max_mean_aods)

    # now relabel and extract bounding boxes








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
    aod_flags = aod_h5['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['QF1'][:]
    lat = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Latitude'][:]
    lon = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Longitude'][:]

    flag = np.zeros(aod_flags.shape)
    for k, v in zip(['00', '01', '10', '11'], [0, 1, 2, 3]):
        mask = read_aod_mask(aod_flags, 0, 2, k)
        flag[mask] = v

    date_to_find = pd.Timestamp(2016, 7, 31)


    identify(aod, flag, lat, lon, fire_df)






if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()