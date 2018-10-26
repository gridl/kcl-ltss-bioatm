import logging
import os
import re
from datetime import datetime

import pandas as pd
import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation

from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def construct_dist_matrix():
    x = np.arange(-P_ID_WIN_SIZE, P_ID_WIN_SIZE+1)
    y = np.arange(-P_ID_WIN_SIZE, P_ID_WIN_SIZE+1)
    dx, dy = np.meshgrid(x, y)
    return np.sqrt(dx**2 + dy**2)

# Constants
MIN_FRP = 10  # Only fires greatert han 10 MW are considered in clustering
CLUSTER_DIST = 10  # fires less than this distance apart (in KM) are clustered
P_ID_WIN_SIZE = 10  # plume identification window size in pix (half window e.g. for 21 use 10)
AOD_RATIO_LIMIT = 3  # if ratio is greater than this then assume a plume (also
AOD_MIN_LIMIT = 0.25  # anything above this that is associated with a fire is assumed to be a plume UPDATE WITH CLIM?
DISTANCE_MATRIX = construct_dist_matrix()  # used to determine the distance of a fire from a plume in pixels


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

    rows = np.arange(lat.shape[0])
    cols = np.arange(lat.shape[1])
    cols, rows = np.meshgrid(cols, rows)
    return rows, cols


def build_balltree(lat, lon):

    lat_in_rads = np.deg2rad(lat.flatten())
    lon_in_rads = np.deg2rad(lon.flatten())

    array_lat_lon = np.dstack([lat_in_rads, lon_in_rads])[0]
    return BallTree(array_lat_lon, metric='haversine')


def locate_fires_in_image(tree, fire_pos, rows, cols):
    point_locations = np.dstack((np.deg2rad(fire_pos.latitude.values),
                                 np.deg2rad(fire_pos.longitude.values))).squeeze()
    distance, index = tree.query(point_locations, k=1)
    return rows[index], cols[index], np.rad2deg(distance)


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

    fire_rows = []
    fire_cols = []

    for fire_lat, fire_lon in zip(fire_coords.latitude.values, fire_coords.longitude.values):

        # get approximate location using
        index = np.argmin(np.linalg.norm(np.array([fire_lat, fire_lon])-np.dstack([lats, lons]), axis=2))
        r = int(index / lats.shape[1])
        c = int(index % lats.shape[1])

        # extract window around approx location (from geo and index)
        rad = 7
        sub_lats = lats[r-rad:r+rad+1, c-rad:c+rad+1]
        sub_lons = lons[r-rad:r+rad+1, c-rad:c+rad+1]
        sub_rows = rows[r-rad:r+rad+1, c-rad:c+rad+1]
        sub_cols = cols[r-rad:r+rad+1, c-rad:c+rad+1]

        # find exact loc using haversine distance
        sub_index = np.argmin(haversine(fire_lon, fire_lat, sub_lons, sub_lats))
        sub_r = int(sub_index / (rad*2+1))
        sub_c = int(sub_index % (rad*2+1))

        # append row and col for  exact location
        fire_rows.append(sub_rows[sub_r, sub_c])
        fire_cols.append(sub_cols[sub_r, sub_c])

    return fire_rows, fire_cols



def locate_fires_near_plumes(aod, fire_rows, fire_cols):

    r_near_plume = []
    c_near_plume = []

    for r, c in zip(fire_rows, fire_cols):


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

        return r_near_plume, c_near_plume


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


def locate_plumes_with_fires(aod, fire_rows_plume, fire_cols_plume):
    '''
    For each fire check its nearest label.  If a label appears
    more than once it is associated with multiple fires, so
    get rid of it.
    '''

    mask = aod >= 0.25  # update using climatological data?  Or ML approach? Pros and Cons.
    mask = binary_erosion(mask)
    mask = binary_dilation(mask)

    # label the mask
    labelled_image = label(mask)

    # find all labels associated with a fire
    all_plume_labels = []
    for r, c in zip(fire_rows_plume, fire_cols_plume):
        nearest_label_for_fire = extract_label(labelled_image, r, c)

        if nearest_label_for_fire is not None:
            all_plume_labels.append(nearest_label_for_fire)

    # drop any labels that are duplicated
    final_plume_labels = []
    for l in all_plume_labels:
        appearences = np.sum(all_plume_labels == l)
        if appearences < 2:
            final_plume_labels.append(l)

    # update labelled image
    for l in np.unique(labelled_image):
        if l not in final_plume_labels:
            labelled_image[labelled_image == l] = 0

    return labelled_image


def extract_plumes(plume_image):
    plume_dict = {}
    labelled_image = label(plume_image)
    for region in regionprops(labelled_image):
        min_r, min_c, max_r, max_c = region.bbox
        plume_dict[region.label] = {'min_r': min_r, 'min_c': min_c, 'max_r': max_r, 'max_c': max_c}
    return plume_dict


def identify(aod, lat, lon, date_to_find, fire_df):

    # subset fires to only those in the image and with certain FRP
    fire_subset_df = subset_fires_to_image(lat, lon, fire_df, date_to_find)
    logger.info('...Extracted fires for image roi')

    # get mean fire cluster geographic locations
    mean_fire_geo_locs = mean_fire_position(fire_subset_df)
    logger.info('...clustered fires')

    # build sensor grid indexes
    image_rows, image_cols = grid_indexes(lat)
    logger.info('...built grid indexes to assign fires to image grid')

    # locate fires in sensor coordinates
    fire_rows, fire_cols = locate_fire_in_image(mean_fire_geo_locs, lat, lon, image_rows, image_cols)
    logger.info('...assigned fires to image grid')

    # determine those fires that are near to plumes
    fire_rows_plume, fire_cols_plume = locate_fires_near_plumes(aod, fire_rows, fire_cols)
    logger.info('...reduced fires to those associated with plumes')

    # find plumes with singleton fires (i.e. plumes that are not attached to another fire
    # that is burning more than 10km away)
    plume_image = locate_plumes_with_fires(aod, fire_rows_plume, fire_cols_plume)
    logger.info('...reduced plumes to only those matched to a single fire')

    # extract bounding boxes
    plume_roi_dict = extract_plumes(plume_image)
    logger.info('...boudning boxes for single fire plumes extracted')


    return plume_roi_dict


def main():

    import src.data.tools as tools
    import h5py
    import time

    PIXEL_SIZE = 750  # size of resampled pixels in m for VIIRS data
    FILL_VALUE = np.nan  # resampling fill value


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

    # define paths to data for testing
    path = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_id_test'

    logger.info('Running test with VIIRS AOD...')

    # data setup for testing with VIIRS
    viirs_aod_fname = 'IVAOT_npp_d20150706_t0603097_e0604339_b19108_c20150706075934735034_noaa_ops.h5'
    viirs_geo_fname = 'GMTCO_npp_d20150706_t0603097_e0604339_b19108_c20171126104946623808_noaa_ops.h5'
    viirs_aod_h5 = h5py.File(os.path.join(path, 'VIIRS', viirs_aod_fname), "r")
    viirs_geo_h5 = h5py.File(os.path.join(path, 'VIIRS', viirs_geo_fname), "r")

    viirs_fire_csv = 'fire_archive_V1_26373.csv'
    viirs_fire_df = pd.read_csv(os.path.join(path, 'VIIRS', viirs_fire_csv))
    viirs_fire_df['date_time'] = pd.to_datetime(viirs_fire_df['acq_date'])

    viirs_aod = viirs_aod_h5['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    viirs_lat = viirs_geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Latitude'][:]
    viirs_lon = viirs_geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Longitude'][:]
    logger.info('...Loaded VIIRS data')


    # strip time for viirs fname
    viirs_dt = datetime.strptime(re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_aod_fname).group(), 'd%Y%m%d_t%H%M%S')
    date_to_find = pd.Timestamp(viirs_dt.year, viirs_dt.month, viirs_dt.day)

    # need to resample VIIRS for the image processing parts
    aod, lat, lon = resample(viirs_aod, viirs_lat, viirs_lon)
    logger.info('...resampled VIIRS data')
    t0 = time.clock()
    viirs_plume_dict = identify(aod, lat, lon, date_to_find, viirs_fire_df)
    logger.info('...processed VIIRS.  Total time:' + str(time.clock() - t0))
    logger.info('')


    # data setup for testing with MAIAC


    # data setup for testing with S5P







if __name__ == "__main__":
    main()