import logging
import os
import re
from datetime import datetime


from pyhdf.SD import SD, SDC
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation
from sklearn.cluster import DBSCAN

import src.config.filepaths as filepaths


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
AOD_MIN_LIMIT = 0.2  # anything above this that is associated with a fire is assumed to be a plume UPDATE WITH CLIM?
DISTANCE_MATRIX = construct_dist_matrix()  # used to determine the distance of a fire from a plume in pixels

def read_modis_aod(hdf_file):
    # Read dataset.
    aod = hdf_file.select('Optical_Depth_055')[0, :, :] * 0.001  # aod scaling factor


    aod[aod < 0] = 0  # just get rid of the filled values for now

    # Read global attribute.
    fattrs = hdf_file.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]

    # Construct the grid.  The needed information is in a global attribute
    # called 'StructMetadata.0'.  Use regular expressions to tease out the
    # extents of the grid.
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                                      (?P<upper_left_x>[+-]?\d+\.\d+)
                                      ,
                                      (?P<upper_left_y>[+-]?\d+\.\d+)
                                      \)''', re.VERBOSE)
    match = ul_regex.search(gridmeta)
    x0 = np.float(match.group('upper_left_x'))
    y0 = np.float(match.group('upper_left_y'))

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                                      (?P<lower_right_x>[+-]?\d+\.\d+)
                                      ,
                                      (?P<lower_right_y>[+-]?\d+\.\d+)
                                      \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = np.float(match.group('lower_right_x'))
    y1 = np.float(match.group('lower_right_y'))
    ny, nx = aod.shape
    xinc = (x1 - x0) / nx
    yinc = (y1 - y0) / ny

    x = np.linspace(x0, x0 + xinc * nx, nx)
    y = np.linspace(y0, y0 + yinc * ny, ny)
    xv, yv = np.meshgrid(x, y)

    # In basemap, the sinusoidal projection is global, so we won't use it.
    # Instead we'll convert the grid back to lat/lons.
    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    lon, lat = pyproj.transform(sinu, wgs84, xv, yv)

    return aod, lat, lon


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

        if np.all(np.isnan(aod_for_window)):
            continue

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


    mask = aod >= AOD_MIN_LIMIT  # update using climatological data?  Or ML approach? Pros and Cons.

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
        elif (labelled_image == l).sum() > 10000:  # get rid of unreasonably large plumes
            labelled_image[labelled_image == l] = 0
        elif (labelled_image == l).sum() < 100:  # get rid of unreasonably large plumes
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
    '''

    :param aod:
    :param lat:
    :param lon:
    :param date_to_find:
    :param fire_df:
    :param type:  If type is 0 return plume bounding boxs, else return pixel indicies
    :return:
    '''

    try:

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

        # extract plume bounding boxes
        plume_roi_dict = extract_plumes(plume_image)
        logger.info('...boudning boxes for single fire plumes extracted')

        # extract background bounding boxes


        return plume_roi_dict, plume_image

    except:
        return None, None



def main():

    plot = True

    # setup paths
    # TODO update when all MAIAC data has been pulled
    maiac_path = '/Users/dnf/Downloads/maiac'
    log_path = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_identification/logs'
    aod_df_outpath = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_identification/dataframes/aod'
    hull_df_outpath = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_identification/dataframes/hull'
    plot_outpath = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_identification/plots_basic'

    # load in VIIRS fires for plume detection purposes
    viirs_fire_csv_fname = 'viirs_americas_201707_201709.csv'
    viirs_fire_df = pd.read_csv(os.path.join(filepaths.path_to_fire, viirs_fire_csv_fname))
    viirs_fire_df['date_time'] = pd.to_datetime(viirs_fire_df['acq_date'])

    for maiac_fname in os.listdir(maiac_path):

        if '.hdf' not in maiac_fname:
            continue

        if 'MCD19A2.A2017255.h12v09.006.2018119143112' not in maiac_fname:
            continue

        # check if MAIAC file has already been processed
        maiac_output_fname = maiac_fname[:-4]
        aod_fname = maiac_output_fname + '_aod.csv'
        hull_fname = maiac_output_fname + '_extent.csv'

        # check if file already processed
        try:
            with open(os.path.join(log_path, 'maiac.log'), 'a+') as log:
                if maiac_fname in log.read():
                    logger.info(maiac_output_fname + ' already processed, continuing...')
                    continue
                else:
                    log.write(maiac_fname+'\n')
        except IOError:
            with open(os.path.join(log_path, 'maiac.log'), 'w+') as log:
                log.write(maiac_fname+'\n')


        hdf_file = SD(os.path.join(maiac_path, maiac_fname), SDC.READ)
        aod, lat, lon = read_modis_aod(hdf_file)

        date_to_find = pd.Timestamp(datetime.strptime(maiac_fname.split('.')[1][1:], '%Y%j'))


        plume_roi_dict, labelled_image = identify(aod, lat, lon, date_to_find, viirs_fire_df)

        if labelled_image is None:
            continue

        if not plume_roi_dict:
            continue

        if plot:
            plot_fname = maiac_output_fname + '_plot.png'

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(aod, cmap='gray')
            for region in regionprops(labelled_image):
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(plot_outpath, plot_fname), bbox_inches='tight')




if __name__ == "__main__":
    main()