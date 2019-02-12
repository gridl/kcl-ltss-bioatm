import logging
import os
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def construct_dist_matrix():
    x = np.arange(-P_ID_WIN_SIZE, P_ID_WIN_SIZE+1)
    y = np.arange(-P_ID_WIN_SIZE, P_ID_WIN_SIZE+1)
    dx, dy = np.meshgrid(x, y)
    return np.sqrt(dx**2 + dy**2)

# Constants
MIN_FRP = 10  # Only fires greater than 10 MW are considered in clustering
CLUSTER_DIST = 10  # fires less than this distance apart (in KM) are clustered
THRESHOLD_SET = np.abs(np.arange(0, 1, 0.02) - 1)
MIN_RATIO_LIMIT = 5
P_ID_WIN_SIZE = 10  # plume identification window size in pix (half window e.g. for 21 use 10)
DISTANCE_MATRIX = construct_dist_matrix()  # used to determine the distance of a fire from a plume in pixels
MIN_PLUME_PIXELS = 50

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

    y_extent = lats.shape[0]
    x_extent = lats.shape[1]

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

            # check if either fire row or column is too close to edge of image
            row = sub_rows[sub_index]
            col = sub_cols[sub_index]
            if (row < P_ID_WIN_SIZE) | (row > y_extent - P_ID_WIN_SIZE):
                continue
            if (col < P_ID_WIN_SIZE) | (col > x_extent - P_ID_WIN_SIZE):
                continue

            # append row and col for  exact location
            fire_rows.append(row)
            fire_cols.append(col)

        except:
            continue

    return fire_rows, fire_cols


def labelled_mask_set(aod):
    """

    :param aod: the aod map
    :return: a set of masks based on different thresholds
    """
    threshold_mask = {}
    for t in THRESHOLD_SET:
        threshold_mask[t] = aod > t
    return threshold_mask


def extract_label(labelled_image, r, c):
    """
    What does this do?

    :param labelled_image:
    :param r:
    :param c:
    :return:
    """
    labelled_subset = labelled_image[r - P_ID_WIN_SIZE:r + P_ID_WIN_SIZE + 1,
                      c - P_ID_WIN_SIZE:c + P_ID_WIN_SIZE + 1]
    label_mask = labelled_subset != 0
    if label_mask.any():
        labelled_subset = labelled_subset[label_mask]
        distances = DISTANCE_MATRIX[label_mask]
        return labelled_subset[np.argmin(distances)]
    else:
        return None


def plume_extent_assessment(threshold_masks, fire_rows, fire_cols):
    """
    What does this do?

    :param threshold_masks:
    :param fire_rows:
    :param fire_cols:
    :return:
    """
    plume_size_records = np.zeros((len(threshold_masks), len(fire_cols)))

    for threshold_id, k in enumerate(threshold_masks):
        labelled = label(threshold_masks[k])
        for fire_id, (r, c) in enumerate(zip(fire_rows, fire_cols)):
            nearest_label_for_fire = extract_label(labelled, r, c)
            if nearest_label_for_fire is not None:
                plume_size = np.sum(labelled == nearest_label_for_fire)
                plume_size_records[threshold_id, fire_id] = plume_size

    return plume_size_records


def find_threshold_index(plume_extent_records):
    """

    :param plume_extent_records:
    :return:
    """
    threshold = []
    for fire_id, e_r in enumerate(plume_extent_records.T):

        null = e_r[:-1] == 0
        ratios = e_r[1:] / e_r[:-1]
        ratios[null] = np.nan

        # if no plume seen at all then set to none
        if np.all(np.isnan(ratios)):
            threshold.append(None)
            continue

        # now lets asses the various ratios
        argmax_ratio = np.nanargmax(ratios)
        argmin_ratio = np.nanargmin(ratios)

        # if max is the first non-nan entry then assume no plume
        if np.any(np.isnan(ratios)):
            if argmax_ratio == np.where(np.isnan(ratios))[0][-1] + 1:
                threshold.append(None)
                continue

        # if max is the last entry then assume no plume
        if argmax_ratio == ratios.size:
            threshold.append(None)

        # if max ratio is not significantly larger than min ratio
        elif ratios[argmax_ratio] / ratios[argmin_ratio] < MIN_RATIO_LIMIT:
            threshold.append(None)

        else:
            threshold.append(THRESHOLD_SET[argmax_ratio])

    return threshold


def extract_plume_roi(cluster_specific_threshold_index, threshold_masks,
                     fire_rows, fire_cols, lat, lon, aod):
    df_list = []

    for fire_id, t in enumerate(cluster_specific_threshold_index):

        if t == None:
            continue

        best_mask_for_fire_cluster = threshold_masks[t]
        labelled_mask = label(best_mask_for_fire_cluster)

        # find all labels associated with a fire
        all_plume_labels = []
        for r, c in zip(fire_rows, fire_cols):
            nearest_label_for_fire = extract_label(labelled_mask, r, c)

            if nearest_label_for_fire is not None:
                all_plume_labels.append(nearest_label_for_fire)
            else:
                all_plume_labels.append(None)


        # check if label is duplicated, and don't add if so
        label_for_fire = all_plume_labels[fire_id]
        appearences = np.sum(all_plume_labels == label_for_fire)
        if appearences > 2:
            continue

        # Check all regions in the image
        for region in regionprops(labelled_mask):
            if region.label == label_for_fire:

                # get rid of small plumes as likely not of use
                if region.area < MIN_PLUME_PIXELS:
                    continue

                # first get the plume mask
                plume_mask = labelled_mask == label_for_fire

                # now get the AOD
                plume_aod = aod[plume_mask]
                aod_mean = np.mean(plume_aod)
                aod_sd = np.std(plume_aod)

                # get bounding coordinates of the plume (for later use in Shapely to assign fires)
                y, x = np.where(plume_mask == 1)
                points = np.array(list(zip(y, x)))
                hull = ConvexHull(points)
                hull_indicies_y, hull_indicies_x = points[hull.vertices, 0], points[hull.vertices, 1]
                hull_lat = lat[hull_indicies_y, hull_indicies_x]
                hull_lon = lon[hull_indicies_y, hull_indicies_x]

                min_r, min_c, max_r, max_c = region.bbox

                # make the dataframe
                df = pd.DataFrame()
                area = int(region.area.copy())
                df['plume_pixel_extent'] = area
                df['plume_bounding_lats'] = [hull_lat]
                df['plume_bounding_lons'] = [hull_lon]
                df['plume_min_row'] = min_r
                df['plume_max_row'] = max_r
                df['plume_min_col'] = min_c
                df['plume_max_col'] = max_c
                df['plume_aod_mean'] = aod_mean
                df['plume_aod_sd'] = aod_sd
                df['bg_aod_level'] = t
                df['plume_pixel_extent'] = area

                df_list.append(df)

    scene_df = pd.concat(df_list)

    return scene_df


def identify(aod, lat, lon, date_to_find, fire_df):
    '''
    What does this do?

    :param aod:
    :param lat:
    :param lon:
    :param date_to_find:
    :param fire_df:
    :param type:  If type is 0 return plume bounding boxs, else return pixel indicies
    :return:
    '''

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

    # setup the plume masks set over the defined threshold
    threshold_masks = labelled_mask_set(aod)

    # iteratve over the set of plume masks and establish plume
    # extents for all fire clusters over the various thresholds
    plume_extents_for_thresholds = plume_extent_assessment(threshold_masks, fire_rows, fire_cols)

    # find  threshold index for each fire cluster that can be used to
    # index into the masks and the specific threshold used to generate
    # the mask
    cluster_specific_thresholds = find_threshold_index(plume_extents_for_thresholds)

    #
    plumes_df = extract_plume_roi(cluster_specific_thresholds, threshold_masks,
                                  fire_rows, fire_cols, lat, lon, aod)

    return plumes_df



def main():

    from pyhdf.SD import SD, SDC
    import src.features.tools as tools
    from datetime import datetime
    import time
    import h5py

    PIXEL_SIZE = 750  # size of resampled pixels in m for VIIRS data
    FILL_VALUE = np.nan  # resampling fill value
    path = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_id_test'

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

    logger.info('Running test with VIIRS AOD...')

    # data setup for testing with VIIRS
    viirs_aod_fname = 'IVAOT_npp_d20160822_t1702001_e1703242_b24974_c20181017161815133750_noaa_ops.h5'
    viirs_geo_fname = 'GMTCO_npp_d20160822_t1702001_e1703242_b24974_c20181019184439006772_noaa_ops.h5'
    viirs_aod_h5 = h5py.File(os.path.join(path, 'VIIRS', viirs_aod_fname), "r")
    viirs_geo_h5 = h5py.File(os.path.join(path, 'VIIRS', viirs_geo_fname), "r")

    viirs_fire_csv = 'fire_archive_V1_24485.csv'
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
    plume_df = identify(aod, lat, lon, date_to_find, viirs_fire_df)
    logger.info('...processed VIIRS.  Total time:' + str(time.clock() - t0))
    logger.info('')


    # # data setup for testing with MAIAC
    # logger.info('Running test with MAIAC AOD...')
    # maiac_aod_fname = 'MCD19A2.A2016235.h12v10.006.2018113135938.hdf'
    # hdf_file = SD(os.path.join(path, 'maiac', maiac_aod_fname), SDC.READ)
    #
    # aod, lat, lon = read_modis_aod(hdf_file)
    #
    # # lets use viirs fires again
    # viirs_fire_csv = 'fire_archive_V1_24485.csv'
    # viirs_fire_df = pd.read_csv(os.path.join(path, 'VIIRS', viirs_fire_csv))
    # viirs_fire_df['date_time'] = pd.to_datetime(viirs_fire_df['acq_date'])
    #
    # date_to_find = pd.Timestamp(2016, 8, 22)
    #
    # plume_df = identify(aod, lat, lon, date_to_find, viirs_fire_df)
    #
    # # TODO append filename / make new df for roi

    #
    # plt.close('all')
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(aod, cmap='gray')
    #
    # for roi in plume_roi_dict:
    #     d = plume_roi_dict[roi]
    #     plt.imshow(aod[d.minr:d.maxr, d.minc:d.maxc], cmap='gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.savefig(str(region.label) + '.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(aod, cmap='gray')
    for i, r in plume_df.iterrows():
        rect = mpatches.Rectangle((r.plume_min_col,
                                  r.plume_min_row),
                                  r.plume_max_col - r.plume_min_col,
                                  r.plume_max_row - r.plume_min_row,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    #plt.savefig('final_plumes.png', bbox_inches='tight')


if __name__ == "__main__":
    main()