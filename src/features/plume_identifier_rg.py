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


def generate_mask_dict(aod):
    """
    :param aod: the aod map
    :return: a set of masks based on different thresholds
    """
    masks_dict = {}
    for t in THRESHOLD_SET:
        masks_dict[t] = aod > t
    return masks_dict


def extract_label(labelled_image, r, c):
    """
    Find the nearest label in a labelled image for a given fire location
    within some window

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


def find_plume_extents(masks_dict, fire_rows, fire_cols):
    """
    For each fire the size in pixels of the most nearby (must be within
     the maximum distance of pixels) labelled raised AOD is recorded.  This
     is repeated across all different thresholds, from this we can see how the
     size of the plume changes with the threshold.  This can be used in the next
     step to determine the best background value.

    :param masks_dict: masks derived from AODs at a range of different treshold values
    :param fire_rows: row locations of aggregated fires
    :param fire_cols: columne locations of aggregated fires
    :return: the extent of the plume for a given fire across the various different thresholds
    """
    plume_extents = np.zeros((len(masks_dict), len(fire_cols)))

    for mask_index, mask_key in enumerate(masks_dict):
        labelled_mask = label(masks_dict[mask_key])
        for fire_index, (r, c) in enumerate(zip(fire_rows, fire_cols)):
            nearest_plume_label_for_fire = extract_label(labelled_mask, r, c)
            if nearest_plume_label_for_fire is not None:
                plume_size = np.sum(labelled_mask == nearest_plume_label_for_fire)
                plume_extents[mask_index, fire_index] = plume_size
    return plume_extents


def find_threshold_index(plume_extents_across_all_fires):
    """

    :param plume_extents_across_all_fires:
    :return:
    """
    best_threshold_index = []
    for fire_id, extents in enumerate(plume_extents_across_all_fires.T):

        # make sure that all divide by zero instances are dealt with
        # by setting to nan
        null = extents[:-1] == 0
        extent_ratios = extents[1:] / extents[:-1]
        extent_ratios[null] = np.nan

        # if all threshold values are nan then no plume
        if np.all(np.isnan(extent_ratios)):
            best_threshold_index.append(None)
            continue

        # now lets asses the various ratios
        argmax_ratio = np.nanargmax(extent_ratios)
        argmin_ratio = np.nanargmin(extent_ratios)

        # if max is the first non-nan entry then assume no plume
        if np.any(np.isnan(extent_ratios)):
            if argmax_ratio == np.where(np.isnan(extent_ratios))[0][-1] + 1:
                best_threshold_index.append(None)
                continue

        # if max is the last entry then assume no plume
        if argmax_ratio == extent_ratios.size:
            best_threshold_index.append(None)

        # if max ratio is not significantly larger than min ratio
        elif extent_ratios[argmax_ratio] / extent_ratios[argmin_ratio] < MIN_RATIO_LIMIT:
            best_threshold_index.append(None)

        else:
            best_threshold_index.append(argmax_ratio)

    return best_threshold_index


def find_plume_mask(threshold_masks, index, fire_rows, fire_cols, fire_id):

    mask = threshold_masks[THRESHOLD_SET[index]]

    # label the two masks
    labelled_mask = label(mask)

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
    if appearences >= 2:
        return None, None

    # Check all regions in the image
    for region in regionprops(labelled_mask):
        if region.label == label_for_fire:

            # get rid of small plumes as likely not of use
            if region.area < MIN_PLUME_PIXELS:
                continue

            # first get the plume mask
            return labelled_mask == label_for_fire, region

    # if we get there then no plume associated with fire
    return None, None


def extract_plume_roi(best_threshold_index, threshold_masks,
                      fire_rows, fire_cols, lat, lon, aod):
    """

    :param best_threshold_index:
    :param threshold_masks:
    :param fire_rows:
    :param fire_cols:
    :param lat:
    :param lon:
    :param aod:
    :return:
    """
    aod_df_list = []
    hull_lats = []
    hull_lons = []
    hull_ids = []
    id = int(0)

    for fire_id, threshold_index in enumerate(best_threshold_index):

        # if no threshold index indentified then move on
        if threshold_index == None:
            continue

        # find the plume associated with the fire
        plume_mask_a, region_a = find_plume_mask(threshold_masks, threshold_index, fire_rows, fire_cols, fire_id)
        plume_mask_b, region_b = find_plume_mask(threshold_masks, threshold_index-1, fire_rows, fire_cols, fire_id)

        # select the right plume mask
        if plume_mask_a is None and plume_mask_b is None:
            continue
        if plume_mask_a is not None and plume_mask_b is not None:
            if np.sum(plume_mask_a) > np.sum(plume_mask_b):
                plume_mask = plume_mask_a
                region = region_a
            else:
                plume_mask = plume_mask_b
                region = region_b
        elif plume_mask_a is None:
            plume_mask = plume_mask_b
            region = region_b
        else:
            plume_mask = plume_mask_a
            region = region_a


        # now get the AOD
        plume_aod = aod[plume_mask]
        aod_mean = np.mean(plume_aod)
        aod_sd = np.std(plume_aod)

        # get bounding coordinates of the plume (for later use in Shapely to assign fires)
        y, x = np.where(plume_mask == 1)
        points = np.array(list(zip(y, x)))
        hull = ConvexHull(points)
        hull_indicies_y, hull_indicies_x = points[hull.vertices, 0], points[hull.vertices, 1]
        hull_lats.extend(lat[hull_indicies_y, hull_indicies_x])
        hull_lons.extend(lon[hull_indicies_y, hull_indicies_x])
        hull_ids.extend(np.ones(hull_indicies_y.size) * id)

        # get the bounding box
        min_r, min_c, max_r, max_c = region.bbox

        # make the aod dataframe
        dd = {'plume_pixel_extent': int(region.area.copy()),
              'plume_min_row': min_r,
              'plume_max_row': max_r,
              'plume_min_col': min_c,
              'plume_max_col': max_c,
              'plume_aod_mean': aod_mean,
              'plume_aod_sd': aod_sd,
              'bg_aod_level': threshold_index,
              'id': id}
        aod_df = pd.DataFrame()
        aod_df = aod_df.append(dd, ignore_index=True)
        aod_df_list.append(aod_df)

        # update the id if we find a region with a fire
        id += 1

    aod_scene_df = pd.concat(aod_df_list)

    # Create the extent dataframe
    extents = [('id', hull_ids),
               ('hull_lats', hull_lats),
               ('hull_lons', hull_lons)]
    extent_scene_df = pd.DataFrame.from_items(extents)

    return aod_scene_df, extent_scene_df


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
    masks_dict = generate_mask_dict(aod)

    # iteratve over the set of plume masks and establish plume
    # extents for all fire clusters over the various thresholds
    plume_extents_across_thresholds = find_plume_extents(masks_dict, fire_rows, fire_cols)

    # find  threshold index for each fire cluster that can be used to
    # index into the masks and the specific threshold used to generate
    # the mask
    threshold_index_for_fires = find_threshold_index(plume_extents_across_thresholds)

    #
    aod_df, extent_df = extract_plume_roi(threshold_index_for_fires, masks_dict,
                                          fire_rows, fire_cols, lat, lon, aod)

    return aod_df, extent_df



def main():

    from pyhdf.SD import SD, SDC
    import src.features.tools as tools
    from datetime import datetime
    import time
    import h5py

    # PIXEL_SIZE = 750  # size of resampled pixels in m for VIIRS data
    # FILL_VALUE = np.nan  # resampling fill value
    #path = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/plume_id_test'
    path = '/Users/danielfisher/Projects/kcl-ltss-bioatm/data/raw/plume_id_test'

    #
    # def resample(img, lat, lon, null_value=0):
    #     resampler = tools.utm_resampler(lat, lon, PIXEL_SIZE)
    #
    #     lonlats = resampler.area_def.get_lonlats()
    #     lat_grid = lonlats[1]
    #     lon_grid = lonlats[0]
    #
    #     mask = img < null_value
    #     masked_lats = np.ma.masked_array(resampler.lats, mask)
    #     masked_lons = np.ma.masked_array(resampler.lons, mask)
    #     img = resampler.resample_image(img, masked_lats, masked_lons, fill_value=FILL_VALUE)
    #     return img, lat_grid, lon_grid
    #
    # # define paths to data for testing
    #
    # logger.info('Running test with VIIRS AOD...')
    #
    # # data setup for testing with VIIRS
    # viirs_aod_fname = 'IVAOT_npp_d20160822_t1702001_e1703242_b24974_c20181017161815133750_noaa_ops.h5'
    # viirs_geo_fname = 'GMTCO_npp_d20160822_t1702001_e1703242_b24974_c20181019184439006772_noaa_ops.h5'
    # viirs_aod_h5 = h5py.File(os.path.join(path, 'VIIRS', viirs_aod_fname), "r")
    # viirs_geo_h5 = h5py.File(os.path.join(path, 'VIIRS', viirs_geo_fname), "r")
    #
    # viirs_fire_csv = 'fire_archive_V1_24485.csv'
    # viirs_fire_df = pd.read_csv(os.path.join(path, 'VIIRS', viirs_fire_csv))
    # viirs_fire_df['date_time'] = pd.to_datetime(viirs_fire_df['acq_date'])
    #
    # viirs_aod = viirs_aod_h5['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    # viirs_lat = viirs_geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Latitude'][:]
    # viirs_lon = viirs_geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Longitude'][:]
    # logger.info('...Loaded VIIRS data')
    #
    #
    # # strip time for viirs fname
    # viirs_dt = datetime.strptime(re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_aod_fname).group(), 'd%Y%m%d_t%H%M%S')
    # date_to_find = pd.Timestamp(viirs_dt.year, viirs_dt.month, viirs_dt.day)
    #
    # # need to resample VIIRS for the image processing parts
    # aod, lat, lon = resample(viirs_aod, viirs_lat, viirs_lon)
    # logger.info('...resampled VIIRS data')
    # t0 = time.clock()
    # plume_df = identify(aod, lat, lon, date_to_find, viirs_fire_df)
    # logger.info('...processed VIIRS.  Total time:' + str(time.clock() - t0))
    # logger.info('')


    # data setup for testing with MAIAC
    logger.info('Running test with MAIAC AOD...')
    maiac_aod_fname = 'MCD19A2.A2016235.h12v10.006.2018113135938.hdf'
    maiac_output_fname = maiac_aod_fname[:-4]
    hdf_file = SD(os.path.join(path, 'maiac', maiac_aod_fname), SDC.READ)

    aod, lat, lon = read_modis_aod(hdf_file)

    # lets use viirs fires again
    viirs_fire_csv = 'fire_archive_V1_24485.csv'
    viirs_fire_df = pd.read_csv(os.path.join(path, 'VIIRS', viirs_fire_csv))
    viirs_fire_df['date_time'] = pd.to_datetime(viirs_fire_df['acq_date'])

    date_to_find = pd.Timestamp(2016, 8, 22)

    aod_df, extent_df = identify(aod, lat, lon, date_to_find, viirs_fire_df)

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
    for i, r in aod_df.iterrows():
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

    # dump the dfs as a csv using the maiac filename
    aod_fname = maiac_output_fname + '_aod.csv'
    extent_fname = maiac_output_fname + '_extent.csv'

    aod_df.to_csv(os.path.join(path, aod_fname))
    extent_df.to_csv(os.path.join(path, extent_fname))


if __name__ == "__main__":
    main()