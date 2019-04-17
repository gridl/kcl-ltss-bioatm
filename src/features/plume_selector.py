import os
import re
import logging

import matplotlib
matplotlib.use("TKAgg")

import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pandas as pd
from pyhdf.SD import SD, SDC
from scipy.spatial import Delaunay

import src.features.tools as tools


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# globals
keep = []


def remove_duplicated_plumes(plume_df):

    # we are having problems joining on the datetime column so lets hack around it
    unique_dt = plume_df.datetime.unique()
    dt_dict_forward = {}
    dt_dict_inverse = {}
    for i, dt in enumerate(unique_dt):
        dt_dict_forward[dt] = i
        dt_dict_inverse[i] = dt
    plume_df.replace({'datetime': dt_dict_forward}, inplace=True)

    # for each plume calculate centroid
    grouped_df = plume_df.groupby(['id', 'datetime']).agg({'hull_lats': np.mean, 'hull_lons': np.mean}).reset_index()

    # round centroids to nearest 0.01 deg and drop any duplicates
    non_duplicates_df = grouped_df.round({"hull_lats": 3,
                                         "hull_lons": 3}).drop_duplicates(['datetime',
                                                                           'hull_lats',
                                                                           'hull_lons'], keep='first')
    # inner join to strip duplicated data
    non_duplicates_df.drop(['hull_lats', 'hull_lons'], axis=1, inplace=True)
    plume_df = pd.merge(plume_df, non_duplicates_df, on=['id','datetime'], how='inner')
    plume_df.replace({'datetime': dt_dict_inverse}, inplace=True)
    return plume_df



def subset_plume(aod, plume_df):


    buffer = 40
    min_x = plume_df.hull_x.min()
    max_x = plume_df.hull_x.max()
    min_y = plume_df.hull_y.min()
    max_y = plume_df.hull_y.max()

    hull_x = plume_df.hull_x.values
    hull_y = plume_df.hull_y.values

    if min_x - buffer < 0:
        min_x = 0
    else:
        hull_x = hull_x - min_x + buffer
        min_x = min_x - buffer

    if min_y - buffer < 0:
        min_y = 0
    else:
        hull_y = hull_y - min_y + buffer
        min_y = min_y - buffer


    # check for image edges
    max_x = aod.shape[1] if max_x + buffer > aod.shape[1] else max_x + buffer
    max_y = aod.shape[0] if max_y + buffer > aod.shape[0] else max_y + buffer

    if np.isnan([min_y, max_y, min_x, max_x]).any():
        return None, None, None
    else:
        return aod[int(min_y):int(max_y), int(min_x):int(max_x)], hull_x, hull_y


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def find_plume_aod(plume_image, hull_x, hull_y):

    y = np.arange(plume_image.shape[0])
    x = np.arange(plume_image.shape[1])
    xx, yy = np.meshgrid(y, x)
    xx = xx.flatten()
    yy = yy.flatten()

    im_coords = np.vstack((xx, yy)).T
    hull = np.vstack((hull_x, hull_y)).T

    mask = in_hull(im_coords, hull)

    plume_aod = plume_image[yy[mask], xx[mask]]
    return plume_aod

def press(event):
    global keep
    if event.key == '1':
        keep.append(True)
        plt.close()
    if event.key == '0':
        keep.append(False)
        plt.close()

def display_image(im, hull_x, hull_y, plume_aod):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,5))
    fig.canvas.mpl_connect('key_press_event', press)
    ax0_im = ax0.imshow(im, vmin=0, vmax=np.max(plume_aod))
    plt.colorbar(ax=ax0, mappable=ax0_im)
    ax0.plot(hull_x, hull_y, 'r--', lw=2,)
    ax1.hist(plume_aod, bins=np.arange(0, 1, 0.02))
    plt.show()


def main():

    plume=True

    global keep

    #root = '/Users/dnf/Projects/kcl-ltss-bioatm/data'
    root = '/Volumes/INTENSO/kcl-ltss-bioatm'
    maiac_path = os.path.join(root, 'raw/plume_identification/maiac')
    hull_df_path = os.path.join(root, 'raw/plume_identification/dataframes/full/hull')

    if plume:
        log_path = os.path.join(root , 'raw/plume_identification/logs')
        log_fname = 'plume_reduced_log.txt'
        hull_df_outpath = os.path.join(root, 'raw/plume_identification/dataframes/reduced/plume/hull')
    else:
        log_path = os.path.join(root , 'raw/plume_identification/logs')
        log_fname = 'not_plume_reduced_log.txt'
        hull_df_outpath = os.path.join(root, 'raw/plume_identification/dataframes/reduced/not_plume/hull')

    # load in a image
    for hull_df_fname in os.listdir(hull_df_path):

        if '.csv' not in hull_df_fname:
            continue

        # check if file already processed
        try:
            with open(os.path.join(log_path, log_fname)) as log:
                if hull_df_fname + '\n' in log.read():
                    logger.info(hull_df_fname + ' already processed, continuing...')
                    continue
                else:
                    with open(os.path.join(log_path, log_fname), 'a+') as log:
                        log.write(hull_df_fname + '\n')
        except IOError:
            with open(os.path.join(log_path, log_fname), 'w+') as log:
                log.write(hull_df_fname + '\n')

        # strip off filename
        base_name = hull_df_fname.replace('_extent.csv', '')
        hdf_fname = base_name + '.hdf'

        # load in the datasets
        hull_df = pd.read_csv(os.path.join(hull_df_path, hull_df_fname))

        # de-duplicate plumes in DF
        hull_df = remove_duplicated_plumes(hull_df)

        hdf_file = SD(os.path.join(maiac_path, hdf_fname), SDC.READ)
        aod_dict, lat, lon = tools.read_modis_aod(hdf_file)

        # iterate over plumes in the dataframe
        dt_df_list = []
        for dt in hull_df.datetime.unique():

            # create list to store IDs for date
            id_list = []

            dt_df = hull_df[hull_df.datetime == dt]
            for id in dt_df.id.unique():
                plume_df = dt_df[dt_df.id == id]

                # subset to plume and adjust hull
                plume_image, hull_x, hull_y = subset_plume(aod_dict[dt], plume_df)

                if plume_image is None:
                    continue

                # get aod for points inside the convex hull
                in_plume_aod = find_plume_aod(plume_image, hull_x, hull_y)

                # check if largest bin is 0, if so dont bother checking it
                h = np.histogram(in_plume_aod, bins=np.arange(0, 1, 0.02))
                if np.argmax((h[0])) == 0:
                    continue

                # display
                display_image(plume_image, hull_x, hull_y, in_plume_aod)

                # if keep
                if keep[0]:
                    # store the valid ID
                    id_list.append(id)
                keep.pop()

            if not id_list:
                continue
            else:
                # reduce the dataframes to only valid IDs and then store for further processing
                dt_df = dt_df[dt_df.id.isin(id_list)]
                dt_df_list.append(dt_df)
        if not dt_df_list:
            continue
        else:
            output_df = pd.concat(dt_df_list)
            output_df.to_csv(os.path.join(hull_df_outpath, hull_df_fname), index=False)


if __name__ == "__main__":
    main()