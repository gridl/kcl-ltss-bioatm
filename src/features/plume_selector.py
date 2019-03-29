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

    return aod[min_y:max_y, min_x:max_x], hull_x, hull_y


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
    ax0_im = ax0.imshow(im, vmin=0)
    plt.colorbar(ax=ax0, mappable=ax0_im)
    ax0.plot(hull_x, hull_y, 'r--', lw=2)
    ax1.hist(plume_aod, bins=np.arange(0, 1, 0.02))
    plt.show()


def main():
    global keep

    #root = '/Users/dnf/Projects/kcl-ltss-bioatm/data'
    root = '/Volumes/INTENSO/kcl-ltss-bioatm'
    maiac_path = os.path.join(root, 'raw/plume_identification/maiac')
    hull_df_path = os.path.join(root, 'raw/plume_identification/dataframes/full/hull')
    aod_df_path = os.path.join(root, 'raw/plume_identification/dataframes/full/aod')
    log_path = os.path.join(root , 'raw/plume_identification/logs')

    hull_df_outpath = os.path.join(root, 'raw/plume_identification/dataframes/reduced/hull')
    aod_df_outpath = os.path.join(root, 'raw/plume_identification/dataframes/reduced/aod')

    # load in a image
    for hull_df_fname in os.listdir(hull_df_path):

        if '.csv' not in hull_df_fname:
            continue

        # check if file already processed
        try:
            with open(os.path.join(log_path, 'reduced_log.txt')) as log:
                if hull_df_fname + '\n' in log.read():
                    logger.info(hull_df_fname + ' already processed, continuing...')
                    continue
                else:
                    with open(os.path.join(log_path, 'reduced_log.txt'), 'a+') as log:
                        log.write(hull_df_fname + '\n')
        except IOError:
            with open(os.path.join(log_path, 'reduced_log.txt'), 'w+') as log:
                log.write(hull_df_fname + '\n')

        # create list to store IDs
        id_list = []

        # strip off filename
        base_name = hull_df_fname.replace('_extent.csv', '')
        hdf_fname = base_name + '.hdf'
        aod_df_fname = base_name + '_aod.csv'

        # load in the datasets
        hull_df = pd.read_csv(os.path.join(hull_df_path, hull_df_fname))
        aod_df = pd.read_csv(os.path.join(aod_df_path, aod_df_fname))

        hdf_file = SD(os.path.join(maiac_path, hdf_fname), SDC.READ)
        aod, lat, lon, ts = tools.read_modis_aod(hdf_file)

        # iterate over plumes in the dataframe
        for id in hull_df.id.unique():
            plume_df = hull_df[hull_df.id == id]

            # subset to plume and adjust hull
            plume_image, hull_x, hull_y = subset_plume(aod, plume_df)

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
            hull_df = hull_df[hull_df.id.isin(id_list)]
            aod_df = aod_df[aod_df.id.isin(id_list)]

            aod_df.to_csv(os.path.join(aod_df_outpath, aod_df_fname), index=False)
            hull_df.to_csv(os.path.join(hull_df_outpath, hull_df_fname), index=False)


if __name__ == "__main__":
    main()