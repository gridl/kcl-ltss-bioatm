import os
import re
import logging

import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pandas as pd
from pyhdf.SD import SD, SDC


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# globals
keep = []

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


def subset_plume(aod, plume_df):
    buffer = 5

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

def press(event):
    global keep
    if event.key == '1':
        keep.append(True)
        plt.close()
    if event.key == '0':
        keep.append(False)
        plt.close()

def display_image(im, hull_x, hull_y):
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)
    ax.imshow(im)
    ax.plot(hull_x, hull_y, 'r--', lw=2)
    plt.show()


def main():
    global keep

    root = '/Users/dnf/Projects/kcl-ltss-bioatm/data'
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
        aod, lat, lon = read_modis_aod(hdf_file)

        # iterate over plumes in the dataframe
        for id in hull_df.id.unique():
            plume_df = hull_df[hull_df.id == id]

            # subset to plume and adjust hull
            plume_image, hull_x, hull_y = subset_plume(aod, plume_df)

            # display
            display_image(plume_image, hull_x, hull_y)

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