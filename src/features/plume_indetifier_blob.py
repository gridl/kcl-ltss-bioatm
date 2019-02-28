import os
import re
import pyproj
import numpy as np

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


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


def main():

    from pyhdf.SD import SD, SDC

    plot = True

    # setup paths
    # TODO update when all MAIAC data has been pulled
    root = '/Volumes/INTENSO/kcl-ltss-bioatm/'
    maiac_path = os.path.join(root, 'raw/plume_identification/maiac')

    for maiac_fname in os.listdir(maiac_path):

        if 'MCD19A2.A2017255.h12v09.006.2018119143112' not in maiac_fname:
            continue
        # MCD19A2.A2017210.h12v11.006.2018117231329
        # 'MCD19A2.A2017255.h12v09.006.2018119143112'

        if '.hdf' not in maiac_fname:
            continue

        hdf_file = SD(os.path.join(maiac_path, maiac_fname), SDC.READ)

        aod, lat, lon = read_modis_aod(hdf_file)

        blobs_log = blob_log(aod, max_sigma=30, num_sigma=10, threshold=.1)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        blobs_dog = blob_dog(aod, max_sigma=30, threshold=.1)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        blobs_doh = blob_doh(aod, max_sigma=30, threshold=.01)

        blobs_list = [blobs_log, blobs_dog, blobs_doh]
        colors = ['yellow', 'lime', 'red']
        titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
                  'Determinant of Hessian']
        sequence = zip(blobs_list, colors, titles)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        for idx, (blobs, color, title) in enumerate(sequence):
            ax[idx].set_title(title)
            ax[idx].imshow(aod, interpolation='nearest')
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax[idx].add_patch(c)
            ax[idx].set_axis_off()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()