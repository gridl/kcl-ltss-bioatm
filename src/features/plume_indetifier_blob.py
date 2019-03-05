import os
import re
import pyproj
import numpy as np

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import src.features.tools


def main():

    from pyhdf.SD import SD, SDC

    plot = True

    # setup paths
    # TODO update when all MAIAC data has been pulled
    root = '/Users/danielfisher/Projects/kcl-ltss-bioatm/data'
    maiac_path = os.path.join(root, 'raw/plume_identification/maiac')

    for maiac_fname in os.listdir(maiac_path):

        if 'MCD19A2.A2017210.h12v11.006.2018117231329' not in maiac_fname:
            continue
        # MCD19A2.A2017210.h12v11.006.2018117231329
        # 'MCD19A2.A2017255.h12v09.006.2018119143112'

        if '.hdf' not in maiac_fname:
            continue

        hdf_file = SD(os.path.join(maiac_path, maiac_fname), SDC.READ)

        aod, lat, lon = tools.read_modis_aod(hdf_file)

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