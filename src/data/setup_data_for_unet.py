import logging
import os
import h5py
import scipy.misc as misc
import scipy.ndimage as ndimage
import numpy as np

import src.config.filepaths as fp

import matplotlib.pyplot as plt

"""
Sets up the data so that it is
ready to be ingested into an
Pytorch CNN

Iterate over each image/mask
find each unique plume then
extract an same size subset
around each plume, these will
form the training data.
"""

def read_h5(f):
    return h5py.File(f,  "r")


def main():
    win_size = 128
    bands = ['1', '4', '5', '6', '7', '10', '11', '12', '15']
    p_number = 0  # keeps track of all the plumes

    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr_reprojected_h5):

        if 'DS' in viirs_sdr_fname:
            continue

        try:
            sdr_path = os.path.join(fp.path_to_viirs_sdr_reprojected_h5, viirs_sdr_fname)
            mask_path = sdr_path.replace('/h5', '/mask').replace('_reproj.h5', '-mask.png')

            viirs_sdr = read_h5(sdr_path)
            viirs_mask = misc.imread(mask_path)[:, :, 0]  # just get top layer

            # to avoid border issues, pad the mask
            viirs_mask = np.pad(viirs_mask, win_size, 'constant', constant_values=0)

            # find plume locations
            labeled_array, num_features = ndimage.label(viirs_mask)
            centres = ndimage.center_of_mass(labeled_array, labels=labeled_array,
                                             index=range(1, num_features+1))
        except Exception as e:
            print(e)
            continue

        # plt.imshow(labeled_array)
        # plt.show()

        # centres returned y, x
        for c in centres:

            y = int(c[0])
            x = int(c[1])

            mask_sub = viirs_mask[y-win_size:y+win_size, x-win_size:x+win_size]

            # setup output array
            arr = np.zeros((win_size*2,  win_size*2, len(bands) + 1))
            for i, band in enumerate(bands):

                # extract band data and place in array
                ds = viirs_sdr['VIIRS-M'+band][:]
                ds = np.pad(ds, win_size, 'constant', constant_values=0)
                subset = ds[y-win_size:y+win_size, x-win_size:x+win_size]

                # normalise to between 0-1
                subset = (subset - np.min(subset)) / np.ptp(subset)
                if i == 0:
                    p_number += 1
                    plt.imshow(subset*mask_sub, cmap='gray')
                    plt.savefig(os.path.join(fp.path_to_cnn_grayscales, 'plume_{}.png'.format(p_number)),
                                bbox_inches='tight')
                arr[:, :, i] = subset
            arr[:, :, -1] = mask_sub

            output = os.path.join(fp.path_to_cnn_data_folder, 'plume_{}.npy'.format(p_number))
            with open(output, 'wb+') as fh:
                np.save(fh, arr, allow_pickle=False)



if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()






