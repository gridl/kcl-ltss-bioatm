import logging
import os

import numpy as np
import scipy.misc as misc
import h5py

import src.config.filepaths as fp


def main():

    bands = ['1', '4', '5', '6', '7', '10', '11', '12', '15']

    # set up the array to hold the data
    samples_estimate = 50000000
    output_array = np.zeros((samples_estimate, len(bands)+1))  # +1 for mask
    current_line = 0  # tracks position in numpy array

    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr_reprojected_h5):

        if 'DS' in viirs_sdr_fname:
            continue

        print(viirs_sdr_fname)

        try:
            sdr_path = os.path.join(fp.path_to_viirs_sdr_reprojected_h5, viirs_sdr_fname)
            mask_path = sdr_path.replace('/h5', '/mask_sub_plume').replace('_reproj.h5', '-mask.png')
            bg_mask_path = sdr_path.replace('/h5', '/mask_bg').replace('_reproj.h5', '-bg_mask.png')

            sdr = h5py.File(sdr_path,  "r")
            smoke_mask = misc.imread(mask_path)[:, :, 0]  # just get top layer
            bg_mask = misc.imread(bg_mask_path)[:, :, 0]
        except Exception as e:
            print(e)
            continue

        # extract indicies
        smoke_indicies = np.where(smoke_mask == 255)
        bg_indicies = np.where(bg_mask == 255)

        # set up the temp array
        n_smoke_samples = len(smoke_indicies[0])
        n_bg_samples = len(bg_indicies[0])

        temp_array = np.zeros((int(n_smoke_samples + n_bg_samples), len(bands)+1))
        temp_array[:n_smoke_samples, -1] = 1  # set smoke flag

        new_line = current_line + (n_smoke_samples + n_bg_samples)

        for i, band in enumerate(bands):
            # extract band data and place in array
            ds = sdr['VIIRS-M' + band][:]

            smoke_dn = ds[smoke_indicies[0], smoke_indicies[1]]
            bg_dn = ds[bg_indicies[0], bg_indicies[1]]

            temp_array[:n_smoke_samples, i] = smoke_dn
            temp_array[n_smoke_samples:, i] = bg_dn

        # insert data into output array and update line
        output_array[current_line:new_line, :] = temp_array
        current_line = new_line
        print(current_line)

    # keep only up to new line
    output_array = output_array[:new_line, :]

    # dump
    output = os.path.join(fp.path_to_model_data_folder, 'rf_data.npy')
    with open(output, 'wb+') as fh:
        np.save(fh, output_array, allow_pickle=False)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
