# -*- coding: utf-8 -*-

import logging
import os

import h5py
import numpy as np
import scipy.misc as misc

import src.data.tools as tools
import src.config.filepaths as fp


def read_h5(f):
    return h5py.File(f,  "r")


def create_resampler(viirs_data):
    lats = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Latitude'][:]
    lons = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Longitude'][:]
    return tools.utm_resampler(lats, lons, 750)


def main():

    fv = -999.0

    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr):

        if 'DS' in viirs_sdr_fname:
            continue

        try:

            h5_file = viirs_sdr_fname.replace('.h5', '_reproj.h5')
            h5_outpath = os.path.join(fp.path_to_viirs_sdr_reprojected_h5, h5_file)

            if os.path.isfile(h5_outpath):
                print(viirs_sdr_fname, 'already reprojected')
                continue

            logger.info("Processing viirs file: " + viirs_sdr_fname)

            viirs_sdr = read_h5(os.path.join(fp.path_to_viirs_sdr, viirs_sdr_fname))
            utm_resampler = create_resampler(viirs_sdr)
            hf = h5py.File(h5_outpath, 'w')

            for band in ['1', '4', '5', '6', '7', '10', '11', '12', '15']:
                ds = viirs_sdr['All_Data']['VIIRS-M' + band + '-SDR_All']['Radiance'][:]

                if band == '1':
                    mask = ds == 65533
                    masked_lats = np.ma.masked_array(utm_resampler.lats, mask)
                    masked_lons = np.ma.masked_array(utm_resampler.lons, mask)
                    resampled_lats = utm_resampler.resample_image(utm_resampler.lats,
                                                            masked_lats,
                                                            masked_lons, fill_value=fv)
                    resampled_lons = utm_resampler.resample_image(utm_resampler.lons,
                                                            masked_lats,
                                                            masked_lons, fill_value=fv)
                    resampled_lats[resampled_lats == fv] = fv
                    resampled_lons[resampled_lons == fv] = fv
                    hf.create_dataset('Latitude', data=resampled_lats)
                    hf.create_dataset('Longitude', data=resampled_lons)


                resampled_ds = utm_resampler.resample_image(ds,
                                                            masked_lats,
                                                            masked_lons, fill_value=fv)
                resampled_ds[resampled_ds == fv] = fv
                hf.create_dataset('VIIRS-M'+band, data=resampled_ds)

            hf.close()

        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()