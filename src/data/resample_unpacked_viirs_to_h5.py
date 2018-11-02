# -*- coding: utf-8 -*-

import logging
import os
import re
import glob

import h5py
import numpy as np

import src.data.tools as tools
import src.config.filepaths as fp


def main():

    # set up datetime list for processing
    file_times = []
    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr):
        try:
            file_times.append(re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group())
        except:
            continue
    file_times = set(file_times)

    for file_time in file_times:

        # set up resampler
        geo_file_path = glob.glob(fp.path_to_viirs_geo + '/*' + file_time + '*')[0]
        geo_h5 = h5py.File(geo_file_path, "r")
        lat = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Latitude'][:]
        lon = geo_h5['All_Data']['VIIRS-MOD-GEO-TC_All']['Longitude'][:]
        resampler = tools.utm_resampler(lat, lon, 750)

        # setup h5 output
        h5_file = 'viirs_sdr_reproj_' + file_time + '.h5'
        h5_outpath = os.path.join(fp.path_to_viirs_sdr_reprojected_h5, h5_file)
        hf = h5py.File(h5_outpath, 'w')

        # run resampling
        viirs_files = glob.glob(fp.path_to_viirs_sdr + '/*' + file_time + '*')
        for i, viirs_file in enumerate(viirs_files):

            viirs_sdr = h5py.File(viirs_file, 'r')

            band = str(int(viirs_file.split('/')[-1][3:5]))
            ds = viirs_sdr['All_Data']['VIIRS-M' + band + '-SDR_All']['Radiance'][:].astype('uint64')

            if i == 0:
                mask = ds >= np.max(ds)
                masked_lats = np.ma.masked_array(resampler.lats, mask)
                masked_lons = np.ma.masked_array(resampler.lons, mask)

                lonlats = resampler.area_def.get_lonlats()
                hf.create_dataset('Latitude', data=lonlats[1])
                hf.create_dataset('Longitude', data=lonlats[0])

            resampled_ds = resampler.resample_image(ds,
                                                    masked_lats,
                                                    masked_lons, fill_value=-999)
            hf.create_dataset('VIIRS-M' + band, data=resampled_ds, dtype='f')
        hf.close()


if __name__ == "__main__":
    main()
