import logging
import os
import h5py

import src.config.filepaths as fp
import src.features.textures as textures


def main():

    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr_reprojected_h5):

        if 'DS' in viirs_sdr_fname:
            continue

        sdr_path = os.path.join(fp.path_to_viirs_sdr_reprojected_h5, viirs_sdr_fname)
        sdr = h5py.File(sdr_path,  "r")

        # get the blue channel
        blue = sdr['VIIRS-M1'][:]

        # run GLCM on blue
        texture_generator = textures.CooccurenceMatrixTextures(blue)

        diss = texture_generator.getDissimlarity()
        corr, var, mean = texture_generator.getCorrVarMean()

        # dump the outputs
        h5_file = viirs_sdr_fname.replace('_reproj.h5', '_texture.h5')
        h5_outpath = os.path.join(fp.path_to_viirs_sdr_texture_h5, h5_file)
        hf = h5py.File(h5_outpath, 'w')

        hf.create_dataset('diss', data=diss)
        hf.create_dataset('corr', data=corr)
        hf.create_dataset('var', data=var)
        hf.create_dataset('mean', data=mean)

        hf.close()


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()