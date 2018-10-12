'''
Contains the various file paths for the Python scripts
'''

import os

#root_path = '/Volumes/INTENSO/kcl-ltss-bioatm'
root_path = '/home/users/dnfisher/data/kcl-ltss-bioatm'

# paths to raw data
path_to_viirs_sdr = os.path.join(root_path, 'raw/viirs')
path_to_viirs_sdr_reprojected_tcc = os.path.join(root_path, 'raw/reprojected_viirs/tcc')
path_to_viirs_sdr_reprojected_fcc = os.path.join(root_path, 'raw/reprojected_viirs/fcc')
path_to_viirs_sdr_reprojected_h5 = os.path.join(root_path, 'raw/reprojected_viirs/h5')
path_to_viirs_sdr_texture_h5 = os.path.join(root_path, 'raw/reprojected_viirs/texture')
path_to_viirs_plume_masks = os.path.join(root_path, 'raw/reprojected_viirs/mask')

# data in form suitable for models
path_to_model_data_folder = os.path.join(root_path, 'interim/model_input')
path_to_model_folder = os.path.join(root_path, 'interim/models')
