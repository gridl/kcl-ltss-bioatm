'''
Contains the various file paths for the Python scripts
'''

import os

root_path = '/Volumes/INTENSO/project_plume/'


# paths to raw data
path_to_viirs_sdr = os.path.join(root_path, 'raw/viirs')
path_to_viirs_sdr_reprojected_tcc = os.path.join(root_path, 'raw/reprojected_viirs/tcc')
path_to_viirs_sdr_reprojected_fcc = os.path.join(root_path, 'raw/reprojected_viirs/fcc')