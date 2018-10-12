import mahotas
import h5py
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

fname = 'GMODO-SVM01-SVM04-SVM05-SVM06-SVM07-SVM10-SVM11-SVM12-SVM15_npp_d20160701_t1814088_e1819492_b24237_c20180423131243009989_noaa_ops_reproj.h5'
path = '/Volumes/INTENSO/kcl-ltss-bioatm/raw/reprojected_viirs/h5'
sdr_path = os.path.join(path, fname)
sdr = h5py.File(sdr_path,  "r")

image = sdr['VIIRS-M1'][100:200, 100:200]
image = (image/256).astype('uint8')


feat = mahotas.features.haralick(image, ignore_zeros=True, return_mean=True)