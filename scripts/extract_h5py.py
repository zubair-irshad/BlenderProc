import h5py
import cv2
import os
import numpy as np

output_dir = './output/test4'

file_list = os.listdir(output_dir)
file_list = [x for x in file_list if x[-5:]=='.hdf5']
# os.mkdir(os.path.join(output_dir, 'imgs'))

for hdf5_name in file_list:
    with h5py.File(os.path.join(output_dir, hdf5_name)) as f:
        colors = np.array(f["colors"])
    cv2.imwrite(os.path.join(output_dir, 'imgs', hdf5_name+'.jpg'), colors)

