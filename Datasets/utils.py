'''
Description: 
Author: heji
Date: 2021-04-26 11:43:56
LastEditTime: 2021-04-29 14:37:03
LastEditors: GuoYi
'''
import glob 
import re 
import pydicom 
import h5py
import numpy as np 


def findFiles(path): return glob.glob(path)

def image_read(image_path, image_type):
    if image_type in ['Mayo', 'Mayo_test']:
        image = pydicom.dcmread(image_path)
        imgdata = image.pixel_array * image.RescaleSlope + image.RescaleIntercept
        return imgdata
    elif image_type in ['MayoRaw', 'MayoRaw_test']:
        image = h5py.File(image_path, 'r')
        if 'img_x' in image:
            return np.transpose(image['img_x']) 
        elif  'sino_fan' in image:
            return np.transpose(image['sino_fan'])

def pop_paths(paths, num):
    for path in paths:
        for i in range(num):
            path.pop(0)
            path.pop()
    return paths


def findpath(path, idx, image_type):
    nums = re.findall('\d+', path)
    if image_type in ['Mayo', 'Mayo_test']:
        new_path = path.replace('_{}.IMA'.format(nums[-1]), '_{}.IMA'.format(int(nums[-1])+idx))
    elif image_type in ['MayoRaw', 'MayoRaw_test']:
        new_path = path.replace('_{}.mat'.format(nums[-1]), '_{}.mat'.format(int(nums[-1])+idx))
    return new_path

