'''
Description: 
Author: heji
Date: 2021-04-26 11:30:06
LastEditTime: 2021-04-27 10:47:09
LastEditors: GuoYi
'''
import torch 
import math 
import numpy as np 


class Normalize(object):
    def __init__(self, normalize_type='image'):
        if normalize_type is 'image':
            self.mean = 128.0
        elif normalize_type is 'sino':
            self.mean = -150.0
        elif normalize_type is 'self':
            self.mean = None

    def __call__(self, image):
        if self.mean is not None:
            img_mean = self.mean
        else:
            img_mean = np.mean(image)

        image = image - img_mean
        image = image / 255.0

        return image, img_mean


class DeNormalize(object):
    def __init__(self, normalize_type='image'):
        if normalize_type is 'image':
            self.mean = 128.0
        elif normalize_type is 'sino':
            self.mean = -150.0
        elif normalize_type is 'self':
            self.mean = None

    def __call__(self, image):
        if self.mean is not None:
            img_mean = self.mean
        else:
            img_mean = np.mean(image)

        image = image * 255.0
        image = image + img_mean

        return image


class Scale2Gen(object):
    def __init__(self, scale_type='image'):
        if scale_type is 'image':
            self.mmin = -1024.0
            self.mmax = 2500.0
        elif scale_type is 'self':
            self.mmin = None
            self.mmax = None

    def __call__(self, image):
        if self.mmin is not None:
            img_min, img_max = self.mmin, self.mmax
        else:
            img_min, img_max = np.min(image), np.max(image)
        
        image = (image - img_min) / (img_max-img_min) * 255.0

        return image, img_min, img_max


class Gen2Scale(object):
    def __init__(self, scale_type='image'):
        if scale_type is 'image':
            self.mmin = -1024.0
            self.mmax = 2500.0
        elif scale_type is 'self':
            self.mmin = None
            self.mmax = None

    def __call__(self, image):
        if self.mmin is not None:
            img_min, img_max = self.mmin, self.mmax
        else:
            img_min, img_max = np.min(image), np.max(image)

        image = (image / 255.0) * (img_max-img_min) + img_min

        return image


class Transpose(object):
    def __call__(self, image):
        return image.transpose((1, 0))
  
    
class TensorFlip(object):
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, image):
        if 0 == self.dim:
            return image[::-1,...]
        if 1 == self.dim:
            return image[...,::-1]


class CTnum2AtValue(object):
    def __init__(self, WaterAtValue):
        self.WaterAtValue = WaterAtValue

    def __call__(self, image):
        image = image *self.WaterAtValue /1000.0 + self.WaterAtValue
        return image


class AtValue2CTnum(object):
    def __init__(self, WaterAtValue):
        self.WaterAtValue = WaterAtValue

    def __call__(self, image):
        image = (image -self.WaterAtValue)/self.WaterAtValue *1000.0
        return image


class MayoTrans(object):
    def __init__(self, WaterAtValue, trans_style='self'):
        self.WaterAtValue = WaterAtValue
        self.AtValue2CTnum = AtValue2CTnum(WaterAtValue)
        self.Scale2Gen = Scale2Gen(trans_style)
        self.Normalize = Normalize(trans_style)

    def __call__(self, image):
        image = self.AtValue2CTnum(image)
        image, img_min, img_max = self.Scale2Gen(image)
        image, img_mean = self.Normalize(image)

        a = 1000.0/((img_max-img_min)*self.WaterAtValue)
        b = -(img_min+1000.0)/(img_max-img_min)-img_mean/255.0

        return image, a, b

class DeMayoTrans(object):
    def __init__(self, WaterAtValue, trans_style='self'):
        self.WaterAtValue = WaterAtValue
        self.DeNormalize = DeNormalize(trans_style)
        self.Gen2Scale = Gen2Scale(trans_style)
        # self.CTnum2AtValue = CTnum2AtValue(WaterAtValue)
        
    def __call__(self, image):
        image = self.DeNormalize(image)
        image = self.Gen2Scale(image)
        # image = self.CTnum2AtValue(image)

        return image

class SinoTrans(object):
    def __init__(self, trans_style='self'):
        self.Normalize = Normalize(trans_style)

    def __call__(self, sino):
        sino, img_mean = self.Normalize(sino)

        a = 1.0/255.0
        b = -img_mean/255.0

        return sino, a, b

class DeSinoTrans(object):
    def __init__(self, trans_style='self'):
        self.DeNormalize = DeNormalize(trans_style)

    def __call__(self, sino):
        sino = self.DeNormalize(sino)

        return sino

class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, crop_point):
        image = np.hstack((image, image))
        image = np.vstack((image, image))

        image = image[crop_point[0]:crop_point[0]+self.crop_size, crop_point[1]:crop_point[1]+self.crop_size]
        image = np.pad(image,((math.ceil((self.crop_size - image.shape[0])/2), math.floor((self.crop_size - image.shape[0])/2)),
                (math.ceil((self.crop_size - image.shape[1])/2), math.floor((self.crop_size - image.shape[1])/2))),'constant')
        return image



