'''
Description: 
Author: heji
Date: 2021-04-26 11:34:03
LastEditTime: 2021-04-29 20:35:30
LastEditors: GuoYi
'''
import torch 
import math 
import numpy as np 
import torch.nn.init as init


class init_ramlak_1D(object):
    def __init__(self, geo):
        self.geo = geo

    def __call__(self):
        ramlak_width = 2*self.geo['nDetecU']+1
        pixel_size = self.geo['dDetecU']

        hw = int((ramlak_width-1)/2)
        weights = [-1 / math.pow( i * math.pi * pixel_size, 2 ) if i%2 == 1 else 0 for i in range(-hw, hw+1)]
        weights[hw] = 1 / (4 * math.pow(pixel_size, 2))

        weights = torch.Tensor(np.array(weights)).unsqueeze(0).unsqueeze(0)

        return weights

class init_Backprojection(object):
    def __init__(self, geo):
        self.geo = geo

    def __call__(self):
        if self.geo['mode'] is 'parallel':
            weights_size = self.geo['nVoxelX']*self.geo['nVoxelY']*self.geo['views']*self.geo['extent']
            weights = torch.Tensor(np.ones(weights_size)).type(torch.FloatTensor)
        elif self.geo['mode'] is 'fanflat':
            weights = self.geo['weights']

        return weights

class init_sino_weighted(object):
    def __init__(self, geo):
        self.geo = geo

    def __call__(self):
        us = np.array([(-self.geo['nDetecU']/2+0.5+i) * self.geo['dDetecU'] for i in range(self.geo['nDetecU'])])
        weights = self.geo['DSD'] / np.sqrt(self.geo['DSD']**2 + us**2)

        return torch.Tensor(weights)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BackProj') != -1:
        if m.weight is not None:
            m.weight.data = backprojector_init()
            # m.weight.data.uniform_(-0.1, 0.1) # Not a very smart way to initialize weights
        if m.bias is not None:
            m.bias.data.zero_()

    if classname.find('Conv1d') != -1:
        if m.weight is not None:
            m.weight.data = ramp_init()
        if m.bias is not None:
            m.bias.data.zero_()

    if classname.find('Sino_weighted') != -1:
        if m.weight is not None:
            m.weight.data = sino_init()
        if m.bias is not None:
            m.bias.data.zero_()

    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        if m.weight is not None:
            init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()



        