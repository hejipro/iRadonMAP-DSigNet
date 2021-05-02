'''
Description: 
Author: heji
Date: 2021-04-26 14:47:29
LastEditTime: 2021-04-29 21:17:00
LastEditors: GuoYi
'''

import pickle
import torch
import time 
from torch.autograd import Variable 
import scipy.io
from Datasets.imageProcess import DeMayoTrans, DeSinoTrans
import re
from torchvision import transforms

def test_model(dataloaders, model, criterion=None, opt=None):
    post_recon_trans_img = DeMayoTrans(opt.WaterAtValue, trans_style='image')
    post_recon_trans_sino = DeSinoTrans(trans_style='sino')

    for i_batch, data in enumerate(dataloaders['test']):
        print('Processing batch_{}...'.format(i_batch+1))

        inputs, labels = data['sinogram'], data['ndct']
        if opt.use_cuda:  # wrap them in Variable
            labels = Variable(labels).cuda(opt.gpu_id_conv)
            inputs = Variable(inputs).cuda(opt.gpu_id_conv)
        else:
            labels = Variable(labels)
            inputs = Variable(inputs)

        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        data['output'] = outputs
        data['loss'] = loss

        data['output'], data['ndct'] = post_recon_trans_img(data['output']), post_recon_trans_img(data['ndct'])
        data['sinogram'] = (post_recon_trans_sino(data['sinogram'])-data['RescaleIntercept'])/data['RescaleSlope']
        data.pop('RescaleIntercept')
        data.pop('RescaleSlope')

        data_name = ''.join(data['name'])
        data.pop('name')

        if opt.save_as_mat:
            data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in data.items()}
            scipy.io.savemat(opt.root_path + '{}/{}.mat'.format(opt.target_folder, data_name), mdict = data_save)
        else:
            f = open(opt.root_path + "{}/{}.dat".format(opt.target_folder, data_name), "wb")
            pickle.dump(data, f, True)
            f.close()
            
