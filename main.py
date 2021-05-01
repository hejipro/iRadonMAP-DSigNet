'''
Description: 
Author: heji
Date: 2021-04-26 14:54:57
LastEditTime: 2021-04-30 08:34:48
LastEditors: GuoYi
'''

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np 
np.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=np.nan)
import pickle

import torch 
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


from Utils.initParameter import InitPara
from Utils.initFunction import weights_init
from Datasets.imageProcess import Transpose, TensorFlip, MayoTrans, SinoTrans
from Datasets.datasets import TrainDataSet
from Solver.pixelIndexCal_cuda import PixelIndexCal_cuda 
from Solver.train import train_model
from Solver.test import test_model
from Model.model import DSigNet 



def main():

    opt = InitPara()

    geo_real = {'nVoxelX': 512, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641, 
                'nVoxelY': 512, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641, 
                'nDetecU': 736, 'sDetecU': 504.0128*2, 'dDetecU': 0.6848*2, 
                'offOriginX': 0.0, 'offOriginY': 0.0, 
                'views': 1152, 'slices': 3,
                'DSD': 1085.6, 'DSO': 595.0, 'DOD': 490.6,
                'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
                'mode': opt.geo_mode, 'extent': 1, # currently extent supports 1, 2, or 3.
                }

    geo_virtual = dict()
    geo_virtual.update({x: int(geo_real[x]/opt.scale_factor) for x in ['views']})
    geo_virtual.update({x: int(geo_real[x]/opt.scale_factor) for x in ['nVoxelX', 'nVoxelY', 'nDetecU']})
    geo_virtual.update({x: geo_real[x]/opt.scale_factor for x in ['sVoxelX', 'sVoxelY', 'sDetecU', 'DSD', 'DSO', 'DOD', 'offOriginX', 'offOriginY']})
    geo_virtual.update({x: geo_real[x] for x in ['dVoxelX', 'dVoxelY', 'dDetecU', 'slices', 'start_angle', 'end_angle', 'mode', 'extent']})


    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    post_trans_img = MayoTrans(opt.WaterAtValue, trans_style='image')
    post_trans_sino = SinoTrans(trans_style='sino')

    print('Constructing Datasets...')
    if opt.is_train:
        datasets = {'train': TrainDataSet(opt.root_path, opt.TrainFolder, geo_real, None if opt.Dataset_name is 'MayoRaw' else pre_trans_img, post_trans_img, post_trans_sino, opt.Dataset_name),
                    'val': TrainDataSet(opt.root_path, opt.ValFolder, geo_real, None if opt.Dataset_name is 'MayoRaw' else pre_trans_img, post_trans_img, post_trans_sino, opt.Dataset_name)}
        
        dataloaders = {x: DataLoader(datasets[x], opt.batch_size[x], shuffle=opt.is_shuffle, pin_memory=True, num_workers=opt.num_workers[x]) for x in ['train', 'val']}
        dataset_sizes = {x: opt.batch_num[x]*opt.batch_size[x] for x in ['train', 'val']}
    else:
        datasets = {'test': TrainDataSet(opt.root_path, opt.TestFolder, geo_real, None, post_trans_img, post_trans_sino, 'Mayo_test' if opt.Dataset_name is 'Mayo' else 'MayoRaw_test')}
        dataloaders = {x: DataLoader(datasets[x], opt.batch_size[x], shuffle=opt.is_shuffle, pin_memory=True, num_workers=opt.num_workers[x]) for x in ['test']}
        dataset_sizes = {x: opt.batch_num[x]*opt.batch_size[x] for x in ['test']}
    print('Done!')
    

    print('Generating sinoIndices...')
    if opt.use_cuda:
        geo_virtual['indices'], geo_virtual['weights'] = PixelIndexCal_cuda(geo_virtual)
        geo_virtual['indices'] = geo_virtual['indices'].cuda(opt.gpu_id_bp)
        # geo['weights'] = geo['weights'].cuda()
    print('Done!')

    dsignet = DSigNet(geo_virtual, opt)
    criterion = nn.MSELoss()


    if not opt.reload_model:
        dsignet.apply(weights_init)
        min_loss = None
        pre_losses = None
    else:
        model_name = 'backup_model_{}'.format(opt.net_name) if opt.is_train else 'best_{}_model_{}'.format(opt.reload_mode, opt.net_name)
        epoch_reload_path = opt.root_path + 'Model_save/{}.pkl'.format(model_name)
        if os.path.isfile(epoch_reload_path):
            print('Loading previously trained network: {}...'.format(model_name))
            checkpoint = torch.load(epoch_reload_path, map_location = lambda storage, loc: storage)
            model_dict = dsignet.state_dict()
            checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)
            dsignet.load_state_dict(model_dict)
            del checkpoint
            torch.cuda.empty_cache()
            print('Done!')
        else:
            dsignet.apply(weights_init)
            
        min_loss_path = opt.root_path + "Loss_save/min_loss_{}.dat".format(opt.net_name) if opt.is_train else ''
        min_loss = pickle.load(open(min_loss_path, "rb")) if os.path.isfile(min_loss_path) else None
        pre_losses_path = opt.root_path + "Loss_save/losses_{}.dat".format(opt.net_name) if opt.is_train else ''
        pre_losses = pickle.load(open(pre_losses_path, "rb")) if os.path.isfile(pre_losses_path) else None
        
    if opt.use_cuda:
        # dsignet.cuda()
        criterion.cuda(opt.gpu_id_conv)

    optimizer_ft = optim.RMSprop(dsignet.parameters(), lr=2e-5, momentum=0.9, weight_decay=0.0)

    if opt.is_train and opt.reload_model:
        optimizer_name = "backup_optimizer_{}".format(opt.net_name)
        optimizer_reload_path = opt.root_path + "Optimizer_save/{}.pkl".format(optimizer_name)
        if os.path.isfile(optimizer_reload_path):
            print('Loading previous optimizer: {}...'.format(optimizer_name))
            checkpoint = torch.load(optimizer_reload_path, map_location = lambda storage, loc: storage)
            optimizer_ft.load_state_dict(checkpoint)
            del checkpoint
            torch.cuda.empty_cache()
            print('Done!')

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=0.5) if opt.is_lr_scheduler else None

    if opt.is_train:
        train_model(dataloaders, dsignet, optimizer_ft, criterion, exp_lr_scheduler, min_loss, pre_losses, num_epochs=50000, dataset_sizes=dataset_sizes, opt=opt)
    else:
        dsignet.eval()
        test_model(dataloaders, dsignet, criterion, opt)


if __name__ == '__main__':
    main()

