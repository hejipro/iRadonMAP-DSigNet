'''
Description: 
Author: heji
Date: 2021-04-26 11:41:11
LastEditTime: 2021-04-29 20:45:34
LastEditors: GuoYi
'''

import astra
import torch 
import copy 
import numpy as np 
from torch.utils.data import Dataset
from torchvision import transforms

from Datasets.utils import findFiles, pop_paths, findpath, image_read
from Datasets.imageProcess import CTnum2AtValue, RandomCrop
from Datasets.imageProcess import TensorFlip, Transpose

class TrainData(Dataset):
    def __init__(self, root_dir, folder, crop_size=None, recon_slices=None, trf_op=None, Dataset_name='Mayo_test'):
        self.Dataset_name = Dataset_name
        self.trf_op = trf_op
        self.crop_size = crop_size
        self.recon_slices = recon_slices
        self.WaterAtValue = 0.0192

        if Dataset_name in ['Mayo', 'Mayo_test']:
            self.hd_image_paths = [findFiles(root_dir+'MayoData/{}/{}_{}/*.IMA'.format(x, y, z)) for x in folder['patients'] for y in folder['HighDose'] for z in folder['SliceThickness']]
            self.hd_image_paths = pop_paths(self.hd_image_paths, self.recon_slices//2)
            self.hd_image_paths = [x for j in self.hd_image_paths for x in j]
            self.ld_image_paths = [findFiles(root_dir+'MayoData/{}/{}_{}/*.IMA'.format(x, y, z)) for x in folder['patients'] for y in folder['LowDose'] for z in folder['SliceThickness']]
            self.ld_image_paths = pop_paths(self.ld_image_paths, self.recon_slices//2)
            self.ld_image_paths = [x for j in self.ld_image_paths for x in j]
            self.fix_list = [CTnum2AtValue(self.WaterAtValue)]
            self.RandomCrop = RandomCrop(self.crop_size)
            self.root_dir = root_dir+'MayoData/{}/{}_{}/'.format(''.join(folder['patients']), ''.join(folder['LowDose']), ''.join(folder['SliceThickness']))
        elif Dataset_name is ['MayoRaw', 'MayoRaw_test']:
            self.hd_image_paths = [findFiles(root_dir+'MayoRawData/{}/imgdata/{}_{}_{}*.mat'.format(x, x, y, z)) for x in folder['patients'] for y in folder['HighDose'] for z in folder['SliceThickness']]
            self.hd_image_paths = pop_paths(self.hd_image_paths, self.recon_slices//2)
            self.hd_image_paths = [x for j in self.hd_image_paths for x in j]
            self.ld_image_paths = [findFiles(root_dir+'MayoRawData/{}/rawdata/{}_{}_{}*.mat'.format(x, x, y, z)) for x in folder['patients'] for y in folder['LowDose'] for z in folder['SliceThickness']]
            self.ld_image_paths = pop_paths(self.ld_image_paths, self.recon_slices//2)
            self.ld_image_paths = [x for j in self.ld_image_paths for x in j]
            self.hd_fix_list = [CTnum2AtValue(self.WaterAtValue)]
            self.ld_fix_list = [Transpose(), TensorFlip(0)]
            self.root_dir = root_dir+'MayoRawData/{}/imgdata/'.format(''.join(folder['patients']))
    
    def __len__(self):
        return len(self.hd_image_paths)
    
    def __getitem__(self, idx):
        hd_image_path_mid = self.hd_image_paths[idx]
        ld_image_path_mid = self.ld_image_paths[idx]

        hd_image, ld_image = [], []
        for i in range(self.recon_slices):
            hd_image_path = findpath(hd_image_path_mid, i-self.recon_slices//2, self.Dataset_name)
            hd_image.append(image_read(hd_image_path, self.Dataset_name))
            ld_image_path = findpath(ld_image_path_mid, i-self.recon_slices//2, self.Dataset_name)
            ld_image.append(image_read(ld_image_path, self.Dataset_name))

        if self.Dataset_name is 'Mayo':
            crop_point = np.random.randint(self.crop_size, size=2)
            for i in range(self.recon_slices):
                hd_image[i] = self.RandomCrop(hd_image[i], crop_point)
                ld_image[i] = self.RandomCrop(ld_image[i], crop_point)

        elif self.Dataset_name is 'Mayo_test':
            imgname = ld_image_path_mid.replace(self.root_dir, '').replace('.IMA', '')
        elif self.Dataset_name is 'MayoRaw_test':
            imgname = ld_image_path_mid.replace(self.root_dir, '').replace('.mat', '')

        random_list = []
        if self.trf_op is not None:
            keys = np.random.randint(2,size=len(self.trf_op))
            for i, key in enumerate(keys):
                random_list.append(self.trf_op[i]) if key == 1 else None

        if self.Dataset_name in ['MayoRaw', 'MayoRaw_test']:
            transform_hd = transforms.Compose(self.hd_fix_list + random_list)
            transform_ld = transforms.Compose(self.ld_fix_list + random_list)
            for i in range(self.recon_slices):
                hd_image[i], ld_image[i] = transform_hd(hd_image[i]), transform_ld(ld_image[i])
        elif self.Dataset_name in ['Mayo', 'Mayo_test']:
            transform = transforms.Compose(self.fix_list + random_list)
            for i in range(self.recon_slices):
                hd_image[i], ld_image[i] = transform(hd_image[i]), transform(ld_image[i])

        if self.Dataset_name in ['Mayo_test', 'MayoRaw_test']:
            return hd_image, ld_image, imgname 
        else:
            return hd_image, ld_image

class TrainDataSet(Dataset):
    def __init__(self, root_dir, folder, geo, pre_trans_img=None, post_trans_img=None, post_trans_sino=None, Dataset_name='Mayo_test'):
        self.Dataset_name = Dataset_name
        self.imgset = TrainData(root_dir, folder, geo['nVoxelX'], geo['slices'], pre_trans_img, Dataset_name)
        self.vol_geom = astra.create_vol_geom(geo['nVoxelY'], geo['nVoxelX'],
                                               -1*geo['sVoxelY']/2 + geo['offOriginY'], geo['sVoxelY']/2 + geo['offOriginY'], -1*geo['sVoxelX']/2 + geo['offOriginX'], geo['sVoxelX']/2 + geo['offOriginX'])
        self.proj_geom = astra.create_proj_geom(geo['mode'], geo['dDetecU'], geo['nDetecU'], 
                                                np.linspace(geo['start_angle'], geo['end_angle'], geo['views'],False), geo['DSO'], geo['DOD'])
        if geo['mode'] is 'parallel':
            self.proj_id = astra.create_projector('linear', self.proj_geom, self.vol_geom)
        elif geo['mode'] is 'fanflat':
            self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom) #line_fanflat

        _, img_a, img_b = post_trans_img(np.ones((geo['nVoxelX'], geo['nVoxelY'])))
        sinogram_id, img_Ab = astra.create_sino(np.ones((geo['nVoxelX'], geo['nVoxelY']))*img_b, self.proj_id)
        astra.data2d.delete(sinogram_id)
        self.img_a, self.img_Ab = img_a, img_Ab

        self.recon_slices = geo['slices']
        self.post_trans_img = post_trans_img
        self.post_trans_sino = post_trans_sino
        
    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx):
        if self.Dataset_name in ['Mayo_test', 'MayoRaw_test']:
            img, ld_img, imgname = self.imgset[idx]
        else:
            img, ld_img = self.imgset[idx]

        if self.Dataset_name in ['MayoRaw_test', 'MayoRaw']:
            sinogram = ld_img
        else:
            sinogram = []
            for i in range(self.recon_slices):
                sinogram_id, sinogram_tmp = astra.create_sino(ld_img[i], self.proj_id)
                astra.data2d.delete(sinogram_id)
                sinogram.append(sinogram_tmp)

        if self.post_trans_img is not None:
            for i in range(self.recon_slices):
                img[i], _, _ = self.post_trans_img(img[i])
                sinogram[i] = self.img_a * sinogram[i] + self.img_Ab

        img, sinogram = np.array(img), np.array(sinogram)

        if self.post_trans_sino is not None:
            sinogram, _, _ = self.post_trans_sino(sinogram)

        if self.Dataset_name in ['Mayo', 'MayoRaw']:
            sample = {'ndct': torch.from_numpy(img).type(torch.FloatTensor), 'sinogram': torch.from_numpy(sinogram).type(torch.FloatTensor)}
        else:
            sample = {'ndct': torch.from_numpy(img).type(torch.FloatTensor), 'sinogram': torch.from_numpy(sinogram).type(torch.FloatTensor), 
                      'name': imgname, 'RescaleSlope': self.img_a, 'RescaleIntercept': self.img_Ab}

        return sample

        