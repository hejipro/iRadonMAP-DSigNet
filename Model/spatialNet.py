'''
Description: 
Author: heji
Date: 2021-04-26 10:59:22
LastEditTime: 2021-04-29 15:42:22
LastEditors: GuoYi
'''
import torch.nn as nn
from Model.utils import ResidualBlock, UpSamplingBlock 


class SpatialNet(nn.Module):
    def __init__(self, geo, bp_channel, num_filters, gpu_id_conv):
        super(SpatialNet, self).__init__()

        model_list  = [nn.Conv2d(bp_channel, num_filters*4, kernel_size=3, stride=1, padding=1, bias=True), nn.GroupNorm(num_channels=num_filters*4, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [UpSamplingBlock(planes=num_filters, length=256, width=256, lups=2, wups=2, gpu_id_conv=gpu_id_conv)]
        
        model_list += [nn.Conv2d(num_filters, geo['slices'], kernel_size=1, stride=1, padding=0, bias=True)]
        self.model = nn.Sequential(*model_list)

    def forward(self, input):

        return self.model(input)

