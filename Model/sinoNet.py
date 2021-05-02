'''
Description: 
Author: heji
Date: 2021-04-26 10:45:46
LastEditTime: 2021-04-29 15:37:57
LastEditors: GuoYi
'''
import torch.nn as nn
from Model.utils import DownSamplingBlock, ResidualBlock

class SinoNet(nn.Module):
    def __init__(self, geo, bp_channel, num_filters, gpu_id_conv):
        super(SinoNet, self).__init__()

        model_list  = [nn.Conv2d(geo['slices'], num_filters, kernel_size=3, stride=1, padding=1, bias=True), nn.GroupNorm(num_channels=num_filters, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
        model_list += [DownSamplingBlock(planes=num_filters*4, length=1152, width=736, lds=2, wds=2, gpu_id_conv=gpu_id_conv)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        
        model_list += [nn.Conv2d(num_filters*4, bp_channel, kernel_size=1, stride=1, padding=0, bias=True)]
        self.model = nn.Sequential(*model_list)

    def forward(self, input):

        return self.model(input)

