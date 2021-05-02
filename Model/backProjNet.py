'''
Description: 
Author: heji
Date: 2021-04-26 10:54:34
LastEditTime: 2021-04-26 10:56:26
LastEditors: GuoYi
'''
import torch 
import torch.nn as nn
from Model.utils import DotProduct 


class BackProjNet(nn.Module):
    def __init__(self, geo, channel=8, learn=False):
        super(BackProjNet, self).__init__()
        self.geo = geo
        self.learn = learn
        self.channel = channel
        
        if self.learn:
            self.weight = nn.Parameter(torch.Tensor(self.geo['nVoxelX']*self.geo['nVoxelY']*self.geo['views']*self.geo['extent']))
            self.bias = nn.Parameter(torch.Tensor(self.geo['nVoxelX']*self.geo['nVoxelY']))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        input = input.view(-1, self.channel, self.geo['views']*self.geo['nDetecU'])
        output = torch.index_select(input, 2, self.geo['indices'])
        if self.learn:
            output = DotProduct.apply(output, self.weight)
        output = output.view(-1, self.channel, self.geo['nVoxelX']*self.geo['nVoxelY'], self.geo['views']*self.geo['extent'])
        output = torch.sum(output, 3) * (self.geo['end_angle']-self.geo['start_angle']) / (2*self.geo['views']*self.geo['extent'])
        if self.learn:
            output += self.bias.unsqueeze(0).expand_as(output)
        output = output.view(-1, self.channel, self.geo['nVoxelX'], self.geo['nVoxelY'])

        return output

