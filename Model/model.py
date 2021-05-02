'''
Description: 
Author: heji
Date: 2021-04-26 10:12:41
LastEditTime: 2021-04-29 15:48:09
LastEditors: GuoYi
'''
import torch.nn as nn
from Model.sinoNet import SinoNet
from Model.backProjNet import BackProjNet 
from Model.spatialNet import SpatialNet


class DSigNet(nn.Module):
    def __init__(self, geo, opt):
        super(DSigNet, self).__init__()
        self.geo = geo
        self.bp_channel = 4
        self.gpu_id_conv = opt.gpu_id_conv
        self.gpu_id_bp = opt.gpu_id_bp


        self.SinoNet = SinoNet(geo, self.bp_channel, opt.num_filters, self.gpu_id_conv).cuda(self.gpu_id_conv)
        self.BackProjNet = BackProjNet(geo, self.bp_channel).cuda(self.gpu_id_bp)
        self.SpatialNet = SpatialNet(geo, self.bp_channel, opt.num_filters, self.gpu_id_conv).cuda(self.gpu_id_conv)

        
    def forward(self, input):
        output = self.SinoNet(input)
        output = output.cuda(self.gpu_id_bp)
        output = self.BackProjNet(output)
        output = output.cuda(self.gpu_id_conv)
        output = self.SpatialNet(output)

        return output

