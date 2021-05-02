'''
Description: 
Author: heji
Date: 2021-04-26 10:42:14
LastEditTime: 2021-04-29 15:48:30
LastEditors: GuoYi
'''
import torch 
from torch.autograd import Function
import torch.nn as nn

def PixelIndexCal_DownSampling(length, width, lds, wds):
    length, width = int(length/lds), int(width/wds)
    ds_indices = torch.zeros(lds*wds, width*length).type(torch.LongTensor)

    for x in range(lds):
        for y in range(wds):
            k = x*width*wds+y
            for z in range(length):
                i, j = z*width, x*wds+y
                st = k+z*width*wds*lds
                ds_indices[j, i:i+width] = torch.tensor(range(st,st+width*wds, wds))

    return ds_indices.view(-1)


def PixelIndexCal_UpSampling(index, length, width):
    index = index.view(-1)
    _, ups_indices = index.sort(dim=0, descending=False)

    return ups_indices.view(-1)


class DotProduct(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = input*weight.unsqueeze(0).expand_as(input)

        return output

    @staticmethod
    def backward(ctx, grad_output): 
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output*weight.unsqueeze(0).expand_as(grad_output)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output*input
            grad_weight = grad_weight.sum(0).squeeze(0)

        return grad_input, grad_weight


class Sino_weighted(nn.Module):
    def __init__(self, geo):
        super(Sino_weighted, self).__init__()
        self.geo = geo

        self.weight = nn.Parameter(torch.Tensor(self.geo['nDetecU']))
        self.register_parameter('bias', None)
        # self.bias = nn.Parameter(torch.Tensor(self.geo['nDetecU']))

    def forward(self, input):
        input = input.view(-1, 1, self.geo['nDetecU'])
        output = DotProduct.apply(input, self.weight)
        # output += self.bias.unsqueeze(0).expand_as(output)

        return output
        

class DownSamplingBlock(nn.Module):
    def __init__(self, planes=8, length=1152, width=736, lds=2, wds=2, gpu_id_conv=0):
        super(DownSamplingBlock, self).__init__()
        self.length = int(length/lds)
        self.width = int(width/wds)
        self.extra_channel = lds*wds
        self.ds_index = PixelIndexCal_DownSampling(length, width, lds, wds).cuda(gpu_id_conv)
        self.filter = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        _, channel, length, width = input.size()
        output = torch.index_select(input.view(-1, channel, length*width), 2, self.ds_index)
        output = output.view(-1, channel*self.extra_channel, self.length, self.width)
        output = self.leakyrelu(self.ln(self.filter(output)))

        return output


class UpSamplingBlock(nn.Module):
    def __init__(self, planes=8, length=64, width=64, lups=2, wups=2, gpu_id_conv=0):
        super(UpSamplingBlock, self).__init__()

        self.length = length*lups
        self.width = width*wups
        self.extra_channel = lups*wups
        ds_index = PixelIndexCal_DownSampling(self.length, self.width, lups, wups)
        self.ups_index = PixelIndexCal_UpSampling(ds_index, self.length, self.width).cuda(gpu_id_conv)
        self.filter = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        
    def forward(self, input):
        _, channel, length, width = input.size()
        channel = int(channel/self.extra_channel)
        output = torch.index_select(input.view(-1, channel, self.extra_channel*length*width), 2, self.ups_index)
        output = output.view(-1, channel, self.length, self.width)
        output = self.leakyrelu(self.ln(self.filter(output)))

        return output


class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super(ResidualBlock, self).__init__()

        self.filter1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln1 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.filter2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln2 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        output = self.leakyrelu1(self.ln1(self.filter1(input)))
        output = self.ln2(self.filter2(output))
        output += input
        output = self.leakyrelu2(output)

        return output

        