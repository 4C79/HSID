from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .base_block import *
import matplotlib.pyplot as plt


draw_mask = False
# drSpa = 1
# drSpaConv = 1
# drSpec = 1
class asign_index(torch.autograd.Function):

    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        if draw_mask == True:
            total_mask = torch.squeeze(guide_mask)
            total_mask = draw_numbers(total_mask)
            return total_mask,torch.sum(kernel * guide_mask, dim=1)
        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out


class Correlation(nn.Module):
    use_slow = False

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8, guide_input_channel=False, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num
        self.guide_input_channel = guide_input_channel

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )

        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, **kwargs)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input, guide_input=None):

        kernel = self.conv_kernel(input)

        # kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H

        if self.guide_input_channel:
            guide_input.requires_grad_(True)
            guide_feature = self.conv_guide(guide_input)
        else:
            guide_feature = self.conv_guide(input)
        self.guide_feature = guide_feature
        # self.guide_feature = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        output = self.asign_index(output, guide_feature)
        return output

class SpecDRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8, guide_input_channel=False, **kwargs):
        super(SpecDRConv2d, self).__init__()
        self.region_num = region_num
        self.guide_input_channel = guide_input_channel

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )

        self.linear_guide = nn.Linear(in_channels, region_num)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input, guide_input=None):
        b, c, h, w = guide_input.shape
        kernel = self.conv_kernel(input)

        # kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H

        if self.guide_input_channel:
            guide_input = guide_input.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            guide_feature = self.linear_guide(guide_input)
            guide_feature = guide_feature.view(b, h, w, self.region_num).permute(0, 3, 1, 2)
        else:
            guide_feature = self.linear_guide(input)
        self.guide_feature = guide_feature
        # self.guide_feature = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        output = self.asign_index(output, guide_feature)
        return output

class DRConv2d_ss(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8, guide_input_channel=False, **kwargs):
        super(DRConv2d_ss, self).__init__()
        self.region_num = region_num
        self.guide_input_channel = guide_input_channel

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Conv2d(region_num * region_num, region_num * out_channels, kernel_size=1,
                      groups=region_num)
        )
        self.linear_guide = nn.Linear(in_channels, region_num)
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, **kwargs)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply
        self.fution_ss = nn.Conv2d(in_channels * 2, out_channels, 1, 1, 0, bias=True)

    def forward(self, input, guide_input=None):

        b, c, h, w = guide_input.shape
        output = self.conv_kernel(input)

        # kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H

        if self.guide_input_channel:
            guide_feature_spa = self.conv_guide(guide_input)
            guide_input = guide_input.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            guide_feature_spec = self.linear_guide(guide_input)
            guide_feature_spec = guide_feature_spec.view(b, h, w, self.region_num).permute(0, 3, 1, 2)
        # self.guide_feature = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        if self.drSpa ==1:
            output_spa = self.asign_index(output, guide_feature_spa)
            output = output_spa
        if self.drSpec == 1:
            output_spec = self.asign_index(output, guide_feature_spec)
            output = output_spec
        if self.drSpa ==1 and self.drSpec ==1:
        # output = output_spa+output_spec
            output = self.fution_ss(torch.cat((output_spa,output_spec),dim=1))

        output = output+input
        return output

class DRConv2d_ss_k(nn.Module):
    def __init__(self, in_channels, out_channels,
                 drSpa,drSpaConv,drSpec,kernel_size, region_num=8, guide_input_channel=False, **kwargs):
        super(DRConv2d_ss_k, self).__init__()
        self.region_num = region_num
        self.guide_input_channel = guide_input_channel
        self.drSpa = drSpa
        self.drSpaConv = drSpaConv
        self.drSpec = drSpec

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )

        self.corr = Correlation(use_slow=False)

        if self.drSpec == 1:
            self.spectral_guide = SpectralMLP(input_dim=in_channels, output_dim=region_num)
        # self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, **kwargs)
        if self.drSpa == 1:
            if self.drSpaConv == 1:
                self.spatial_guide = TAdaConvBlock_2D(in_channels, region_num)
            else:
                self.spatial_guide = nn.Conv2d(in_channels, region_num,kernel_size=3,stride=1,padding=1)

        self.kwargs = kwargs
        self.asign_index = asign_index.apply
        self.fution_ss = nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1, bias=True)
        # print(self.drSpa,self.drSpaConv,self.drSpec)
        # print("opt: drParams " , self.drSpa,self.drSpaConv,self.drSpec)

    def forward(self, input, guide_input=None):
        
        # print(self.drSpa,self.drSpaConv,self.drSpec)

        b, c, h, w = guide_input.shape
        kernel = self.conv_kernel(input)
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H
        # kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H

        if self.drSpa == 1:
            guide_feature_spa = self.spatial_guide(guide_input)
        if self.drSpec == 1:
            guide_feature_spec = self.spectral_guide(guide_input)

        # self.guide_feature = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        # guide_feature = guide_feature_spa + guide_feature_spec
        # guide_feature = self.fution_ss(torch.cat((guide_feature_spa,guide_feature_spec),dim=1))
        if draw_mask == True:
            spa_mask,output_spa = self.asign_index(output, guide_feature_spa)
            spec_mask,output_spec = self.asign_index(output, guide_feature_spec)
            output_1 = self.fution_ss(torch.cat((output_spa, output_spec), dim=1))
        else:
            if self.drSpa == 1:
                output_spa = self.asign_index(output, guide_feature_spa)
                output_1 = output_spa
            if self.drSpec == 1:
                # print("test")
                output_spec = self.asign_index(output, guide_feature_spec)
                output_1 = output_spec
            if self.drSpa == 1 and self.drSpec == 1:
                # output = output_spa+output_spec
                output_1 = self.fution_ss(torch.cat((output_spa, output_spec), dim=1))
        if draw_mask == True:
            draw_featrue(spa_mask,"spa_mask")
            draw_featrue(spec_mask,"spec_mask")
        output = output_1+input
        return output

class SpecDRConv2d_Easy(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8, guide_input_channel=False, **kwargs):
        super(SpecDRConv2d_Easy, self).__init__()
        self.region_num = region_num
        self.guide_input_channel = guide_input_channel

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )

        self.linear_guide = nn.Linear(in_channels, region_num)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input, guide_input=None):
        b, c, h, w = guide_input.shape
        kernel = self.conv_kernel(input)

        # kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H

        if self.guide_input_channel:
            guide_input = guide_input.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            guide_feature = self.linear_guide(guide_input)
            guide_feature = guide_feature.view(b, h, w, self.region_num).permute(0, 3, 1, 2)
        else:
            guide_feature = self.linear_guide(input)
        self.guide_feature = guide_feature
        # self.guide_feature = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        output = self.asign_index(output, guide_feature)
        return output

class HistDRConv2d(DRConv2d):
    def forward(self, input, histmap):
        """
        use histmap as guide feature directly.
        histmap.shape: [bs, n_bins, h, w]
        """
        histmap.requires_grad_(False)

        kernel = self.conv_kernel(input)
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        output = self.asign_index(output, histmap)
        return output


if __name__ == '__main__':
    B = 1
    in_channels = 48
    out_channels = 48
    size = 32
    conv = SpecDRConv2d(in_channels, out_channels, kernel_size=3, region_num=8,padding=1,guide_input_channel=True).cuda()
    conv.train()
    input = torch.ones(B, in_channels, size, size).cuda()
    guide = torch.ones(B, in_channels, size, size).cuda()
    output = conv(input,guide)
    loss = torch.sum(output - input)
    loss.backward()


    # print(input.shape, output.shape)

    # flops, params
    from thop import profile


    class Conv2d(nn.Module):
        def __init__(self):
            super(Conv2d, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)

        def forward(self, input):
            return self.conv(input)


    conv2 = Conv2d().cuda()
    conv2.train()
    print(input.shape, conv2(input).shape)
    flops2, params2 = profile(conv2, inputs=(input,))
    flops, params = profile(conv, inputs=(input,guide,))

    print('[ * ] DRconv FLOPs      =  ' + str(flops / 1000 ** 3) + 'G')
    print('[ * ] DRconv Params Num =  ' + str(params / 1000 ** 2) + 'M')

    print('[ * ] Conv FLOPs      =  ' + str(flops2 / 1000 ** 3) + 'G')
    print('[ * ] Conv Params Num =  ' + str(params2 / 1000 ** 2) + 'M')
