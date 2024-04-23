from models.blocks import *
import torch
from torch import nn
import numpy as np
from .DConv.atten.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION
from .matching import *

class COARSENet(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body
        - up_res_depth: depth of residual layers in each upsample block

    """

    def __init__(
            self,
            channels = 32,
            out_channels = 1,
            feat_layer = 1,
            feat_num = 4,
            has_ad = False,
            is_2d = True,
            opt=None
    ):
        super(COARSENet, self).__init__()
        self.opt = opt
        self.feat_num =feat_num

        if self.opt.is_de == 1:
           self.in_channels = self.opt.n_sharp + self.feat_num + 3
        else:
            self.in_channels = self.feat_num+ 3


        self.in_block= nn.Sequential()
        self.in_block.append(nn.Conv2d(self.in_channels, channels, 3, 1, 1))

        self.feat_ext = nn.Sequential()
        self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.feat_ext.append(nn.LeakyReLU(0.1))
        self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.feat_ext.append(nn.LeakyReLU(0.1))
        # self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        # self.feat_ext.append(nn.LeakyReLU(0.1))
        # self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        # self.feat_ext.append(nn.LeakyReLU(0.1))
        self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.feat_ext_low = nn.Sequential()
        self.feat_ext_low.append(nn.Conv2d(channels, self.feat_num, 3, 1, 1))

        self.sr_in_block = nn.Sequential()
        self.sr_in_block.append(nn.Conv2d(self.feat_num, channels, 3, 1, 1))
        self.sr = nn.Sequential()
        # self.sr.append(nn.Conv2d(channels, channels, 3, 1, 1))
        # self.sr.append(nn.LeakyReLU(0.1))
        self.sr.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.sr.append(nn.Tanh())
        self.sr.append(nn.Conv2d(channels, channels, 3, 1, 1))

        self.sr_1_out_block = nn.Sequential()
        self.sr_1_out_block.append(nn.Conv2d(channels, out_channels, 3, 1, 1))

        self.sr_2_out_block = nn.Sequential()
        self.sr_2_out_block.append(nn.Conv2d(channels, out_channels, 3, 1, 1))

        self.pixelatten = nn.Sequential()
        self.pixelatten.append(nn.Conv3d(1,1,3,1,1))
        self.pixelatten.append(nn.Tanh())
        # self.pixelatten2 = nn.Sequential()
        # self.pixelatten2.append(nn.Conv3d(1,1,3,1,1))
        # self.pixelatten2.append(nn.Tanh())


    def spectral_residual(self,hidden):

        out1= self.sr_in_block(hidden)
        out2 = self.sr(out1)
        out2_atten_F = self.pixelatten(torch.unsqueeze(out2, dim=1)).squeeze(dim=1)
        out2_atten_R = self.pixelatten(torch.unsqueeze(out2, dim=1)).squeeze(dim=1)
        out2_f = out2 + out2 * out2_atten_F
        out2_R = out2 + out2 * out2_atten_R
        F_t = self.sr_1_out_block(out2_f)
        R_t = self.sr_2_out_block(out2_R)

        return F_t,R_t
    def recu_unit(self, input, hidden_f):

        combined = torch.cat((input, hidden_f), 1)
        first_f = self.in_block(combined)
        hidden_all = self.feat_ext(first_f) + first_f
        hidden_low = self.feat_ext_low(hidden_all)
        F_t, R_t = self.spectral_residual(hidden_low)
        return hidden_low, F_t, R_t

    def forward(self, input, pos):
        self.device = input.device
        if pos == 0:
            device = input.device
            self.n, self.b, self.h, self.w = input.shape
            self.hidden_f = self.initHidden()
            self.hidden_f = self.hidden_f.to(device)

        self.hidden_f, self.F_t, self.R_t = self.recu_unit(input, self.hidden_f)
        return self.F_t,self.R_t


    def initHidden(self):

        # 避免随机生成的 H0 干扰后续结果
        # return torch.zeros(size=(self.n,self.feat_num,self.b,self.h,self.w))

        return torch.zeros(size=(self.n, self.feat_num,self.h, self.w))