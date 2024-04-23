from models.blocks import *
import torch
from torch import nn
import numpy as np
from .DConv.atten.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION

class BasicConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True):
        super(BasicConv3d, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=bias))

class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.act = act

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []

        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                    reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        # return concatenated hidden states
        return torch.cat(h_time, dim=2)

    def extra_repr(self):
        return 'act={}'.format(self.act)

class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels*2, k, s, p, bn=bn), act=act)


class QRNN3D(nn.Module):
    def __init__(self,in_channels, channels, num_half_layer, QRNNConv3D=None,
                 is_2d=False, has_ad=True, bn=True, act='tanh'):
        super(QRNN3D, self).__init__()
        # Encoder
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in range(num_half_layer):
            if is_2d:
                layer = QRNNConv3D(in_channels, channels, k=(1, 3, 3), s=1, p=(0, 1, 1), bn=bn, act=act)
            else:
                layer = QRNNConv3D(in_channels, channels, bn=bn, act=act)

            self.layers.append(layer)

    def forward(self, x, reverse=False):
        if not self.enable_ad:
            num_layer = len(self.layers)
            for i in range(num_layer):
                x = self.layers[i](x)
            # x = self.layers[-1](x)
            return x
        else:
            num_layer = len(self.layers)
            for i in range(num_layer - 1):
                x = self.layers[i](x, reverse=reverse)
                reverse = not reverse
            # x = self.layers[-1](x, reverse=reverse)
            # reverse = not reverse
            return x, reverse



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
            channels = 64,
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
        self.feat_num_single = self.feat_num // 3
        if self.opt.is_de == 1:
           self.in_channels = self.opt.n_sharp + self.feat_num + 3
        else:
            self.in_channels = self.feat_num+ 3

        self.deconv = DeformableAttnBlock_FUSION(d_model = self.feat_num, n_levels = 3)

        self.feat_ext = nn.Sequential()
        self.feat_ext.append(nn.Conv2d(self.feat_num*3, channels, 3, 1, 1))
        self.feat_ext.append(nn.LeakyReLU(0.1,inplace=True))
        # self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        # self.feat_ext.append(nn.LeakyReLU(0.1,inplace=True))
        # self.feat_ext.append([ResBlock(channels, channels, kernel_size=3, stride=1)
        #  for _ in range(3)])
        if opt.argu == 1:
        # self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        # self.feat_ext.append(nn.ReLU(inplace=True))
            self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
            self.feat_ext.append(nn.LeakyReLU(0.1,inplace=True))
        self.feat_ext.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.feat_ext.append(nn.LeakyReLU(0.1,inplace=True))
        self.feat_ext.append(nn.Conv2d(channels, self.feat_num, 3, 1, 1))

        self.feat_ext1 = nn.Sequential()
        self.feat_ext1.append(nn.Conv2d(self.feat_num+1,  channels, 3, 1, 1))
        self.feat_ext1.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_ext1.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.feat_ext1.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_ext1.append(nn.Conv2d(channels,feat_num, 3, 1, 1))

        self.feat_ext2 = nn.Sequential()
        self.feat_ext2.append(nn.Conv2d(self.feat_num+1, channels, 3, 1, 1))
        self.feat_ext2.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_ext2.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.feat_ext2.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_ext2.append(nn.Conv2d(channels, feat_num, 3, 1, 1))


        self.feat_ext3 = nn.Sequential()
        self.feat_ext3.append(nn.Conv2d(self.feat_num+1, channels, 3, 1, 1))
        self.feat_ext3.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_ext3.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.feat_ext3.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_ext3.append(nn.Conv2d(channels, feat_num, 3, 1, 1))

        self.feat_forth = nn.Sequential()
        self.feat_forth.append(nn.Conv2d(feat_num*2, channels, 3, 1, 1))
        self.feat_forth.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_forth.append(nn.Conv2d(channels, self.feat_num, 3, 1, 1))


        self.feat_back = nn.Sequential()
        self.feat_back.append(nn.Conv2d(feat_num*2, channels, 3, 1, 1))
        self.feat_back.append(nn.LeakyReLU(0.1, inplace=True))
        self.feat_back.append(nn.Conv2d(channels, self.feat_num, 3, 1, 1))
        # self.feat_ext.append(QRNN3D(in_channels, channels, feat_layer, QRNNConv3D=QRNNConv3D,has_ad=has_ad, is_2d=is_2d))
        # # self.feat_ext.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # # self.feat_ext.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # self.feat_ext.append(QRNN3D(channels, feat_num, feat_layer, QRNNConv3D=QRNNConv3D,has_ad=has_ad, is_2d=is_2d))

        self.forth = nn.Sequential()
        self.forth.append(nn.Conv2d(self.feat_num*2, channels, 3, 1, 1))
        self.forth.append(nn.LeakyReLU(0.1, inplace=True))
        if opt.argu == 1:
            self.forth.append(nn.Conv2d(channels, channels, 3, 1, 1))
            self.forth.append(nn.LeakyReLU(0.1, inplace=True))
        self.forth.append(nn.Conv2d(channels, out_channels, 3, 1, 1))
        # self.forth.append(QRNN3D(feat_num, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # # self.forth.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # # self.forth.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # self.forth.append(QRNN3D(channels, out_channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))

        self.key = nn.Sequential()
        self.key.append(nn.Conv2d(self.feat_num, channels, 3, 1, 1))
        self.key.append(nn.LeakyReLU(0.1, inplace=True))
        if opt.argu == 1:
            self.key.append(nn.Conv2d(channels, channels, 3, 1, 1))
            self.key.append(nn.LeakyReLU(0.1, inplace=True))
        self.key.append(nn.Conv2d(channels, out_channels, 3, 1, 1))
        # self.key.append(QRNN3D(feat_num, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # # self.key.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # # self.key.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # self.key.append(QRNN3D(channels, out_channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))

        self.back = nn.Sequential()
        self.back.append(nn.Conv2d(self.feat_num*2, channels, 3, 1, 1))
        self.back.append(nn.LeakyReLU(0.1, inplace=True))
        if opt.argu == 1:
            self.back.append(nn.Conv2d(channels, channels, 3, 1, 1))
            self.back.append(nn.LeakyReLU(0.1, inplace=True))
        self.back.append(nn.Conv2d(channels, out_channels, 3, 1, 1))
        # self.back.append(QRNN3D(feat_num, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # # self.back.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # # self.back.append(QRNN3D(channels, channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))
        # self.back.append(QRNN3D(channels, out_channels, feat_layer, QRNNConv3D=QRNNConv3D, has_ad=has_ad, is_2d=is_2d))

    def recu_unit(self, input, hidden_f):
        # print(input.shape)
        # print(hidden_f.shape)
        # combined = torch.cat((input, hidden_f), 1)
        # combined = combined.unsqueeze(dim=1)
        f = input[:, 2:3]
        c = input[:, 1:2]
        r = input[:, 0:1]
        f_h,c_h,r_h = torch.concat((f,hidden_f),dim=1),torch.concat((c,hidden_f),dim=1),torch.concat((r,hidden_f),dim=1)

        hidden_f = self.feat_ext1(f_h)
        hidden_c = self.feat_ext2(c_h)
        hidden_r = self.feat_ext3(r_h)

        hidden_all = torch.concat((hidden_f, hidden_c, hidden_r), dim=1)
        hidden_all = self.feat_ext(hidden_all)

        hidden_c_f = torch.concat((hidden_f,hidden_c),dim=1)
        hidden_c_r = torch.concat((hidden_r, hidden_c), dim=1)

        F_t = self.forth(hidden_c_f)
        R_t = self.back(hidden_c_r)
        # hidden_r_c_f_fusion = torch.concat((hidden_f, hidden_c, hidden_r), dim=1)
        # hidden_r_c_f = self.fusion(hidden_r_c_f_fusion)
        hidden_f,hidden_c,hidden_r = hidden_f.unsqueeze(dim=1),hidden_c.unsqueeze(dim=1),hidden_r.unsqueeze(dim=1)

        hidden_r_c_f = torch.concat((hidden_f,hidden_c,hidden_r),dim=1)
        hidden_c_fusion = self.deconv(hidden_r_c_f,hidden_r_c_f)

        C_t = self.key(hidden_c_fusion)


        return hidden_all, F_t, C_t, R_t

    def forward(self, input, pos):

        if pos == 0:
            device = input.device
            # self.n, self.c, self.b,self.h, self.w = input.shape
            self.n, self.b, self.h, self.w = input.shape
            self.hidden_f = self.initHidden()
            self.hidden_f = self.hidden_f.to(device)
            self.hidden_f,self.F_t, self.C_t, self.R_t = self.recu_unit(input,self.hidden_f)
        else:
            # print(self.hidden_f.shape)
            self.hidden_f, self.F_t, self.C_t, self.R_t = self.recu_unit(input, self.hidden_f)


        return self.F_t, self.C_t, self.R_t

    def initHidden(self):

        # 避免随机生成的 H0 干扰后续结果
        # return torch.zeros(size=(self.n,self.feat_num,self.b,self.h,self.w))

        return torch.zeros(size=(self.n, self.feat_num,self.h, self.w))