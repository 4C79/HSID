from models.blocks import *
import torch
from torch import nn
import numpy as np
from .HSID.networks import *
from .AODN.AODN import *
from .ENCAM.ENCAM import *
from .Partial_Dnet.Partial_Dnet import *
from .Memnet.memnet import *
from .Denet.denet import *
from .HSIE.hsi_lptn_model import *
from .SST.SST import *
from .DPHSIR.GRUNet import GRUnet
from .QRNN3D.qrnn3d import qrnn3d
from .SERT.SERT import SERT


def denoiser(net_name,k):
    if net_name is None:
        return None
    if net_name == "Denet":
        return DeNet(in_channels=k*2 +1)
    if net_name == "Resnet":
        return resnet()
    if net_name == "Memnet":
        return MemNet(k*2 +1,64,1,6,6)
    if net_name == "Dncnn":
        return DnCNN(image_channels = k*2 +1)
    if net_name == "HSID":
        return HSID(in_channels=2 * k)
    if net_name == "ENCAM":
        return ENCAM().cuda()
    if net_name == "Partial_Dnet":
        return PartialDNet(k = 2 * k).cuda()
    if net_name == "AODN":
        return AODN(K=k*2)
    if net_name == "HSIE":
        return HSIRDNECA_LPTN_FUSE(k*2)
    if net_name =="SST":
        net = SST(inp_channels=31, dim=90,
                  window_size=8,
                  depths=[6, 6, 6, 6, 6, 6],
                  num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2)
        net.use_2dconv = True
        net.bandwise = False
        return net
    if net_name == "SERT":
        net = SERT(inp_channels=31, dim=96, window_sizes=[8, 8, 8], depths=[6,6,6], num_heads=[6, 6, 6],
                   split_sizes=[1, 2, 4], mlp_ratio=2, weight_factor=0.1, memory_blocks=128,
                   down_rank=8)  # 16,32,32

        net.use_2dconv = True
        net.bandwise = False
        return net

    if net_name == "DPHSIR":
        return GRUnet(in_ch=1, out_ch=1, use_noise_map=False, bn=False)
    if net_name == "QRNN3D":
        return qrnn3d()



class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=24, out_channels = 1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = nn.Sequential()
        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class resnet(nn.Module):
    def __init__(
            self,
            n_ch = 64,
            res_depth=3,
            hg_depth=2,
            relu_type='leakyrelu',
            norm_type='bn',
            att_name='spar'
    ):
        super(resnet, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}

        self.refine = nn.ModuleList()
        self.refine.append(ConvLayer(3, n_ch, 3, 1))
        for i in range(res_depth):
            channels = n_ch
            self.refine.append(ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs))
        self.refine.append(ConvLayer(channels, 1, 3, 1))
        self.refine = nn.Sequential(*self.refine)
    def forward(self,combined):
        DN = self.refine(combined)
        return DN

class REFINENet(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body
        - up_res_depth: depth of residual layers in each upsample block

    """

    def __init__(
            self,
            opt = None
    ):
        super(REFINENet, self).__init__()
        # nrargs = {'norm_type': norm_type, 'relu_type': relu_type}
        #
        # ch_clip = lambda x: max(min_ch, min(x, max_ch))
        #
        # down_steps = int(np.log2(in_size // min_feat_size))
        # up_steps = int(np.log2(out_size // min_feat_size))
        # n_ch = ch_clip(max_ch // int(np.log2(in_size // min_feat_size) + 1))
        #
        # hg_depth = int(np.log2(64 / bottleneck_size))
        # # ------------ define feature extraction layers --------------------
        #
        # self.refine = nn.ModuleList()
        # self.refine.append(ConvLayer(3, n_ch, 3, 1))
        # for i in range(res_depth + 3 - down_steps):
        #     channels = ch_clip(n_ch)
        #     self.refine.append(ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs))
        # self.refine.append(ConvLayer(channels, 1, 3, 1))
        # self.refine = nn.Sequential(*self.refine)
        self.opt = opt
        self.refine = denoiser(self.opt.refine,self.opt.k)

    def forward(self,R_res,F_res,C,noise_map=None,noise_map_y=None):
        # combined = torch.concat((F_res, C, R_res), dim=1)
        # DN = self.refine(combined)
        combined = torch.concat((F_res, R_res), dim=1)

        if self.opt.refine == "Partial_Dnet":
            DN = self.refine(C, combined, noise_map, noise_map_y)

        elif self.opt.refine =="AODN":
            combined = combined.unsqueeze(dim=1)
            DN = self.refine(C, combined)

        else:
            DN = self.refine(C,combined)


        return DN

